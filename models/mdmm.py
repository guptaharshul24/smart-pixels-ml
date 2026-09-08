"""
MDMM (Modified Differential Method of Multipliers) for output constraints.

Adapted from das214's TF implementation (github.com/guptaharshul24/mdmm_tf_mnist,
mdmm.py): same Constraint penalty math (lambda * infeasibility + damping *
infeasibility^2 / 2) and the same train_step multiplier trick (gradient *ascent*
on the lambdas via sign flip, descent on everything else).

Differences from the original:
- Keras 3 native (this repo's env resolves keras / tf.keras to Keras 3.14):
  lambdas via add_weight, compute_loss instead of the deprecated compiled_loss,
  id()-based lambda identification (Keras 3 Variables have no .ref()).
- Constraints act on the wrapped model's OUTPUT only (no per-layer python loop),
  so everything stays graph-compilable -- the original forced
  tf.config.run_functions_eagerly(True) globally, which is unaffordable for
  5000-epoch runs.
- The wrapper delegates get_layer/save_weights/load_weights to the inner model so
  existing callbacks (AnnealingScheduler, SoftQuantizeLoggerCallback,
  ModelCheckpoint) and the checkpoint format keep working unchanged. Note: the
  lambdas are NOT saved in checkpoints; a mid-run resume restarts them from 0 and
  they re-grow wherever the constraint is still violated.

Keras runtime (2026-08-25, revised 2026-09-03): this file is shared by every
train_*.py (ViT, non-quantized Conv2D, AND QConv2D), unlike models/models.py
which is QConv2D-only. QKeras 0.9.0 requires legacy Keras 2 (tf_keras) --
running it under Keras 3 was root-caused as the reason QConv2D's per-layer
gradient tracking silently breaks (a layer with both kernel_quantizer and
bias_quantizer only gets a real gradient on one of the two under Keras 3;
both work under tf_keras -- see models/models.py's comment for the full
verification). The ViT and non-quantized Conv2D paths never touch QKeras and
work fine under Keras 3.

The first version of this fix branched on the TF_USE_LEGACY_KERAS env var
here. That was subtly wrong and broke every ViT+MDMM script: `import qkeras`
sets that env var as an import SIDE EFFECT, and qkeras arrives transitively
via DG/OptimizedDataGenerator_v3.py (`from qkeras import quantized_bits`,
line 27, genuinely used by QKeras_data_prep_quantizer). So in a ViT script
the ordering is: tensorflow imported (tf.keras caches to Keras 3, since TF
reads the var only once at its own first import) -> data generator imported
(qkeras sets the var to "1") -> this module imported (sees "1", picks
tf_keras). MDMM then subclassed tf_keras.Model while the script's
tf.keras.optimizers.Nadam stayed Keras 3, and model.compile() died with
"Could not interpret optimizer identifier". Campaign 4 (2026-07-13..15)
predates the conditional entirely, so it was unaffected -- the ViT+MDMM
scripts simply had not been run between 2026-09-01 (when the conditional
landed, commit 33f8cd4) and 2026-09-03 (when this was hit).

Deferring to tf.keras instead removes the ordering dependency completely:
whatever the calling script's tf.keras.* symbols resolve to, MDMM's base
class matches. Verified both paths: a ViT script (no env var set before TF)
gets Keras 3 for MDMM, optimizer and callbacks alike; a QConv2D script (env
var set as its first statement) gets tf_keras for all three. id()-based
lambda identification (see above) is a plain Python builtin, not
Keras-version-specific, and works under either runtime; add_weight/
compute_loss are standard Keras APIs present in both.

Verified end-to-end (2026-08-25): a full cold-start no-noise Part 2.5 run
under this fix (fp 54d7875b) trained cleanly through 170+ epochs with real,
smooth loss improvement (val_loss -14477.61 and still falling, first cold-
start attempt in the whole investigation to clear the -10000 floor gate) and
well-calibrated pulls (sigma 0.84-0.99 on all four targets) -- not just the
isolated single-layer gradient check. MDMM's lambda mechanics (ascent,
id()-based tracking, penalty computation) are confirmed working correctly
under tf_keras through this run: pen_corr_* stayed at 0.0 throughout
(constraint satisfied, no collapse), unlike every pre-fix attempt where it
climbed unboundedly.
"""
import tensorflow as tf
# Follow whatever tf.keras itself resolved to, rather than re-deciding from
# TF_USE_LEGACY_KERAS here. See the "Keras runtime (2026-09-03)" note above:
# reading the env var directly made this module's choice depend on WHEN it was
# imported relative to qkeras (which sets that var as an import side effect),
# which is not something this module can control -- and getting it wrong
# crashes model.compile() with "Could not interpret optimizer identifier".
# tf.keras is resolved once, at TensorFlow's own first import, so deferring to
# it guarantees MDMM's base class always matches the runtime the calling
# script's tf.keras.optimizers/callbacks come from, in every path.
keras = tf.keras
layers = tf.keras.layers


class OutputConstraint(layers.Layer):
    """Base class: a constraint on the model output with its own multiplier."""
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.damping = damping
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda',
            shape=(),
            initializer='zeros',
            trainable=True,
        )

    def fn(self, outputs):
        raise NotImplementedError

    def infeasibility(self, fn_value):
        raise NotImplementedError

    def call(self, outputs):
        inf = self.infeasibility(self.fn(outputs))
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * (l_term + damp_term)


class MinStdConstraint(OutputConstraint):
    """std(outputs[:, column]) >= min_value (hard inequality, no slack).

    Used to forbid the collapsed solution where the network predicts a constant
    for one regression target (observed for cotA/cotB: predicted std ~0.0004 vs
    true std ~0.53). Infeasibility is zero once the spread is achieved, so the
    penalty vanishes at any healthy minimum.

    DEPRECATED for anti-collapse use (kept for record/comparison): std is
    quadratically sensitive to outliers, and the 2ns5ns scale=1e4 campaign
    (run 438bcf1c) showed the network gaming it -- ~99.7% of predictions stayed
    at the collapsed constant while ~0.3% extreme outliers inflated the batch
    std past the target. Use MinMadConstraint (+ MinCorrConstraint) instead.
    """
    def __init__(self, column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.min_value = min_value

    def fn(self, outputs):
        return tf.math.reduce_std(outputs[:, self.column])

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)


class MinMadConstraint(OutputConstraint):
    """mean(|outputs[:, column] - mean|) >= min_value (hard inequality).

    Mean absolute deviation is only *linearly* sensitive to outliers, so the
    "constant bulk + a few extreme outliers" cheat that defeats MinStdConstraint
    barely moves it -- satisfying this requires the *bulk* of predictions to
    spread. (Metric suggested by a collaborator; wrapped in the MDMM lambda
    machinery instead of their fixed 1/(sum+eps) barrier so the penalty still
    vanishes exactly once satisfied.)
    """
    def __init__(self, column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.min_value = min_value

    def fn(self, outputs):
        col = outputs[:, self.column]
        return tf.reduce_mean(tf.abs(col - tf.reduce_mean(col)))

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)


class MinCorrConstraint(OutputConstraint):
    """Pearson corr(outputs[:, column], y_true[:, label_column]) >= min_value.

    Truth-aware: forbids ALL lazy strategies at once (constant, outlier-salted,
    and spread-but-uncorrelated predictions), since only genuine dependence on
    the true value raises the correlation. Requires the MDMM wrapper to pass
    y_true (needs_truth=True).
    """
    needs_truth = True

    def __init__(self, column, label_column, min_value, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.column = column
        self.label_column = label_column
        self.min_value = min_value

    def fn(self, outputs, y_true=None):
        p = outputs[:, self.column]
        t = tf.cast(y_true[:, self.label_column], p.dtype)
        p_c = p - tf.reduce_mean(p)
        t_c = t - tf.reduce_mean(t)
        cov = tf.reduce_mean(p_c * t_c)
        denom = tf.math.reduce_std(p) * tf.math.reduce_std(t) + 1e-6
        return cov / denom

    def infeasibility(self, fn_value):
        return tf.math.maximum(self.min_value - fn_value, 0.0)

    def call(self, outputs, y_true=None):
        inf = self.infeasibility(self.fn(outputs, y_true=y_true))
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        return self.scale * (l_term + damp_term)


class MDMM(keras.Model):
    """Wraps a model; adds constraint penalties to the training loss.

    train_step: loss = compute_loss + sum(constraint penalties); the gradient
    sign is flipped for the lambda variables (ascent) so each multiplier grows
    while its constraint is violated and stops moving once satisfied.
    val_loss stays the plain compiled loss (test_step is untouched), so
    checkpoint filenames and best_val_loss remain comparable to non-MDMM runs.
    """
    def __init__(self, model, constraints, constraint_samples=None, name='MDMM', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = model
        self.constraints_list = list(constraints)
        self._lmbda_ids = {id(c.lmbda) for c in self.constraints_list}
        # Evaluate constraints on only the first N samples of each batch: the
        # deterministic second forward pass otherwise doubles activation memory
        # (OOMs the 5GB MIG slice at batch 5000, since a spread/correlation
        # estimate only needs ~3-6% accuracy at N~128-512 anyway). GPU UPGRADE
        # (2026-07-13): now running on a 7g.40gb slice (full GPU, 40GB), so the
        # callers pass constraint_samples=None (full batch, exact estimate, no
        # OOM risk) -- N-sample subsampling stays available here as a safety
        # valve if this ever runs on a smaller slice again.
        self.constraint_samples = constraint_samples

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_obj = self.compute_loss(x=x, y=y, y_pred=y_pred)
            # Constraints are evaluated on a deterministic (training=False) forward
            # pass: dropout noise inflates the batch std of the outputs by ~15-35%,
            # which silently satisfies a spread constraint even when the underlying
            # deterministic prediction has collapsed to a constant (verified on the
            # ViT: cotA std 0.85 with dropout vs 0.72 without at init, and >0.43 vs
            # 0.21 mid-collapse). Gradients still flow through this second pass.
            x_c = x if self.constraint_samples is None else x[:self.constraint_samples]
            y_c = y if self.constraint_samples is None else y[:self.constraint_samples]
            y_det = self.model(x_c, training=False)
            # pre-MinCorrConstraint version (truth-free constraints only):
            # penalties = {("pen_" + c.name): c(y_det) for c in self.constraints_list}
            penalties = {}
            for c in self.constraints_list:
                if getattr(c, "needs_truth", False):
                    penalties["pen_" + c.name] = c(y_det, y_true=y_c)
                else:
                    penalties["pen_" + c.name] = c(y_det)
            loss = loss_obj + tf.add_n(list(penalties.values()))

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = []
        for grad, var in zip(grads, self.trainable_variables):
            if grad is None:
                continue
            if id(var) in self._lmbda_ids:
                grads_and_vars.append((-grad, var))
            else:
                grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)

        out = {"loss": loss, "loss_obj": loss_obj}
        out.update(penalties)
        return out

    # --- delegation so existing callbacks/checkpoints work on the inner model ---
    def get_layer(self, name=None, index=None):
        return self.model.get_layer(name=name, index=index)

    def save_weights(self, filepath, *args, **kwargs):
        self.model.save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        self.model.load_weights(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
