# %%
import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras.layers import *
from qkeras import *

from keras.callbacks import CSVLogger

import os
import random
from datetime import datetime
import logging
import csv
import time
import numpy as np # Added for seeding

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Script Execution Started ---")
# -----------------------------

pi = 3.14159265359
maxval=1e9
minval=1e-9

# %%
from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.SoftQuantizeLayer import SoftQuantizeLayer
from models.AnnealingScheduler import AnnealingScheduler

# %%
# Transformer model
class PatchExtractor(layers.Layer):
  """Extract 2D patches from images."""
  def __init__(self, patch_size=(3,7)):
    super().__init__()
    self.patch_size = patch_size

  def call(self, images):
    patch_h, patch_w = self.patch_size
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, patch_h, patch_w, 1),
        strides=(1, patch_h, patch_w, 1),
        rates=(1,1,1,1),
        padding='VALID'
    )
    patch_dims = tf.shape(patches)[-1]
    patches = tf.reshape(
        patches,
        [batch_size, -1, patch_dims]
    )
    return patches

class PatchEncoder(layers.Layer):
  """Linear embedding + learnable positional encoding."""
  def __init__(self, num_patches, embed_dim):
    super().__init__()
    self.num_patches = num_patches
    self.projection  = layers.Dense(embed_dim)
    self.pos_embed   = tf.Variable(
        initial_value=tf.zeros((1,num_patches,embed_dim)),
        trainable=True,
        name="pos_embedding"
    )

  def call(self, patch_batch):
    projected = self.projection(patch_batch)
    return projected + self.pos_embed

def transformer_encoder(inputs,
                        head_size,
                        num_heads,
                        ff_dim,
                        dropout=0.1):
  x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads,
                                key_dim=head_size,
                                dropout=dropout)(x, x)
  x = layers.Dropout(dropout)(x)
  res = x + inputs
  
  x = layers.LayerNormalization(epsilon=1e-6)(res)
  x = layers.Dense(ff_dim, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1], activation="linear")(x)
  x = layers.Dropout(dropout)(x)
  
  return x + res

def sample_thresholds(seed, low, high, num_thresholds):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(low=low, high=high, size=num_thresholds)
    return sorted(vals.tolist())

# CHANGE: Add strict enforcement only for selected variable scopes.
# REASON: Fail fast if your *target* vars are not matched; don't punish unrelated vars.

class LRMultiplierModel(tf.keras.Model):
    def __init__(self, *args, lr_multipliers=None,
                 require_match_substrings=None,  # vars containing any of these substrings MUST match a key
                 error_on_unmatched_required=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_multipliers = lr_multipliers or {}
        self.require_match_substrings = tuple(require_match_substrings or ())
        self.error_on_unmatched_required = bool(error_on_unmatched_required)

    def _match_key(self, var_name: str):
        best_key = None
        for k in self.lr_multipliers:
            if k in var_name and (best_key is None or len(k) > len(best_key)):
                best_key = k
        return best_key

    def _factor_for(self, var_name: str) -> float:
        key = self._match_key(var_name)
        if key is None:
            # If this var belongs to a required scope, error out with context.
            if any(s in var_name for s in self.require_match_substrings):
                raise KeyError(
                    f"[LRMultiplierModel] No LR multiplier matched required variable:\n"
                    f"  var: {var_name}\n"
                    f"  required substrings: {self.require_match_substrings}\n"
                    f"  available keys: {list(self.lr_multipliers.keys())}\n"
                    f"Hint: check layer names or use a longer/more specific substring."
                )
            # For non-required vars, neutral factor.
            return 1.0
        return float(self.lr_multipliers[key])

    @staticmethod
    def _scale_grad(grad, factor: float):
        if grad is None or factor == 1.0:
            return grad
        if isinstance(grad, tf.IndexedSlices):
            return tf.IndexedSlices(grad.values * factor, grad.indices, grad.dense_shape)
        return grad * factor

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses
            )
        grads = tape.gradient(loss, self.trainable_variables)
        scaled_grads = [self._scale_grad(g, self._factor_for(v.name))
                        for g, v in zip(grads, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(scaled_grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics} | {"loss": loss}

    def verify_lr_map(self, strict_keys=True):
        names = [v.name for v in self.trainable_variables]
        key_hits = {k: 0 for k in self.lr_multipliers}
        for n in names:
            for k in self.lr_multipliers:
                if k in n:
                    key_hits[k] += 1
        missing = [k for k, c in key_hits.items() if c == 0]
        if strict_keys and missing:
            sample = "\n  - ".join(names[:12])
            raise KeyError(
                f"[LRMultiplierModel] LR map keys matched 0 variables: {missing}\n"
                f"First few variable names for debugging:\n  - {sample}\n"
                f"Hint: print full names via model.log_lr_multiplier_assignments()."
            )

    def log_lr_multiplier_assignments(self):
        print("\n[LRMultiplierModel] Per-variable LR multipliers:")
        for v in self.trainable_variables:
            print(f"  {v.name:80s} x{self._factor_for(v.name)}")
        print()






def create_vit_model(input_shape=(13,21,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     final_outputs=14,
                     initial_thresholds=None,
                     threshold_offset=80):
  inp = layers.Input(shape=input_shape, name="raw_input")
  q_out = SoftQuantizeLayer(
      n_bits=2,
    #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
      threshold_offset=threshold_offset,
      initial_thresholds=initial_thresholds,
      trainable_levels=False,
      trainable_thresholds=True,
      initial_k=1.0,
      trainable_k=True,
      name="soft_quantizer_output"
  )(inp)
  patches = PatchExtractor(patch_size=patch_size)(q_out)
  H, W, C = input_shape
  ph, pw  = patch_size
  num_patches = (H // ph) * (W // pw)
  encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)
  x = encoded_patches
  for _ in range(num_layers):
    x = transformer_encoder(x,
                            head_size=embed_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
  x = layers.LayerNormalization(epsilon=1e-6)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(final_outputs, activation='linear')(x)
  # model = keras.Model(inputs=inp, outputs=outputs)

  model = LRMultiplierModel(
        inputs=inp, outputs=outputs,
        lr_multipliers={
            "soft_quantizer_output/threshold_deltas_raw": 1e5,
            # "soft_quantizer_output/level_deltas_raw": 1.0,
            # "soft_quantizer_output/first_level": 1.0,
            # "soft_quantizer_output/log_k": 1.0,
        },
        require_match_substrings=["soft_quantizer_output/threshold_deltas_raw"],
        error_on_unmatched_required=True,
    )
  return model

# %%
# Dataset and TFRecord paths
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = "/depot/cms/users/das214/datasets/dataset_3sr/dataset_3sr_16x16_50x12P5_parquets/"
logging.info(f"Dataset base directory: {dataset_base_dir}")

NOISE_MU = 0.0
NOISE_SIGMA = 80.0 # e-
logging.info(f"Noise parameters: MU={NOISE_MU}, SIGMA={NOISE_SIGMA} e-")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", f"2t_N_{NOISE_MU}mu_{NOISE_SIGMA}sig_NoLog_Stdr")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

os.makedirs(tfrecords_dir_train, exist_ok=True)
os.makedirs(tfrecords_dir_val, exist_ok=True)

# %%
class SoftQuantizeLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_filepath, layer_name="soft_quantizer_output"):
        super().__init__()
        self.log_filepath = log_filepath
        self.layer_name = layer_name
        self.header_written = False

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # --- find layer safely ---
        try:
            layer = self.model.get_layer(self.layer_name)
        except ValueError:
            logging.warning(f"Layer '{self.layer_name}' not found; skipping log.")
            return
        if not hasattr(layer, "num_levels"):
            logging.warning(f"Layer '{self.layer_name}' is not SoftQuantizeLayer; skipping log.")
            return

        # --- gather values (absolute + raw) ---
        num_levels = layer.num_levels
        num_thresholds = num_levels - 1

        k_val = float(layer.k.numpy())
        levels = list(layer.levels.numpy())                  # abs levels: L items
        thresholds = list(layer.thresholds.numpy())          # abs thresholds: B items

        raw_first_level = float(layer.first_level.numpy())   # raw first-level scalar
        raw_level_deltas = list(layer.level_deltas_raw.numpy())          # length L-1
        raw_thr_deltas = list(layer.threshold_deltas_raw.numpy())        # length B
        raw_first_thr_delta = float(raw_thr_deltas[0]) if raw_thr_deltas else float("nan")

        # --- write header once ---
        if not self.header_written:
            header = (
                ["epoch", "k"] +
                [f"level_{i}" for i in range(num_levels)] +
                [f"threshold_{i}" for i in range(num_thresholds)] +
                ["raw_first_level"] +
                [f"raw_level_delta_{i}" for i in range(num_levels - 1)] +
                ["raw_first_threshold_delta"] +
                [f"raw_threshold_delta_{i+1}" for i in range(num_thresholds - 1)]
            )
            with open(self.log_filepath, "w", newline="") as f:
                csv.writer(f).writerow(header)
            self.header_written = True

        # --- row ---
        row = (
            [epoch, k_val] +
            levels +
            thresholds +
            [raw_first_level] +
            raw_level_deltas +
            [raw_first_thr_delta] +
            (raw_thr_deltas[1:] if len(raw_thr_deltas) > 1 else [])
        )
        with open(self.log_filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)

# CHANGE: Group-aware gradient logger with CSV + per-group top-K prints.
# REASON: Compare quantizer vs selected model weights side-by-side.

class GradientLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        sample_source,                # generator / tf.data / callable -> (x, y) / tuple (x, y)
        log_every=1,
        max_vars_per_group=6,
        group_specs=None,            # dict[group_name] = list[str substrings]
        to_csv=None,
    ):
        super().__init__()
        self.sample_source = sample_source
        self.log_every = int(log_every)
        self.max_vars_per_group = int(max_vars_per_group)
        self.group_specs = group_specs or {}   # e.g., {"quant": ["soft_quantizer_output/"], "attn_qkv": ["multi_head_attention/.*(query|key|value)/kernel"]}
        self.to_csv = to_csv
        self._csv_header_written = False
        if self.to_csv:
            os.makedirs(os.path.dirname(self.to_csv), exist_ok=True)

    # ---------- helpers ----------
    def _get_sample(self):
        # callable -> (x,y)
        if callable(self.sample_source):
            out = self.sample_source()
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return out[0], out[1]
            return out

        # (x,y) tuple
        if isinstance(self.sample_source, (tuple, list)) and len(self.sample_source) >= 2:
            return self.sample_source[0], self.sample_source[1]

        # iterable (Sequence/generator/dataset)
        it = iter(self.sample_source)
        batch = next(it)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        return batch

    @staticmethod
    def _norm(g):
        if g is None:
            return None
        if isinstance(g, tf.IndexedSlices):
            g = g.values
        return float(tf.norm(tf.reshape(g, [-1])).numpy())

    def _assign_group(self, var_name: str) -> str:
        # first matching group wins
        for gname, patterns in self.group_specs.items():
            for pat in patterns:
                if pat in var_name:
                    return gname
        return "others"

    # ---------- callback ----------
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.log_every) != 0:
            return

        x, y = self._get_sample()
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.model.compiled_loss(
                y, y_pred, regularization_losses=self.model.losses
            )
        grads = tape.gradient(loss, self.model.trainable_variables)

        # collect rows: (group, name, norm)
        rows = []
        for v, g in zip(self.model.trainable_variables, grads):
            n = self._norm(g)
            if n is None:
                continue
            grp = self._assign_group(v.name)
            rows.append((grp, v.name, n))

        # print top-K per group
        by_group = {}
        for grp, name, n in rows:
            by_group.setdefault(grp, []).append((name, n))
        print(f"\n[GradLogger] Epoch {epoch} — per-group top-{self.max_vars_per_group}:")
        for grp, items in by_group.items():
            items.sort(key=lambda t: t[1], reverse=True)
            print(f"  [{grp}]")
            for name, n in items[:self.max_vars_per_group]:
                print(f"    {name:80s} | norm={n:.4e}")
        print("-" * 80)

        # CSV dump
        if self.to_csv:
            write_header = (not self._csv_header_written) or (not os.path.exists(self.to_csv))
            with open(self.to_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["epoch", "group", "var_name", "grad_norm"])
                    self._csv_header_written = True
                for grp, name, n in rows:
                    w.writerow([epoch, grp, name, f"{n:.6e}"])


class AbortOnStuck(tf.keras.callbacks.Callback):
    """
    Stop training early if val_loss stays > `threshold` : val_loss_threshold
    for `patience` consecutive epochs.
    """
    def __init__(self, threshold=1e5, patience=3):
        super().__init__()
        self.thr = threshold
        self.pat = patience
        self.bad = 0

    def on_epoch_end(self, epoch, logs=None):
        vloss = logs.get("val_loss", np.inf)
        if vloss > self.thr or not np.isfinite(vloss):
            self.bad += 1
            if self.bad >= self.pat:
                print(f"[AbortOnStuck] val_loss {vloss:.1f} ≥ {self.thr} "
                      f"for {self.pat} epochs — aborting run.")
                self.model.stop_training = True
        else:
            self.bad = 0


def main(seed):
    # Set all random seeds for reproducibility for this specific run
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    fingerprint = '%08x' % random.randrange(16**8)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info(f"--- Starting new training run with SEED: {seed} ---")
    logging.info(f"Run Fingerprint: {fingerprint}")
    logging.info(f"Run Timestamp:   {timestamp}")

    num_thresholds = 3
    threshold_offset = 80.0
    thr_low, thr_high = threshold_offset, 2000.0
    random_thresholds = sample_thresholds(seed, thr_low, thr_high, num_thresholds)
    logging.info(f"Initial thresholds (run-seeded): {random_thresholds}")

    
    # --- MODEL CREATION AND COMPILATION (MOVED INSIDE MAIN) ---
    logging.info("Creating Vision Transformer (ViT) model...")
    model_params = {
        'input_shape': (16,16,2),
        'patch_size': (3,4),
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_layers': 4,
        'dropout': 0.1,
        'final_outputs': 14,
        'initial_thresholds': random_thresholds,
        'threshold_offset': threshold_offset
    }
    model = create_vit_model(**model_params)
    logging.info(f"Model created with parameters: {model_params}")
    model.summary(print_fn=logging.info)

    logging.info("Compiling model...")
    model.compile(
        # optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3, clipnorm=1.0),
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss=custom_loss,
    )

    if isinstance(model, LRMultiplierModel):
        model.verify_lr_map(strict_keys=True)
        model.log_lr_multiplier_assignments() 
    
    logging.info("Model compiled with Nadam optimizer and custom_loss.")
    # --- END OF MODEL CREATION BLOCK ---

    # --- DATA GENERATORS AND CALLBACKS ---
    logging.info("Creating data generators from TFRecords...")
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= tfrecords_dir_val,
        shuffle=True,
        seed=seed,  # Use dynamic seed
        quantize=False,
    )
    logging.info("Validation generator created.")
    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = tfrecords_dir_train,
        shuffle=True,
        seed=seed, # Use dynamic seed
        quantize=False,
    )
    logging.info("Training generator created.")

    os.makedirs("trained_models", exist_ok=True)
    base_dir = f'/home/das214/work/users/das214/SmartPixels/SoftQuantize/trained_models/2t_N_rnd_thr_{NOISE_MU}mu_{NOISE_SIGMA}sig_NoLog_Stdr_3p0/Transformer_model-{fingerprint}-checkpoints'
    logging.info(f"Base output directory: {base_dir}")
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # CRITICAL FIX: Changed .hdf5 to .weights.h5
    checkpoint_filepath = os.path.join(checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')
    logging.info(f"Checkpoints will be saved to: {checkpoint_filepath}")

    logging.info("Setting up Callbacks...")
    mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq='epoch'
    )
    csv_log_path = f'{base_dir}/training_log.csv'
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, append=True)
    logging.info(f"Training history will be logged to: {csv_log_path}")
    scheduler_callback = AnnealingScheduler(
        schedule='cosine',
        target_layer_name='soft_quantizer_output',
        initial_k=1.0,
        final_k=67.0,
        verbose=1
    )
    quantizer_log_path = f"{base_dir}/soft_quantizer_state_log.csv"
    quantizer_logger = SoftQuantizeLoggerCallback(
        log_filepath=quantizer_log_path,
        layer_name="soft_quantizer_output"
    )
    logging.info(f"SoftQuantizeLayer state will be logged to: {quantizer_log_path}")
    abort_bad = AbortOnStuck(threshold=1e5, patience=5)
    
    sample_x, sample_y = next(iter(training_generator))
        
    group_specs = {
        # Quantizer internals
        "quant_thr": [
            "soft_quantizer_output/threshold_deltas_raw",
        ],
        "quant_levels": [
            "soft_quantizer_output/first_level",
            "soft_quantizer_output/level_deltas_raw",
        ],
        "quant_k": [
            "soft_quantizer_output/log_k",   # keep only if that var exists in your layer
        ],
    
        # ViT core blocks
        "pos_embed": [
            "pos_embedding:0",
        ],
        "patch_embed": [
            "patch_encoder/dense/kernel:0",
            "patch_encoder/dense/bias:0",
        ],
        "attn_qkv": [
            "multi_head_attention/query/kernel:0",
            "multi_head_attention/key/kernel:0",
            "multi_head_attention/value/kernel:0",
        ],
        "attn_out": [
            "multi_head_attention/attention_output/kernel:0",
            "multi_head_attention/attention_output/bias:0",
        ],
        "ln": [
            "layer_normalization/gamma:0",
            "layer_normalization/beta:0",
        ],
        "mlp": [
            "dense/kernel:0",     # the FFN dense layers inside transformer block
            "dense/bias:0",
            "dense_1/kernel:0",
            "dense_1/bias:0",
            "dense_2/kernel:0",
            "dense_2/bias:0",
            "dense_3/kernel:0",
            "dense_3/bias:0",
            "dense_4/kernel:0",
            "dense_4/bias:0",
        ],
        "head": [
            "dense_5/kernel:0",   # adjust indices if your naming differs
            "dense_5/bias:0",
        ],
    }
    
    gradient_logger = GradientLogger(
        sample_source=(sample_x, sample_y),     # fixed tiny batch = cheap + reproducible
        log_every=1,
        max_vars_per_group=6,
        group_specs=group_specs,
    )
    
    all_callbacks = [mcp, csv_logger, scheduler_callback, quantizer_logger, abort_bad, gradient_logger]

    # --- MODEL TRAINING ---
    logging.info("--- Starting model.fit() ---")
    model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=all_callbacks,
            epochs=1000,
            shuffle=False,
            verbose=1
        )
    logging.info("--- Model training finished for this run ---")

if __name__ == "__main__":
    logging.info("Script invoked directly. Starting main execution loop.")
    while True:
        try:
            # Generate a new random seed for each full run attempt
            run_seed = random.randint(0, 2**32 - 1)
            main(seed=run_seed)
            # break # Exit loop if main() completes successfully
        except Exception as e:
            logging.error(f"An exception occurred in main execution: {e}", exc_info=True)
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)

    logging.info("--- Training script completed successfully ---")