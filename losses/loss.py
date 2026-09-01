import tensorflow as tf
import tensorflow_probability as tfp


# custom loss function
def custom_loss(y, p_base, minval=1e-9, maxval=1e9, scale = 512):
    """Multivariate-Gaussian NLL over (x, y, cotA, cotB), parameterized via a
    lower-triangular Cholesky factor (scale_tril) predicted per-sample: 4
    diagonal "sigma" terms (Mdia) + 6 off-diagonal covariance terms (Mcov)
    from the model's 14 output columns (interleaved mu/sigma pairs in
    p[:, 0:8], raw covariance terms in p[:, 8:]).

    --- Why Mdia is floored with max(x, 0.0) + minval, and why likelihood is
    clipped to [minval, maxval], investigated 2026-07-30/31 ---

    Both exist for real numerical-stability reasons, not just as arbitrary
    safety pads: this loss surface has UNBOUNDED gradient blowup as the
    predicted width (Mdia) shrinks toward 0 while mu is still wrong (nonzero
    residual). Verified directly: with the clip removed entirely, a raw
    sigma output of -10 alone produces a loss gradient of order 1e29 --
    enough to send every weight to NaN in a single optimizer step. So the
    floor + clip pair is what keeps this loss trainable at all; it is not
    safe to simply delete them.

    The floor's known side effect: tf.math.maximum(x, 0.0) maps the ENTIRE
    negative half-line of raw sigma outputs to the exact same value (0, then
    +minval). If that floor value makes the predicted likelihood underflow
    below `minval`, tf.clip_by_value's gradient is EXACTLY zero outside
    [minval, maxval], so the NLL term contributes no gradient at all. A
    weaker, symmetric version of the same trap exists at the opposite
    extreme (very large positive raw sigma -> near-flat Gaussian ->
    likelihood also underflows), but is far less likely to be hit at typical
    small-magnitude (Glorot-scale) init.

    *** This dead zone is real, but it was NOT the cause of the repeated
    stuck-at-init QConv2D runs (2026-08-25). *** A softplus + leaky-clip +
    global_clipnorm patch to this file was built and tested specifically to
    remove it, and it did NOT fix QConv2D -- cold starts still collapsed to
    a ~15600-16300 plateau. The actual root cause was a QKeras/Keras-3
    incompatibility that silently dropped the gradient for one of each
    layer's kernel_quantizer/bias_quantizer; see models/models.py and
    models/mdmm.py for the fix (TF_USE_LEGACY_KERAS=1). Once that was fixed,
    a cold start under THIS ORIGINAL, UNPATCHED loss trained cleanly to
    val_loss -20864 (fp e61b24cc, 1293 epochs, first attempt, no retries) --
    matching the patched-loss run to within noise, which is why the patch
    was reverted rather than kept. The archived patch and the runs it
    produced are in wrong_qconv_fixes/ (untracked) for reference.

    The non-quantized stuck run this trap WAS observed on (Part 2 no-noise
    2ns5ns attempt 1, fp 692b1b40, frozen at loss_obj ~103620 for 536
    epochs -- 5000 * -log(1e-9) = 103,616, the entire batch pinned at the
    clip floor) remains handled by the seed-retry loop in every train_*.py,
    which is the standing mitigation.
    """

    p = p_base

    mu = p[:, 0:8:2]

    # creating each matrix element in 4x4
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]

    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])

    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril)

    likelihood = dist.prob(y)
    likelihood = tf.clip_by_value(likelihood,minval,maxval)
    NLL = -1*tf.math.log(likelihood)

    return tf.keras.backend.sum(NLL)
