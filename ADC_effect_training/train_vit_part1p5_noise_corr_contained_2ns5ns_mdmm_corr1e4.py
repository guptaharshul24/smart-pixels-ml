"""
Part 1.5 (2ns5ns): retrains the ViT from scratch with campaign 4's median
thresholds FROZEN and hard-digitized (real ADC-style bucketize, not the
soft/differentiable SoftQuantizeLayer used in Part 1). Isolates the cost of
freezing the thresholds -- same architecture as Part 1's ViT, so any
accuracy drop vs Part 1 comes purely from no longer being able to keep
optimizing the thresholds during training.

Digitization happens in the data generator (DG.OptimizedDataGenerator_v3's
map_to_levels, das214's validated Part-2 pattern: a true tf.raw_ops.Bucketize
step against fixed thresholds), not inside the model -- the model here has no
threshold layer at all, just the plain ViT taking already-digitized input.

Uses the same MDMM (MinCorrConstraint) angle-collapse protection as campaign
4 -- freezing the thresholds does nothing to protect against the collapsed-
angle-prediction failure mode during this fresh weight retrain (verified: a
first attempt without MDMM here landed cotA/cotB correlation ~0.03/0.05,
the same collapse signature as every pre-MDMM Part 1 run). MDMM is a
training-time-only mechanism (Lagrange multipliers discarded after training,
see models/mdmm.py) -- it does not change the deployed model's architecture,
weights format, or quantization in any way, so it doesn't compromise this
script's role as a stand-in for what actually ships on-chip.
"""
# %%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import keras
from keras.layers import *
from qkeras import *

from keras.callbacks import CSVLogger

import os
import sys
import json
import random
import time
from datetime import datetime
import logging
import numpy as np

# DG/losses/models live at the repo root, one level up from this ADC_effect_training/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_part1p5_2ns5ns.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Part 1.5 (2ns5ns, frozen hard-digitized thresholds) Script Execution Started ---")

pi = 3.14159265359
maxval = 1e9
minval = 1e-9

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as campaign 4) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk (see campaign 4)
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Transformer model (same architecture as Part 1) -- no threshold layer here at
# all; digitization is done upstream by the data generator (digitize=True).
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

def create_vit_model(input_shape=(16,16,2),
                     patch_size=(3,4),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     final_outputs=14):
  inp = layers.Input(shape=input_shape, name="digitized_input")
  patches = PatchExtractor(patch_size=patch_size)(inp)
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
  model = keras.Model(inputs=inp, outputs=outputs)

  return model

# %%
# Dataset and TFRecord paths -- correlated-noise, contained-cluster, 2ns/5ns case
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_2_5_noise_corr_contained_2ns5ns")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# Campaign 4 (Pearson correlation-constraint MDMM) -- the source of the frozen thresholds
campaign4_dir = os.path.join(dataset_base_dir, "trained_models_2_5_noise_corr_contained_2ns5ns_mdmm")
median_thresholds_path = os.path.join(
    campaign4_dir, "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json")

# Part 1.5 output lives under the same campaign dir, in its own subfolder
part1p5_output_dir = os.path.join(campaign4_dir, "part1p5_vit")

# %%
class AbortOnStuck(tf.keras.callbacks.Callback):
    """
    Divergence guard only (das's original design): aborts (sets self.aborted,
    stops training) if val_loss stays > `threshold` for `patience` consecutive
    epochs, or goes non-finite.

    This class used to also carry a "stuck-at-init" escape-window check (val_loss
    hadn't moved from its starting value within an early window) -- removed. That
    check only ever asked "did the loss move," never "is the loss any good," so a
    single noisy dip below the margin counted as a permanent pass even if the run
    immediately reconverged to the same bad plateau (observed directly on Part 2.5's
    QConv2D: fp 4e4c3f5a "escaped" at epoch 1, refroze at val_loss=98980.2 by
    epoch 5, and was accepted as a successful run at epoch 52 with
    best_val_loss=97154.6 -- barely different from the runs that got correctly
    flagged as stuck). The replacement is a post-hoc floor check in the retry
    loop below (`best_val_loss > GOOD_VAL_LOSS_THRESHOLD` => retry) that judges
    the actual achieved loss value instead of its trend, which both EarlyStopping
    and the old escape-check were structurally unable to do -- see session notes
    for the full derivation.
    """
    def __init__(self, threshold=1e5, patience=5):
        super().__init__()
        self.thr = threshold
        self.pat = patience
        self.bad = 0
        self.aborted = False

    def on_epoch_end(self, epoch, logs=None):
        vloss = (logs or {}).get("val_loss", np.inf)

        if vloss > self.thr or not np.isfinite(vloss):
            self.bad += 1
            if self.bad >= self.pat:
                print(f"[AbortOnStuck] val_loss {vloss:.1f} > {self.thr} "
                      f"for {self.pat} epochs -- diverged, aborting attempt.")
                self.aborted = True
                self.model.stop_training = True
        else:
            self.bad = 0


# %%
EPOCHS = 5000
EARLY_STOP_PATIENCE = 100
STUCK_THRESHOLD = 1e5
STUCK_PATIENCE = 10
MAX_RETRIES = 10
# Floor gate: a run is only accepted if best_val_loss crosses into a regime that
# actually reflects real learning, not just "training stopped." See Part 2.5's
# script for the full calibration note (das's historical QConv2D reference
# converges to -15,000 to -20,000; this script's own actual healthy run reached
# -35,906, comfortably clearing this bar).
GOOD_VAL_LOSS_THRESHOLD = -10000.0


def main():
    logging.info(f"Loading campaign 4 median thresholds from: {median_thresholds_path}")
    med_info = json.load(open(median_thresholds_path))
    fixed_thresholds = med_info["median_thresholds"]
    fixed_levels = med_info["levels"]
    n_runs_campaign4 = med_info["n_runs"]
    logging.info(f"Campaign 4: {n_runs_campaign4} runs -> median thresholds = {fixed_thresholds}, "
                  f"levels = {fixed_levels}")

    base_dir = os.path.join(
        part1p5_output_dir,
        "2t_part1p5_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
    )
    os.makedirs(base_dir, exist_ok=True)
    logging.info(f"Base output directory: {base_dir}")

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        seed = random.randint(0, 2**32 - 1)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        fingerprint = '%08x' % random.randrange(16**8)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        logging.info(f"=== Part 1.5 attempt {attempt}/{MAX_RETRIES}, seed={seed}, "
                      f"fingerprint={fingerprint} ===")

        vit = create_vit_model(
            input_shape=(16,16,2),
            patch_size=(3,4),
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_layers=4,
            dropout=0.1,
            final_outputs=14,
        )
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(vit, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_part1p5_vit")
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
            loss=custom_loss,
        )

        validation_generator = OptimizedDataGenerator(
            load_from_tfrecords_dir=tfrecords_dir_val,
            shuffle=True,
            seed=seed,
            quantize=False,
            digitize=True,
            digitize_thresholds=fixed_thresholds,
            digitize_levels=fixed_levels,
        )
        training_generator = OptimizedDataGenerator(
            load_from_tfrecords_dir=tfrecords_dir_train,
            shuffle=True,
            seed=seed,
            quantize=False,
            digitize=True,
            digitize_thresholds=fixed_thresholds,
            digitize_levels=fixed_levels,
        )

        run_dir = os.path.join(base_dir, f"Transformer_model-{fingerprint}-checkpoints")
        checkpoints_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        checkpoint_filepath = os.path.join(
            checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.weights.h5')
        mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq='epoch'
        )
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, 'training_log.csv'), append=True)
        abort_cb = AbortOnStuck(threshold=STUCK_THRESHOLD, patience=STUCK_PATIENCE)
        early_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1)

        history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=[mcp, csv_logger, abort_cb, early_cb],
            epochs=EPOCHS,
            shuffle=False,
            verbose=1,
        )

        if abort_cb.aborted:
            logging.info(f"Attempt {attempt} aborted (diverged). Retrying with new seed.")
            continue

        best_val_loss = float(min(history.history.get('val_loss', [np.inf])))
        epochs_run = len(history.history.get('val_loss', []))

        if best_val_loss > GOOD_VAL_LOSS_THRESHOLD:
            logging.info(f"Attempt {attempt}: best_val_loss={best_val_loss:.2f} "
                          f"(epochs_run={epochs_run}) doesn't clear the "
                          f"{GOOD_VAL_LOSS_THRESHOLD} floor -- stuck, not a real "
                          f"convergence. Retrying with new seed.")
            continue

        logging.info(f"Attempt {attempt} succeeded: best_val_loss={best_val_loss:.2f}, "
                      f"epochs_run={epochs_run}")

        summary = {
            "fixed_thresholds": fixed_thresholds,
            "fixed_levels": fixed_levels,
            "campaign4_n_runs": n_runs_campaign4,
            "campaign4_median_source": median_thresholds_path,
            "seed": seed,
            "fingerprint": fingerprint,
            "timestamp": timestamp,
            "attempt": attempt,
            "epochs_run": epochs_run,
            "best_val_loss": best_val_loss,
            "final_val_loss": float(history.history["val_loss"][-1]),
            "checkpoint_dir": run_dir,
        }
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        logging.info(f"Summary written to: {os.path.join(run_dir, 'summary.json')}")
        success = True
        break

    if not success:
        raise RuntimeError(f"Part 1.5 failed to clear the {GOOD_VAL_LOSS_THRESHOLD} val_loss "
                            f"floor after {MAX_RETRIES} attempts.")

    logging.info("--- Part 1.5 training complete ---")


if __name__ == "__main__":
    main()
