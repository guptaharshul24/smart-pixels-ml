"""
Stage 1.5 (ViT, frozen hard-digitized thresholds) on the pixelAV-matched
dataset. Retrains from scratch with the Stage 1 MDMM campaign's median
thresholds FROZEN and hard-digitized (real ADC-style bucketize, not the
soft/differentiable SoftQuantizeLayer used in Stage 1). Isolates the cost of
freezing the thresholds -- same architecture as the Stage 1 ViT, so any
accuracy drop vs Stage 1 comes purely from no longer optimizing thresholds
during training. Adapted from
ADC_effect_training/train_vit_part1p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py.

Thresholds: [274.0, 620.4, 1528.9] e-, median over the 5-run pixelAV-matched
MDMM campaign (raw_pixelAV_training/run_orchestrator_pixelav_matched_mdmm.py,
2026-09-07/08, all 5 runs non-stuck). Read directly from that campaign's
median_thresholds_pixelav_matched_mdmm.json, not hardcoded, so a future
rerun of the campaign updates this script's behavior automatically.
Deliberately using our own median rather than das's previous-studies
reference ([247.8, 668.4, 1662.8] e-) -- our own is the value the rest of
this dataset's pipeline is built around, matches how every other stage in
this repo consumes its own Stage 1 campaign's output, and das's reference
carries its own unresolved uncertainty-interpretation question (see
smartpixels_regression.md memory).

Noise: trains on the NO-NOISE pixelAV-matched TFR set (TFR_files/2t/), not
the noisy one Stage 1 used to derive the thresholds above. Convention
decided 2026-09-08: for this dataset, only Stage 1 (threshold search) uses
noisy data -- realistic thresholds need realistic noise margins -- but
every stage after that trains no-noise, with those noise-derived thresholds
frozen in. The no-noise TFR set (generate_tfr_pixelav_matched_no_noise.py)
reads from the exact same parquet source as the noisy one (train/test dirs,
untouched), so the underlying 195,000 events are identical between the two
-- only the noise setting differs.

Digitization happens in the data generator (map_to_levels, a real
tf.raw_ops.Bucketize step), not inside the model -- no threshold layer here
at all, just the plain ViT taking already-digitized input.

MDMM (MinCorrConstraint) angle-collapse protection carried over unchanged --
freezing the thresholds does nothing to protect against the collapsed-angle
failure mode during this fresh weight retrain (the whole reason Stage 1
needed MDMM here in the first place, see the non-MDMM pixelAV baseline).
"""
# %%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import keras
from keras.layers import *
from keras.callbacks import CSVLogger

import os
import sys
import json
import random
import time
from datetime import datetime
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_part1p5_pixelav_matched.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Stage 1.5 (pixelAV-matched, frozen hard-digitized thresholds) Script Execution Started ---")

pi = 3.14159265359
maxval = 1e9
minval = 1e-9

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as the Stage 1 campaign) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Transformer model (same architecture as Stage 1) -- no threshold layer here at
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
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t")  # no-noise -- see module docstring
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# Stage 1 MDMM campaign -- the source of the frozen thresholds
rnd_thr_dir = os.path.join(dataset_base_dir, "trained_models_rnd_thr_mdmm")
median_thresholds_path = os.path.join(rnd_thr_dir, "median_thresholds_pixelav_matched_mdmm.json")

# Stage 1.5 output lives under the same rnd_thr_mdmm dir, in its own subfolder
part1p5_output_dir = os.path.join(rnd_thr_dir, "part1p5_vit")

# %%
class AbortOnStuck(tf.keras.callbacks.Callback):
    """Divergence guard only: aborts if val_loss stays > `threshold` for
    `patience` consecutive epochs, or goes non-finite."""
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
# Floor gate: our own Stage 1 MDMM runs on this dataset landed -40.7k to
# -44.4k; das's historical QConv2D reference converges to -15k to -20k;
# campaign 4 (ADC dataset) reached -35,906. -10000 stays comfortably below
# all of those as a lenient stuck-vs-real-convergence gate.
GOOD_VAL_LOSS_THRESHOLD = -10000.0


def main():
    logging.info(f"Loading pixelAV-matched Stage 1 median thresholds from: {median_thresholds_path}")
    med_info = json.load(open(median_thresholds_path))
    fixed_thresholds = med_info["median_thresholds"]
    fixed_levels = med_info["levels"]
    n_runs_stage1 = med_info["n_runs"]
    logging.info(f"Stage 1: {n_runs_stage1} runs -> median thresholds = {fixed_thresholds}, "
                  f"levels = {fixed_levels}")

    base_dir = os.path.join(
        part1p5_output_dir,
        "1t_part1p5_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Stage 1.5 attempt {attempt}/{MAX_RETRIES}, seed={seed}, "
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
        model = MDMM(vit, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_part1p5_pixelav_vit")
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
            "stage1_n_runs": n_runs_stage1,
            "stage1_median_source": median_thresholds_path,
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
        raise RuntimeError(f"Stage 1.5 failed to clear the {GOOD_VAL_LOSS_THRESHOLD} val_loss "
                            f"floor after {MAX_RETRIES} attempts.")

    logging.info("--- Stage 1.5 (pixelAV-matched) training complete ---")


if __name__ == "__main__":
    main()
