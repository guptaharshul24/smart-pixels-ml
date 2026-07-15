"""
Part 3 (2ns5ns): trains the compact QKeras-quantized Conv2D architecture
(models.models.CreateModel -- QSeparableConv2D -> QConv2D -> AvgPool -> 3x
QDense, 4-bit conv / 8-bit dense, matching das214's QConv2D_Max and the
paper's synthesized "Max Conv2D") on campaign 4's frozen, hard-digitized
median thresholds. Isolates the additional cost of shrinking down to a
compact, chip-deployable architecture, on top of Part 2's threshold-freezing
cost -- same frozen/digitized input pipeline as Part 2, different model.

Digitization happens in the data generator (DG.OptimizedDataGenerator_v3's
map_to_levels), identical setup to Part 2 -- no threshold layer in the model.

Uses the same MDMM (MinCorrConstraint) angle-collapse protection as Part 2
and campaign 4. MDMM is training-time-only (Lagrange multipliers discarded
after training, see models/mdmm.py) -- it changes nothing about the deployed
model's architecture, weights, or quantization, so it doesn't compromise this
script's role as a stand-in for what actually ships on-chip. Without it, this
compact model is just as vulnerable to the collapsed-angle-prediction failure
mode as the ViT (confirmed on Part 2's first, non-MDMM attempt).
"""
# %%
import tensorflow as tf
import tensorflow_probability as tfp  # must precede `from qkeras import *` -- see losses.loss import-order note
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
        logging.FileHandler("runLOG_part3_2ns5ns.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Part 3 (2ns5ns, QConv2D, frozen hard-digitized thresholds) Script Execution Started ---")

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.models import CreateModel
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as Part 2 / campaign 4) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk (see campaign 4)
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Dataset and TFRecord paths -- correlated-noise, contained-cluster, 2ns/5ns case
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_1_6_noise_corr_contained_2ns5ns")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# Campaign 4 (Pearson correlation-constraint MDMM) -- the source of the frozen thresholds
campaign4_dir = os.path.join(dataset_base_dir, "trained_models_1_6_noise_corr_contained_2ns5ns_mdmm")
median_thresholds_path = os.path.join(
    campaign4_dir, "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json")

# Part 3 output lives under the same campaign dir, in its own subfolder
part3_output_dir = os.path.join(campaign4_dir, "part3_qconv2d")

# %%
class AbortOnStuck(tf.keras.callbacks.Callback):
    """
    Aborts (sets self.aborted, stops training) on two independent signals so
    the outer retry loop knows to reseed and try again:
    - Divergence: val_loss stays > `threshold` for `patience` consecutive epochs.
    - Stuck-at-init: val_loss hasn't improved from its starting (epoch-0) value
      by more than `escape_margin` within the first `escape_window` epochs.
      This is a ONE-TIME early check, not a sliding window -- an earlier sliding
      "no improvement in the last N epochs" version of this class was found to
      also fire on genuinely-converging runs during ordinary flat patches on the
      way to a good minimum (confirmed on Part 2's actual successful run: it had
      a 20+ epoch flat stretch around epoch 240 despite already having improved
      by +47,000 units from its start -- a sliding check would have wrongly
      discarded it). Checking only against the STARTING value within an early
      fixed window avoids this: a run that's escaped its initial basin at all
      convincingly (Part 2: +16,554 units by epoch 10) is never at risk again,
      while a run still glued to its starting value after the window (QConv2D
      run c9d6337c: total drift ~3.6 units across 78 epochs) is unambiguously
      stuck. Once escaped, this check permanently disables itself -- from then
      on, EarlyStopping alone decides when training is done, since "has this
      converged and plateaued" is its job, not this callback's.
    """
    def __init__(self, threshold=1e5, patience=5, escape_margin=100.0, escape_window=30):
        super().__init__()
        self.thr = threshold
        self.pat = patience
        self.escape_margin = escape_margin
        self.escape_window = escape_window
        self.bad = 0
        self.initial_loss = None
        self.escaped = False
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
                return
        else:
            self.bad = 0

        if self.initial_loss is None:
            self.initial_loss = vloss

        if not self.escaped:
            if vloss < self.initial_loss - self.escape_margin:
                self.escaped = True
            elif epoch >= self.escape_window:
                print(f"[AbortOnStuck] val_loss {vloss:.1f} hasn't improved by "
                      f">{self.escape_margin} from its starting value "
                      f"{self.initial_loss:.1f} within {self.escape_window} epochs "
                      f"-- stuck at init, aborting attempt.")
                self.aborted = True
                self.model.stop_training = True


# %%
EPOCHS = 5000
EARLY_STOP_PATIENCE = 50
STUCK_THRESHOLD = 1e5
STUCK_PATIENCE = 10
STUCK_ESCAPE_MARGIN = 100.0
STUCK_ESCAPE_WINDOW = 30
MAX_RETRIES = 10


def main():
    logging.info(f"Loading campaign 4 median thresholds from: {median_thresholds_path}")
    med_info = json.load(open(median_thresholds_path))
    fixed_thresholds = med_info["median_thresholds"]
    fixed_levels = med_info["levels"]
    n_runs_campaign4 = med_info["n_runs"]
    logging.info(f"Campaign 4: {n_runs_campaign4} runs -> median thresholds = {fixed_thresholds}, "
                  f"levels = {fixed_levels}")

    base_dir = os.path.join(
        part3_output_dir,
        "2t_part3_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Part 3 attempt {attempt}/{MAX_RETRIES}, seed={seed}, "
                      f"fingerprint={fingerprint} ===")

        qconv = CreateModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(qconv, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_part3_qconv2d")
        # run_eagerly=True: QSeparableConv2D's quantizer calls .numpy() internally,
        # which raises NotImplementedError under graph-mode tracing (this QKeras
        # version's quantizer code predates this Keras/TF version's tf.function
        # tracing behavior) -- confirmed via smoke test that model.fit()/predict()
        # both fail under the default compiled/graph path and both succeed under
        # eager. Not needed for the ViT (Part 2) model, which has no QKeras layers.
        # Combined with MDMM's extra full-batch constraint-evaluation forward pass,
        # this makes each step noticeably slower than a purely graph-compiled run --
        # a real cost, watched for on relaunch (see stall investigation note).
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
            loss=custom_loss,
            run_eagerly=True,
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

        run_dir = os.path.join(base_dir, f"QConv2D_model-{fingerprint}-checkpoints")
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
        abort_cb = AbortOnStuck(threshold=STUCK_THRESHOLD, patience=STUCK_PATIENCE,
                                 escape_margin=STUCK_ESCAPE_MARGIN, escape_window=STUCK_ESCAPE_WINDOW)
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
            logging.info(f"Attempt {attempt} aborted (stuck plateau). Retrying with new seed.")
            continue

        best_val_loss = float(min(history.history.get('val_loss', [np.inf])))
        epochs_run = len(history.history.get('val_loss', []))
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
        raise RuntimeError(f"Part 3 failed to escape the stuck plateau after {MAX_RETRIES} attempts.")

    logging.info("--- Part 3 training complete ---")


if __name__ == "__main__":
    main()
