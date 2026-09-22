"""
Stage 2.5 (QKeras-quantized Conv2D, das's naming QConv2D_Max) on the
pixelAV-matched dataset, frozen hard-digitized thresholds, MDMM
angle-collapse protection. Adapted from
ADC_effect_training/train_qconv2d_part2p5_no_noise_2ns5ns_mdmm_corr1e4.py --
the ALREADY-FIXED QConv2D template. Only dataset paths, threshold source,
and output dir changed; the import block and everything downstream of it is
copied byte-for-byte from that script, deliberately not touched, since this
is the highest-landmine-density part of the whole repo.

*** The one thing that makes QConv2D train at all: TF_USE_LEGACY_KERAS=1
(set below, before any import). *** QKeras 0.9.0 requires legacy Keras 2;
under this env's default Keras 3 it silently drops the gradient for one of
each layer's kernel_quantizer/bias_quantizer -- never both -- so the model
cannot learn and every run collapses to a near-constant prediction. See
models/models.py and models/mdmm.py for the full verification. DO NOT
remove this line or reorder it after any TF/Keras/QKeras import.

Noise: NO-NOISE TFR set (TFR_files/2t/), same convention as Stage 1.5/2 --
only Stage 1 (threshold search) used noisy data to derive these thresholds;
every stage after that trains no-noise with them frozen in.

Thresholds: [274.0, 620.4, 1528.9] e-, median over the pixelAV-matched
Stage 1 MDMM campaign (read from median_thresholds_pixelav_matched_mdmm.json,
not hardcoded) -- same file Stage 1.5 and Stage 2 read from. Reusing the
NOISY Stage 1 search's thresholds for this NO-NOISE training run is the
same convention the ADC no-noise Stage 2.5 script uses (reuses campaign 4's
noisy-derived thresholds) -- keeps noise as the only variable that differs
from what produced the thresholds.

This uses the ORIGINAL, UNPATCHED losses/loss.py and a plain optimizer (no
global_clipnorm) -- confirmed sufficient on its own by the ADC isolation
test (fp e61b24cc), see that script's docstring for the full history.
run_eagerly=True is still required: QSeparableConv2D's quantizer calls
.numpy() internally, which breaks under graph-mode tracing regardless of
dataset.
"""
# %%
# TF_USE_LEGACY_KERAS=1 MUST be set before any TF/Keras/QKeras import -- see
# models/models.py and models/mdmm.py for the full verification and
# rationale. This single line is what makes QConv2D trainable at all.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_probability as tfp  # must precede `from qkeras import *` -- see losses.loss import-order note
import tf_keras as keras
from tf_keras.layers import *
from qkeras import *

from tf_keras.callbacks import CSVLogger

import sys
import json
import random
import time
from datetime import datetime
import logging
import numpy as np

# DG/losses/models live at the repo root, one level up from this raw_pixelAV_training/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_part2p5_pixelav_matched.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Stage 2.5 (pixelAV-matched, QConv2D, frozen hard-digitized thresholds) Script Execution Started ---")

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.models import CreateModel
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as Stage 1.5 / Stage 2) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t")  # no-noise
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# pixelAV-matched Stage 1 MDMM campaign (computed on NOISY data) -- the
# source of the frozen thresholds. Reused as-is, same convention as Stage
# 1.5/2.
rnd_thr_dir = os.path.join(dataset_base_dir, "trained_models_rnd_thr_mdmm")
median_thresholds_path = os.path.join(rnd_thr_dir, "median_thresholds_pixelav_matched_mdmm.json")

# Stage 2.5 output lives under the same rnd_thr_mdmm dir, in its own subfolder
part2p5_output_dir = os.path.join(rnd_thr_dir, "part2p5_qconv2d")

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
        part2p5_output_dir,
        "1t_part2p5_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Stage 2.5 attempt {attempt}/{MAX_RETRIES}, "
                      f"seed={seed}, fingerprint={fingerprint} ===")

        qconv = CreateModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(qconv, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_part2p5_pixelav_qconv2d")
        # run_eagerly=True: QSeparableConv2D's quantizer calls .numpy() internally,
        # which raises NotImplementedError under graph-mode tracing.
        # Plain optimizer, no global_clipnorm: the original loss's hard
        # clip_by_value already bounds the gradient (see losses/loss.py).
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
        raise RuntimeError(f"Stage 2.5 failed to clear the {GOOD_VAL_LOSS_THRESHOLD} val_loss "
                            f"floor after {MAX_RETRIES} attempts.")

    logging.info("--- Stage 2.5 (pixelAV-matched) training complete ---")


if __name__ == "__main__":
    main()
