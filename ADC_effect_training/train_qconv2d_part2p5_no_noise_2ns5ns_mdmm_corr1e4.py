"""
Part 2.5 (2ns5ns), no-noise variant: compact QKeras-quantized Conv2D (das's
naming QConv2D_Max) on no-noise input, with frozen hard-digitized campaign-4
thresholds and MDMM angle-collapse protection. Identical to
train_qconv2d_part2p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py except the
input data has no injected noise at all (see
generate_tfr_no_noise_contained_2ns5ns.py -- noise=-1, everything else
identical: same contained-cluster filter, same 2ns/5ns time slices).

*** The one thing that makes QConv2D train at all: TF_USE_LEGACY_KERAS=1
(set below, before any import). *** QKeras 0.9.0 requires legacy Keras 2;
under this env's default Keras 3 it silently drops the gradient for one of
each layer's kernel_quantizer/bias_quantizer -- never both -- so the model
cannot learn and every run collapses to a near-constant prediction. That
single bug accounts for the entire historical 0/20+ QConv2D failure record
(noisy and no-noise, cold-start and warm-start, across every loss function,
learning rate and MDMM scale tried). See models/models.py and
models/mdmm.py for the verification and the import-side changes.

Result under the fix: fp e61b24cc, attempt 1/10 (no retries), EarlyStopping
at epoch 1293, best_val_loss=-20864.39, pulls sigma 0.81-0.99 on all four
targets, predicted cotA/cotB spread matching truth (std 0.513 vs 0.532 and
0.432 vs 0.448, corr 0.982/0.968) -- i.e. genuine, non-degenerate
predictions, not the collapsed constant every pre-fix attempt produced.

This uses the ORIGINAL, UNPATCHED losses/loss.py and a plain optimizer (no
global_clipnorm). A softplus/leaky-clip "dead-zone fix" to the loss was
built and tested first, before the Keras bug was found; it did not fix
QConv2D on its own, and once the Keras fix was in place an isolation run
proved it was not needed either (this run IS that isolation run). It has
been reverted; the archived patch and the runs it produced live in
wrong_qconv_fixes/ (untracked).

Reuses campaign 4's median thresholds (computed on the NOISY threshold
search) rather than running a fresh no-noise threshold search -- keeps noise
as the only variable that differs from the noisy Part 2.5. Writes to a
separate output tree (part2p5_qconv2d_no_noise/) -- does not touch or
overwrite any existing Part 2.5 (noisy) output.
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

# DG/losses/models live at the repo root, one level up from this ADC_effect_training/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_part2p5_no_noise_2ns5ns.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Part 2.5 (2ns5ns, QConv2D, frozen hard-digitized thresholds, NO NOISE) Script Execution Started ---")

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.models import CreateModel
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as campaign 4 / the noisy Part 2.5) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk (see campaign 4)
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Dataset and TFRecord paths -- NO NOISE, contained-cluster, 2ns/5ns case
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_2_5_no_noise_contained")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# Campaign 4 (Pearson correlation-constraint MDMM, computed on NOISY data) -- the
# source of the frozen thresholds. Reused as-is, same as every other Part 2.5
# no-noise script.
campaign4_dir = os.path.join(dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm")
median_thresholds_path = os.path.join(
    campaign4_dir, "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json")

# Same shared output dir as every other Part 2.5 no-noise attempt --
# fingerprint-based subdirs
# don't collide.
part2p5_output_dir = os.path.join(campaign4_dir, "part2p5_qconv2d_no_noise")

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
# Same floor gate as every other Part 2.5 attempt -- see the noisy Part 2.5
# script for the full calibration note.
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
        part2p5_output_dir,
        "2t_part2p5_no_noise_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Part 2.5 (no-noise) attempt {attempt}/{MAX_RETRIES}, "
                      f"seed={seed}, fingerprint={fingerprint} ===")

        qconv = CreateModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(qconv, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_part2p5_no_noise_qconv2d")
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
            "no_noise": True,
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
        raise RuntimeError(f"Part 2.5 (no-noise) failed to clear the "
                            f"{GOOD_VAL_LOSS_THRESHOLD} val_loss floor after {MAX_RETRIES} attempts.")

    logging.info("--- Part 2.5 (no-noise) training complete ---")


if __name__ == "__main__":
    main()
