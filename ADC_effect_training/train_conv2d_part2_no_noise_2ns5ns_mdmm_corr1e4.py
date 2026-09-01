"""
Part 2 (2ns5ns), no-noise variant: identical to
train_conv2d_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py (non-quantized
Conv2D twin of Part 2.5, das's naming: Conv2D_Max vs. QConv2D_Max; frozen,
hard-digitized campaign-4 thresholds; MDMM angle-collapse protection) except
the input data has no injected noise at all (see
generate_tfr_no_noise_contained_2ns5ns.py --
noise=-1, everything else identical: same contained-cluster filter, same
2ns/5ns time slices). Isolates the cost of the noise model itself for this
architecture, same as the no-noise Part 1.5 rerun did for the ViT.

Reuses campaign 4's median thresholds (computed on the NOISY threshold
search) rather than running a fresh no-noise threshold search -- keeps noise
as the only variable that differs from the noisy Part 2 run. Writes to a
new, separate output tree (part2_conv2d_no_noise/) -- does not touch or
overwrite any existing Part 2 (noisy) output.
"""
# %%
import tensorflow as tf
# Forces deterministic (non-autotuned) cuDNN algorithm selection. Without this,
# repeated CUDNN_STATUS_EXECUTION_FAILED crashes were observed a few dozen
# steps into epoch 1 on this GPU's MIG partition (A100 MIG 7g.40gb) -- reduced
# SM/cache resources on a MIG slice vs. full GPU can make cuDNN's autotuned
# algorithm picks fail at execution time even though they pass selection.
# Disabling XLA auto-jit alone (TF_XLA_FLAGS=--tf_xla_auto_jit=0) did NOT fix
# it, so this isn't an XLA-fusion issue -- it's the cuDNN algorithm itself.
tf.config.experimental.enable_op_determinism()
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
        logging.FileHandler("runLOG_part2_no_noise_2ns5ns.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Part 2 (2ns5ns, non-quantized Conv2D, frozen hard-digitized thresholds, NO NOISE) Script Execution Started ---")

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as campaign 4 / the noisy Part 2) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk (see campaign 4)
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Non-quantized twin of models.models.CreateModel/conv_network/var_network --
# identical to the noisy Part 2's architecture.
def nonquantized_conv_network(var, n_filters=5, kernel_size=3):
    var = SeparableConv2D(
        n_filters, kernel_size,
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    var = Conv2D(
        n_filters, 1,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    return var

def nonquantized_var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    return Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def CreateNonQuantizedModel(shape, output, n_filters, pool_size):
    x_in = Input(shape)
    stack = nonquantized_conv_network(x_in, n_filters=n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size),
        strides=None,
        padding="valid",
        data_format=None,
    )(stack)
    stack = nonquantized_var_network(stack, hidden=16, output=output)
    model = Model(inputs=x_in, outputs=stack)
    return model

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
# source of the frozen thresholds. Deliberately reused as-is (see module
# docstring) so noise is the only variable that differs from the noisy Part 2.
campaign4_dir = os.path.join(dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm")
median_thresholds_path = os.path.join(
    campaign4_dir, "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json")

# Separate sibling output dir -- does not touch the noisy Part 2's part2_conv2d/
part2_output_dir = os.path.join(campaign4_dir, "part2_conv2d_no_noise")

# %%
class AbortOnStuck(tf.keras.callbacks.Callback):
    """Divergence guard only: aborts if val_loss stays > `threshold` for
    `patience` consecutive epochs, or goes non-finite. Same design as the
    noisy Part 2's AbortOnStuck."""
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
# Same floor gate as the noisy Part 2 -- see that script for the full
# calibration note.
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
        part2_output_dir,
        "2t_part2_no_noise_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Part 2 (no-noise) attempt {attempt}/{MAX_RETRIES}, seed={seed}, "
                      f"fingerprint={fingerprint} ===")

        nonquantized_conv = CreateNonQuantizedModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(nonquantized_conv, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_conv2d_max_nonquantized_no_noise")
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

        run_dir = os.path.join(base_dir, f"Conv2D_model-{fingerprint}-checkpoints")
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
        raise RuntimeError(f"Part 2 (no-noise) failed to clear the {GOOD_VAL_LOSS_THRESHOLD} val_loss "
                            f"floor after {MAX_RETRIES} attempts.")

    logging.info("--- Part 2 (no-noise) training complete ---")


if __name__ == "__main__":
    main()
