"""
Stage 2 (non-quantized Conv2D) on the pixelAV-matched dataset. Same
architecture shape as models.models.CreateModel (SeparableConv2D -> Conv2D
-> AvgPool -> 3x Dense) but with every QKeras Q-layer/quantizer swapped for
its plain Keras equivalent -- isolates the cost of shrinking to a
chip-sized architecture, independent of quantization (that's Stage 2.5).
Adapted from
ADC_effect_training/train_conv2d_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py.

Noise: NO-NOISE TFR set (TFR_files/2t/), same convention as Stage 1.5 --
only Stage 1 (threshold search) uses noisy data; every stage after that
trains no-noise with those thresholds frozen in. See
train_vit_part1p5_pixelav_matched_mdmm.py's docstring for the full
rationale.

Thresholds: [274.0, 620.4, 1528.9] e-, median over the pixelAV-matched
Stage 1 MDMM campaign (read from median_thresholds_pixelav_matched_mdmm.json,
not hardcoded).

No run_eagerly needed (unlike Stage 2.5) -- plain SeparableConv2D/Conv2D
have no quantizer forcing eager mode, so this compiles and trains in normal
graph mode.
"""
# %%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import keras
from keras.layers import *
from qkeras import *  # unused directly, but this is what actually exports Model into scope -- neither `keras` nor `keras.layers.*` does in this env
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
        logging.FileHandler("runLOG_part2_pixelav_matched.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Stage 2 (pixelAV-matched, non-quantized Conv2D, frozen hard-digitized thresholds) Script Execution Started ---")

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.mdmm import MDMM, MinCorrConstraint

# --- MDMM output-spread constraints (same config as Stage 1.5 / Stage 2.5) ---
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
MDMM_CONSTRAINT_SAMPLES = None  # full batch -- 40GB slice, no OOM risk
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# %%
# Non-quantized twin of models.models.CreateModel/conv_network/var_network.
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
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t")  # no-noise
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# pixelAV-matched Stage 1 MDMM campaign -- the source of the frozen thresholds
rnd_thr_dir = os.path.join(dataset_base_dir, "trained_models_rnd_thr_mdmm")
median_thresholds_path = os.path.join(rnd_thr_dir, "median_thresholds_pixelav_matched_mdmm.json")

# Stage 2 output lives under the same rnd_thr_mdmm dir, in its own subfolder
part2_output_dir = os.path.join(rnd_thr_dir, "part2_conv2d")

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
        part2_output_dir,
        "1t_part2_fixed_thr_{:.2f}_{:.2f}_{:.2f}".format(*fixed_thresholds),
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
        logging.info(f"=== Stage 2 attempt {attempt}/{MAX_RETRIES}, seed={seed}, "
                      f"fingerprint={fingerprint} ===")

        nonquantized_conv = CreateNonQuantizedModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
        constraints = [
            MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                              min_value=MDMM_MIN_CORR[p],
                              scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
            for p in ("x", "y", "cotA", "cotB")
        ]
        model = MDMM(nonquantized_conv, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_conv2d_max_pixelav")
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
        raise RuntimeError(f"Stage 2 failed to clear the {GOOD_VAL_LOSS_THRESHOLD} val_loss "
                            f"floor after {MAX_RETRIES} attempts.")

    logging.info("--- Stage 2 (pixelAV-matched) training complete ---")


if __name__ == "__main__":
    main()
