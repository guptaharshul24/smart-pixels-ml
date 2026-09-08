# %%
"""
Stage 1 (ViT, trainable SoftQuantizeLayer threshold search) on the
pixelAV-matched dataset (dataset_3srb_16x16_50x12P5_centeredIncidence --
containment + |cotBeta|<2, 155k train / 40k val, noise=[0,80] e- i.i.d.,
time slices [0,19] -- first and last of pixelAV's 20, NOT the [10,25]
"2ns/5ns" pair used for our own frontend dataset, see
generate_tfr_pixelav_matched.py's docstring).

Adapted from train_vit_part1_rnd_thr_noise_corr_contained_2ns5ns.py.
Differences from that template:
  - Points at the pixelAV-matched TFR set instead of our own frontend one.
  - Threshold init sampling range is ELECTRON-scale (raw pixelAV charge
    units), not mV -- checked directly (2026-09-02): per-pixel nonzero
    charge in slice 0 has p50~82, p90~459 e-; slice 19 has p50~1149,
    p90~3993 e-. sample_low/sample_high set to [50, 3000] accordingly, a
    generic reasonable range spanning both slices' scale (thresholds are
    trainable, so the init just needs to be the right order of magnitude,
    not exact).
  - TARGET_RUNS=5 (bumped from an initial 1-round confirmatory run,
    2026-09-02: das provided a previous-studies reference [248, 668, 1663]
    e-, and our first run's converged thresholds [240, 636, 1589] sat
    consistently below it by ~3-5% / 3-9 sigma of das's reported spread --
    a real, same-direction offset, not run-to-run noise. 5 runs matches
    campaign 4's convention against our own dataset, to get a proper
    median + scatter estimate before concluding anything from the gap.
    Resumable: rerunning this script picks up from
    threshold_runs_pixelav_matched.jsonl's existing entries and only
    launches the remaining runs needed to reach TARGET_RUNS.
  - Independent seed pool, unrelated to the shared 20260627 pool used by
    every campaign against our own dataset (this is a different dataset).
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import keras
from keras.layers import *
from qkeras import *

from keras.callbacks import CSVLogger

import os
import sys
import random
import secrets
import json
import glob
import re
from datetime import datetime
import logging
import csv
import time
import numpy as np

# DG/losses/models live at the repo root, one level up from this raw_pixelAV_training/ dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
utils.check_GPU()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_pixelav_matched_rnd_thr.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Script Execution Started (pixelAV-matched, Stage 1 ViT threshold search, 5-run campaign) ---")

pi = 3.14159265359
maxval=1e9
minval=1e-9

from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.SoftQuantizeLayer import SoftQuantizeLayer
from models.AnnealingScheduler import AnnealingScheduler

# %%
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

def create_vit_model(input_shape=(16,16,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     final_outputs=14,
                     initial_thresholds=None,
                     threshold_offset=0.0):
  inp = layers.Input(shape=input_shape, name="raw_input")
  q_out = SoftQuantizeLayer(
      n_bits=2,
      initial_levels=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
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
  model = keras.Model(inputs=inp, outputs=outputs)

  return model

# %%
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t_N_0.0mu_80.0sig_NoLog_Stdr")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

# %%
EPOCHS = 5000
TARGET_RUNS = 5
# Seeds are drawn fresh at run time (secrets.randbits, see the orchestrator
# loop below), not pre-generated from a fixed RNG seed -- avoids the
# deterministic-seed-pool fingerprint collision documented in
# ADC_effect_training/README.md (two non-MDMM Stage-1 campaigns sharing
# np.random.default_rng(20260627) draw identical seeds at the same
# run_index). Matches the MDMM orchestrators' secrets.randbits(31) pattern.
TIME_STAMPS = [0, 19]   # first and last of pixelAV's 20 raw slices
STUCK_THRESHOLD = 1e5
STUCK_PATIENCE = 20
ESCAPE_BELOW = 5e4

trained_models_dir = os.path.join(dataset_base_dir, "trained_models_rnd_thr")
os.makedirs(trained_models_dir, exist_ok=True)
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_pixelav_matched.jsonl")
median_thresholds_path = os.path.join(trained_models_dir, "median_thresholds_pixelav_matched.json")


def load_collected():
    if not os.path.exists(threshold_runs_path):
        return []
    return [json.loads(l) for l in open(threshold_runs_path) if l.strip()]


def running_median():
    rows = [r for r in load_collected() if not r.get("stuck", False)]
    if not rows:
        return None, 0
    arr = np.array([r["final_thresholds"] for r in rows])
    return np.median(arr, axis=0).tolist(), len(rows)


# %%
class SoftQuantizeLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_filepath, layer_name="soft_quantizer_output"):
        super().__init__()
        self.log_filepath = log_filepath
        self.layer_name = layer_name
        self.header_written = False

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        if os.path.exists(self.log_filepath) and os.path.getsize(self.log_filepath) > 0:
            self.header_written = True

    def on_epoch_end(self, epoch, logs=None):
        try:
            layer = self.model.get_layer(self.layer_name)
        except ValueError:
            logging.warning(f"Layer '{self.layer_name}' not found; skipping log.")
            return
        if not hasattr(layer, "num_levels"):
            logging.warning(f"Layer '{self.layer_name}' is not SoftQuantizeLayer; skipping log.")
            return

        num_levels = layer.num_levels
        num_thresholds = num_levels - 1

        k_val = float(layer.k.numpy().item())
        levels = list(layer.levels.numpy())
        thresholds = list(layer.thresholds.numpy())

        if not self.header_written:
            header = (
                ["epoch", "k"] +
                [f"level_{i}" for i in range(num_levels)] +
                [f"threshold_{i}" for i in range(num_thresholds)]
            )
            with open(self.log_filepath, "w", newline="") as f:
                csv.writer(f).writerow(header)
            self.header_written = True

        row = (
            [epoch, k_val] +
            levels +
            thresholds
        )
        with open(self.log_filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)

class AbortOnStuck(tf.keras.callbacks.Callback):
    """Stop training early if val_loss stays > `threshold` for `patience` consecutive epochs."""
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
                print(f"[AbortOnStuck] val_loss {vloss:.1f} >= {self.thr} "
                      f"for {self.pat} epochs -- aborting run.")
                self.model.stop_training = True
        else:
            self.bad = 0


def main(seed, run_index):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    fingerprint = '%08x' % random.randrange(16**8)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info(f"--- Starting new training run with SEED: {seed} ---")
    logging.info(f"Run Fingerprint: {fingerprint}")
    logging.info(f"Run Timestamp:   {timestamp}")

    num_thresholds = 3
    threshold_offset = 0.0
    # electron-scale (raw pixelAV charge), not mV -- see module docstring
    sample_low, sample_high = 50.0, 3000.0
    random_thresholds = sample_thresholds(seed, sample_low, sample_high, num_thresholds)
    thr_low, thr_high = min(random_thresholds), max(random_thresholds)
    logging.info(f"Initial thresholds (random, sampled in [{sample_low}, {sample_high}] e-): {random_thresholds}")

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
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss=custom_loss,
    )
    logging.info("Model compiled with Nadam optimizer and custom_loss.")

    logging.info("Creating data generators from TFRecords...")
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= tfrecords_dir_val,
        shuffle=True,
        seed=seed,
        quantize=False,
    )
    logging.info("Validation generator created.")
    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = tfrecords_dir_train,
        shuffle=True,
        seed=seed,
        quantize=False,
    )
    logging.info("Training generator created.")

    base_dir = f'{trained_models_dir}/1t_rnd_thr_pixelav_matched_5000ep_ThOf{threshold_offset}_ThL{thr_low}_ThH{thr_high}/Transformer_model-{fingerprint}-checkpoints'
    logging.info(f"Base output directory: {base_dir}")
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_filepath = os.path.join(checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.weights.h5')
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
    abort_bad = AbortOnStuck(threshold=STUCK_THRESHOLD, patience=STUCK_PATIENCE)

    all_callbacks = [mcp, csv_logger, scheduler_callback, quantizer_logger, abort_bad]

    initial_epoch = 0
    if os.path.exists(csv_log_path):
        with open(csv_log_path) as f:
            prior_rows = list(csv.DictReader(f))
        if prior_rows:
            ckpt_files = glob.glob(os.path.join(checkpoints_dir, "weights.*.weights.h5"))
            if ckpt_files:
                latest_ckpt = max(
                    ckpt_files,
                    key=lambda p: int(re.search(r"weights\.(\d+)-", os.path.basename(p)).group(1))
                )
                initial_epoch = int(prior_rows[-1]["epoch"]) + 1
                logging.info(f"Resuming run from epoch {initial_epoch}, loading weights from {latest_ckpt}")
                model.load_weights(latest_ckpt)
            else:
                logging.warning(f"{csv_log_path} has prior rows but no checkpoint files found in "
                                 f"{checkpoints_dir}; starting fresh from epoch 0")

    logging.info("--- Starting model.fit() ---")
    history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=all_callbacks,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            shuffle=False,
            verbose=1
        )
    logging.info("--- Model training finished for this run ---")

    with open(csv_log_path) as f:
        full_log_rows = list(csv.DictReader(f))
    val_losses = [float(r["val_loss"]) for r in full_log_rows] if full_log_rows else [np.inf]
    best_val_loss = float(min(val_losses))
    epochs_run = len(val_losses)
    stuck = best_val_loss >= ESCAPE_BELOW

    final_thresholds, final_levels = None, None
    try:
        with open(quantizer_log_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            final_thresholds = [float(last[f"threshold_{i}"]) for i in range(num_thresholds)]
            final_levels = [float(last[f"level_{i}"]) for i in range(len(random_thresholds) + 1)]
    except Exception as e:
        logging.warning(f"Could not read final thresholds/levels from {quantizer_log_path}: {e}")

    rec = {
        "run_index": run_index,
        "seed": seed,
        "fingerprint": fingerprint,
        "timestamp": timestamp,
        "threshold_offset": threshold_offset,
        "init_thresholds": random_thresholds,
        "thr_low": thr_low,
        "thr_high": thr_high,
        "final_thresholds": final_thresholds,
        "final_levels": final_levels,
        "best_val_loss": best_val_loss,
        "epochs_run": epochs_run,
        "stuck": stuck,
        "time_stamps": TIME_STAMPS,
        "checkpoint_dir": base_dir,
    }
    with open(threshold_runs_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logging.info(f"Run record appended to: {threshold_runs_path} -> {rec}")

    if stuck:
        logging.info(f"[run seed={seed}] STUCK (epochs={epochs_run}, best_val={best_val_loss:.1f})")
    else:
        med, n = running_median()
        logging.info(f"[run seed={seed}] escaped (best_val={best_val_loss:.1f}); running median over {n} runs = {med}")

    return stuck

if __name__ == "__main__":
    import subprocess
    import argparse

    if '--seed' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, required=True)
        parser.add_argument('--run_index', type=int, required=True)
        args = parser.parse_args()
        main(seed=args.seed, run_index=args.run_index)
        sys.exit(0)

    logging.info("Script invoked directly (orchestrator mode). Starting main execution loop.")

    done_seeds = {r["seed"] for r in load_collected()}
    completed_runs = sum(1 for r in load_collected() if not r.get("stuck", False))
    if done_seeds:
        logging.info(f"Resuming: {len(done_seeds)} seed(s) already attempted, "
                      f"{completed_runs} completed (non-stuck) run(s).")

    MAX_OOM_RETRIES = 3
    run_index = len(done_seeds)
    while completed_runs < TARGET_RUNS:
        run_seed = secrets.randbits(31)
        while run_seed in done_seeds:  # ~1e-9 odds, but a collision would confuse the journal
            run_seed = secrets.randbits(31)
        done_seeds.add(run_seed)
        oom_retries = 0
        while True:
            logging.info(f"Launching subprocess for seed={run_seed}, run_index={run_index} "
                          f"(attempt {oom_retries + 1}/{MAX_OOM_RETRIES + 1})")
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 '--seed', str(run_seed), '--run_index', str(run_index)],
                env=os.environ.copy()
            )
            if result.returncode == 0:
                new_rec = next((r for r in load_collected() if r['seed'] == run_seed), None)
                if new_rec is None:
                    logging.error(f"Subprocess exited 0 but no JSONL record for seed={run_seed}; skipping.")
                    break
                stuck = new_rec.get('stuck', False)
                if not stuck:
                    completed_runs += 1
                logging.info(f"--- Completed run {completed_runs}/{TARGET_RUNS} "
                              f"(run_index={run_index}, seed={run_seed}, stuck={stuck}) ---")
                break
            else:
                oom_retries += 1
                if oom_retries > MAX_OOM_RETRIES:
                    logging.error(f"Subprocess failed after {MAX_OOM_RETRIES} retries for "
                                   f"seed={run_seed} (exit code {result.returncode}); skipping seed.")
                    break
                logging.warning(f"Subprocess failed (exit code {result.returncode}) for "
                                 f"seed={run_seed} (retry {oom_retries}/{MAX_OOM_RETRIES}); waiting 30s.")
                time.sleep(30)
        run_index += 1

    med, n = running_median()
    summary = {
        "n_runs": n,
        "median_thresholds": med,
        "levels": [0.0, 1.0, 2.0, 3.0],
        "time_stamps": TIME_STAMPS,
        "note": "median over non-stuck runs; per-run details in threshold_runs_pixelav_matched.jsonl",
    }
    with open(median_thresholds_path, "w") as f:
        json.dump(summary, f, indent=1)
    logging.info(f"--- Training script completed: {n} runs, median thresholds={med} ---")
