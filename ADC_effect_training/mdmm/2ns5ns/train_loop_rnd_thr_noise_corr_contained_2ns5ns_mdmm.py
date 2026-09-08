# %%
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import keras
from keras.layers import *
# NOTE (2026-09-03): `from qkeras import *` was removed here -- it was dead
# weight, this is a pure ViT (SoftQuantizeLayer + MultiHeadAttention/Dense)
# and uses no QKeras symbol. Removing it is hygiene only, NOT the fix for the
# Keras-runtime mismatch that broke this script: qkeras still arrives
# transitively via DG/OptimizedDataGenerator_v3.py (which genuinely needs it),
# and importing it sets TF_USE_LEGACY_KERAS=1 as a side effect. The actual fix
# lives in models/mdmm.py, which now follows tf.keras's own resolution instead
# of re-reading that env var at import time -- see its module docstring for
# the full explanation. This script runs on Keras 3, the same runtime
# campaign 4 (2026-07-13..15) actually used, so reruns stay comparable.

from keras.callbacks import CSVLogger

import os
import sys
import random
import json
import glob
import re
from datetime import datetime
import logging
import csv
import time
import numpy as np # Added for seeding

# DG/losses/models live at the repo root, one level up from this ADC_effect_training/ dir
_repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, _repo_root)

import utils
# Without this, TF grabs a large upfront chunk of GPU memory (not on-demand) on first use --
# fine in isolation, but on this shared 5GB MIG slice it races with a just-killed process's
# not-yet-fully-reclaimed allocation, causing a same-process-restart OOM (observed 2026-06-29).
utils.check_GPU()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG_rnd_thr_noise_corr_contained_2ns5ns_mdmm.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Script Execution Started (correlated noise + contained-cluster filter + 2ns/5ns time slices + 5000-epoch + MDMM angle-spread constraints variant) ---")

pi = 3.14159265359
maxval=1e9
minval=1e-9

# %%
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.SoftQuantizeLayer import SoftQuantizeLayer
from models.AnnealingScheduler import AnnealingScheduler
from models.mdmm import MDMM, MinStdConstraint, MinMadConstraint, MinCorrConstraint

# %%
# Transformer model (identical architecture to train_loop_rnd_thr_noise_corr_contained.py)
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
                     threshold_offset=80):
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
# Dataset and TFRecord paths -- correlated-noise TFRecords built from the contained-cluster
# (original_atEdge==False) parquet pool, time slices 2ns (index 10) and 5ns (index 25)
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

logging.info(f"Dataset base directory: {dataset_base_dir}")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_2_5_noise_corr_contained")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

os.makedirs(tfrecords_dir_train, exist_ok=True)
os.makedirs(tfrecords_dir_val, exist_ok=True)

# %%
# --- RUN COUNT, RESUMABILITY, AND STUCK-RUN HANDLING ---
EPOCHS = 5000
TARGET_RUNS = 5
# No fixed seed pool: the orchestrator draws each run's seed from OS entropy and
# every seed is journaled to the JSONL at run START (status="started"), so the
# JSONL alone is sufficient to resume/audit the campaign. Individual runs stay
# reproducible because fingerprint/init-thresholds derive deterministically from
# the seed.
TIME_STAMPS = [10, 25]   # 2ns and 5ns (0.2 ns/index; index 10 = 2ns, index 25 = 5ns)
STUCK_THRESHOLD = 1e5
STUCK_PATIENCE = 20
ESCAPE_BELOW = 5e4

# --- MDMM output-spread constraints ---
# All 5 plain 1ns6ns runs collapsed to constant angle predictions (predicted std
# ~0.0004 vs true std ~0.53 for cotA). Constrain std(pred) >= 0.8 * true std for
# each regressed parameter (normalized units, true stds measured on TFR_val:
# x 0.4495, y 0.4495, cotA 0.5316, cotB 0.4484). One Lagrange multiplier per
# parameter; x/y start satisfied so their lambdas stay ~0.
# Scale sized against the NLL magnitude (~26k): at scale 1.0 the constraint was
# ~25x too weak -- the scale1 campaign (archived as *_mdmm_scale1) showed lambda
# ratcheting all 5000 epochs (Nadam caps ascent at ~lr/step) while cotA stayed
# collapsed and cotB crept up far too slowly. At 1e4 the damping term is ~3% of
# the NLL at full violation and each unit of lambda is worth 1e4, so effective
# pressure reaches NLL scale within tens of epochs (while k is still soft).
MDMM_SCALE = 1e4
MDMM_DAMPING = 1.0
# GPU UPGRADE (2026-07-13): was capped at 128 because the deterministic second
# forward pass (needed for the constraint) OOM'd at full batch (5000) on the old
# 1g.5gb MIG slice (1/7 compute, 5GB) -- that pass allocates ~3.4-4.7GB on top of
# the ~3.2GB primary training pass, more than the 5GB slice had free. Now running
# on a 7g.40gb slice (full GPU, all 7/7 compute + 40GB), so use the FULL batch:
# exact correlation per epoch instead of a 128-sample estimate, no OOM risk.
MDMM_CONSTRAINT_SAMPLES = None  # None = use the full batch (no subsampling)
# superseded -- std is outlier-gameable: run 438bcf1c kept ~99.7% of predictions
# at the collapsed constant and inflated batch std past target with ~0.3% extreme
# outliers (std is quadratically outlier-sensitive):
# MDMM_MIN_STD = {"x": 0.36, "y": 0.36, "cotA": 0.43, "cotB": 0.36}
# MAD targets = 0.8 * true mean-absolute-deviation of the normalized labels
# (true MADs on TFR_val: x 0.384, y 0.386, cotA 0.458, cotB 0.384). MAD is only
# linearly outlier-sensitive, so the BULK of predictions must spread
# (collaborator-suggested metric, wrapped in the MDMM lambda machinery).
# superseded -- MAD forced dispersion but achieved ~0 correlation with truth on
# both completed MAD-1e4 runs (eabfe9f3, 4c28f1e8: cotA/cotB corr 0.0004/-0.014
# and -0.0006/0.006 respectively, statistically identical to the fully collapsed
# baseline's 0.002/-0.013). An outlier tail can buy dispersion without any
# truth-dependence -- see MinMadConstraint docstring.
# MDMM_MIN_MAD = {"x": 0.31, "y": 0.31, "cotA": 0.37, "cotB": 0.31}
# Truth-aware floor: Pearson corr(pred, true) >= 0.5 per parameter. Healthy runs
# sit at 0.98-0.996 on angles and ~0.99 on x/y (measured on the non-MDMM healthy
# run a43ed7b9), so 0.5 only has to evict the model from zero-correlation
# collapse -- it forbids every cheat seen so far (constant, outlier-salted,
# spread-but-uncorrelated) since none of them can fake correlation with the label.
MDMM_MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
MDMM_OUTPUT_COLUMNS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
MDMM_LABEL_COLUMNS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

trained_models_dir = os.path.join(dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm")
os.makedirs(trained_models_dir, exist_ok=True)
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl")
median_thresholds_path = os.path.join(trained_models_dir, "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json")


def load_events():
    if not os.path.exists(threshold_runs_path):
        return []
    return [json.loads(l) for l in open(threshold_runs_path) if l.strip()]


def append_event(rec):
    with open(threshold_runs_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_collected():
    # completed runs only (records without a status field are legacy = completed)
    return [r for r in load_events() if r.get("status", "completed") == "completed"]


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
        # Resuming a run: file already has a header+rows from before the interruption --
        # don't let on_epoch_end's header_written check re-open (and truncate) it.
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

class MDMMStateLoggerCallback(tf.keras.callbacks.Callback):
    """Per-epoch log of each constraint's lambda and the predicted output spreads
    (computed on one cached validation batch) -- shows the constraints engaging
    (lambda rising while collapsed) and releasing (infeasibility -> 0)."""
    def __init__(self, log_filepath, constraints, sample_inputs, sample_labels, inner_model):
        super().__init__()
        self.log_filepath = log_filepath
        self.constraints = constraints
        self.sample_inputs = sample_inputs
        self.sample_labels = sample_labels
        self.inner_model = inner_model
        self.header_written = False

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        if os.path.exists(self.log_filepath) and os.path.getsize(self.log_filepath) > 0:
            self.header_written = True

    def on_epoch_end(self, epoch, logs=None):
        preds = self.inner_model(self.sample_inputs, training=False).numpy()
        stds = {name: float(preds[:, col].std())
                for name, col in MDMM_OUTPUT_COLUMNS.items()}
        # std kept as an outlier-salting diagnostic (see MinMadConstraint/MinStdConstraint
        # docstrings): a healthy correlated prediction has std close to the true std, so
        # std blowing up relative to the label's true spread flags a gaming attempt again.
        # corr is the constrained metric now.
        corrs = {}
        for name, col in MDMM_OUTPUT_COLUMNS.items():
            p_col = preds[:, col]
            t_col = self.sample_labels[:, MDMM_LABEL_COLUMNS[name]]
            denom = p_col.std() * t_col.std() + 1e-6
            corrs[name] = float(np.mean((p_col - p_col.mean()) * (t_col - t_col.mean())) / denom)
        lmbdas = {c.name: float(c.lmbda.numpy()) for c in self.constraints}

        if not self.header_written:
            header = (["epoch"] +
                      [f"lmbda_{n}" for n in lmbdas] +
                      [f"pred_std_{n}" for n in stds] +
                      [f"pred_corr_{n}" for n in corrs])
            with open(self.log_filepath, "w", newline="") as f:
                csv.writer(f).writerow(header)
            self.header_written = True

        with open(self.log_filepath, "a", newline="") as f:
            csv.writer(f).writerow([epoch] + list(lmbdas.values())
                                   + list(stds.values()) + list(corrs.values()))


class AbortOnStuck(tf.keras.callbacks.Callback):
    """
    Stop training early if val_loss stays > `threshold` for `patience` consecutive epochs.
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
    sample_low, sample_high = 12.0, 160.0
    random_thresholds = sample_thresholds(seed, sample_low, sample_high, num_thresholds)
    thr_low, thr_high = min(random_thresholds), max(random_thresholds)
    logging.info(f"Initial thresholds (random, sampled in [{sample_low}, {sample_high}]): {random_thresholds}")

    append_event({
        "status": "started",
        "run_index": run_index,
        "seed": seed,
        "fingerprint": fingerprint,
        "timestamp": timestamp,
        "init_thresholds": random_thresholds,
        "thr_low": thr_low,
        "thr_high": thr_high,
        "time_stamps": TIME_STAMPS,
        "mdmm": {
            "scale": MDMM_SCALE,
            "damping": MDMM_DAMPING,
            # "min_std": MDMM_MIN_STD,  # superseded by min_mad
            # "min_mad": MDMM_MIN_MAD,  # superseded by min_corr
            "min_corr": MDMM_MIN_CORR,
            "constraint_samples": MDMM_CONSTRAINT_SAMPLES,
        },
    })

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
    vit = create_vit_model(**model_params)
    # superseded std-only version (gamed by outlier-salting, run 438bcf1c):
    # constraints = [
    #     MinStdConstraint(column=MDMM_OUTPUT_COLUMNS[p], min_value=MDMM_MIN_STD[p],
    #                      scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"std_{p}")
    #     for p in ("x", "y", "cotA", "cotB")
    # ]
    # superseded MAD-only version (gamed by outlier-salting -- zero truth
    # correlation despite satisfying the dispersion floor, see const block above):
    # constraints = [
    #     MinMadConstraint(column=MDMM_OUTPUT_COLUMNS[p], min_value=MDMM_MIN_MAD[p],
    #                      scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"mad_{p}")
    #     for p in ("x", "y", "cotA", "cotB")
    # ]
    constraints = [
        MinCorrConstraint(column=MDMM_OUTPUT_COLUMNS[p], label_column=MDMM_LABEL_COLUMNS[p],
                          min_value=MDMM_MIN_CORR[p],
                          scale=MDMM_SCALE, damping=MDMM_DAMPING, name=f"corr_{p}")
        for p in ("x", "y", "cotA", "cotB")
    ]
    model = MDMM(vit, constraints, constraint_samples=MDMM_CONSTRAINT_SAMPLES, name="mdmm_vit")
    logging.info(f"Model created with parameters: {model_params}")
    # old (std campaign): logging.info(f"MDMM constraints: min std {MDMM_MIN_STD} ...")
    # old (MAD campaign): logging.info(f"MDMM constraints: min MAD {MDMM_MIN_MAD} ...")
    logging.info(f"MDMM constraints: min corr {MDMM_MIN_CORR} on output columns "
                 f"{MDMM_OUTPUT_COLUMNS} vs label columns {MDMM_LABEL_COLUMNS} "
                 f"(scale={MDMM_SCALE}, damping={MDMM_DAMPING})")
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

    base_dir = f'{trained_models_dir}/2t_rnd_thr_noise_corr_contained_2ns5ns_mdmm_5000ep_NoLog_Stdr_4p0_ThOf{threshold_offset}_ThL{thr_low}_ThH{thr_high}/Transformer_model-{fingerprint}-checkpoints'
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

    mdmm_log_path = f"{base_dir}/mdmm_state_log.csv"
    sample_inputs, sample_labels = validation_generator[0]
    sample_labels = np.asarray(sample_labels)
    mdmm_logger = MDMMStateLoggerCallback(
        log_filepath=mdmm_log_path,
        constraints=constraints,
        sample_inputs=sample_inputs,
        sample_labels=sample_labels,
        inner_model=vit,
    )
    logging.info(f"MDMM lambdas/spreads will be logged to: {mdmm_log_path}")

    all_callbacks = [mcp, csv_logger, scheduler_callback, quantizer_logger, mdmm_logger, abort_bad]

    # --- Mid-run resume: base_dir/fingerprint/thresholds are all deterministic functions of
    # `seed` alone, so a relaunch with the same seed recomputes the same checkpoints_dir. If a
    # prior (interrupted) attempt already logged epochs here, pick up from the last one instead
    # of retraining from epoch 0.
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
        "status": "completed",
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
    append_event(rec)
    logging.info(f"Run record appended to: {threshold_runs_path} -> {rec}")

    if stuck:
        logging.info(f"[run seed={seed}] STUCK (epochs={epochs_run}, best_val={best_val_loss:.1f})")
    else:
        med, n = running_median()
        logging.info(f"[run seed={seed}] escaped (best_val={best_val_loss:.1f}); running median over {n} runs = {med}")

    return stuck

if __name__ == "__main__":
    import argparse

    # Subprocess-only entry point: one seed per process (GPU memory fully released
    # on exit). Campaign orchestration -- random seed draws, retries, resume,
    # median -- lives in run_orchestrator_2ns5ns_mdmm.py.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run_index', type=int, required=True)
    args = parser.parse_args()
    # Uncaught exceptions (OOM, etc.) crash the subprocess with exit code 1 --
    # the orchestrator detects this via returncode and retries.
    main(seed=args.seed, run_index=args.run_index)
    sys.exit(0)
