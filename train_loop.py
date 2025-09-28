# %%
import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras.layers import *
from qkeras import *

from keras.callbacks import CSVLogger

import os
import random
from datetime import datetime
import logging
import csv
import time
import numpy as np # Added for seeding

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG.txt"),
        logging.StreamHandler()
    ]
)
logging.info("--- Script Execution Started ---")
# -----------------------------

pi = 3.14159265359
maxval=1e9
minval=1e-9

# %%
from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator
from losses.loss import custom_loss
from models.SoftQuantizeLayer import SoftQuantizeLayer
from models.AnnealingScheduler import AnnealingScheduler

# %%
# Transformer model
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

def create_vit_model(input_shape=(13,21,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     final_outputs=14):
  inp = layers.Input(shape=input_shape, name="raw_input")
  q_out = SoftQuantizeLayer(
      n_bits=2,
    #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
      threshold_offset=80,
      initial_thresholds=[629.6, 1121.1, 2056.3],
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
# Dataset and TFRecord paths
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = "/depot/cms/users/das214/datasets/dataset_3sr/dataset_3sr_16x16_50x12P5_parquets/"
logging.info(f"Dataset base directory: {dataset_base_dir}")

NOISE_MU = 0.0
NOISE_SIGMA = 80.0 # e-
logging.info(f"Noise parameters: MU={NOISE_MU}, SIGMA={NOISE_SIGMA} e-")

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", f"2t_N_{NOISE_MU}mu_{NOISE_SIGMA}sig_NoLog_Stdr")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")

os.makedirs(tfrecords_dir_train, exist_ok=True)
os.makedirs(tfrecords_dir_val, exist_ok=True)

# %%
class SoftQuantizeLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_filepath, layer_name="soft_quantizer_output"):
        super().__init__()
        self.log_filepath = log_filepath
        self.layer_name = layer_name
        self.header_written = False

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        try:
            layer = self.model.get_layer(self.layer_name)
            if not hasattr(layer, 'n_bits'):
                 logging.warning(f"Layer '{self.layer_name}' is not a SoftQuantizeLayer. Skipping logging.")
                 return
        except ValueError:
            logging.warning(f"Layer '{self.layer_name}' not found in the model. Skipping logging.")
            return

        if not self.header_written:
            num_levels = layer.num_levels
            num_thresholds = num_levels - 1
            header = ['epoch', 'k']
            header.extend([f'level_{i}' for i in range(num_levels)])
            header.extend([f'threshold_{i}' for i in range(num_thresholds)])
            header.append('raw_first_level')
            header.extend([f'raw_log_level_delta_{i}' for i in range(num_levels - 1)])
            header.append('raw_first_threshold')
            if hasattr(layer, 'log_threshold_deltas'):
                header.extend([f'raw_log_threshold_delta_{i}' for i in range(num_thresholds - 1)])
            with open(self.log_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self.header_written = True
        k_val = layer.k.numpy().item()
        levels = layer.levels.numpy().tolist()
        thresholds = layer.thresholds.numpy().tolist()
        first_level = layer.first_level.numpy().item()
        log_level_deltas = layer.log_level_deltas.numpy().tolist()
        first_threshold = layer.first_threshold.numpy().item()
        row_data = [epoch, k_val]
        row_data.extend(levels)
        row_data.extend(thresholds)
        row_data.append(first_level)
        row_data.extend(log_level_deltas)
        row_data.append(first_threshold)
        if hasattr(layer, 'log_threshold_deltas'):
            log_threshold_deltas = layer.log_threshold_deltas.numpy().tolist()
            row_data.extend(log_threshold_deltas)
        with open(self.log_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

def main(seed):
    # Set all random seeds for reproducibility for this specific run
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    fingerprint = '%08x' % random.randrange(16**8)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info(f"--- Starting new training run with SEED: {seed} ---")
    logging.info(f"Run Fingerprint: {fingerprint}")
    logging.info(f"Run Timestamp:   {timestamp}")
    
    # --- MODEL CREATION AND COMPILATION (MOVED INSIDE MAIN) ---
    logging.info("Creating Vision Transformer (ViT) model...")
    model_params = {
        'input_shape': (16,16,2),
        'patch_size': (3,4),
        'embed_dim': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_layers': 4,
        'dropout': 0.1,
        'final_outputs': 14
    }
    model = create_vit_model(**model_params)
    logging.info(f"Model created with parameters: {model_params}")
    model.summary(print_fn=logging.info)

    logging.info("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3, clipnorm=1.0),
        loss=custom_loss,
    )
    logging.info("Model compiled with Nadam optimizer and custom_loss.")
    # --- END OF MODEL CREATION BLOCK ---

    # --- DATA GENERATORS AND CALLBACKS ---
    logging.info("Creating data generators from TFRecords...")
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= tfrecords_dir_val,
        shuffle=True,
        seed=seed,  # Use dynamic seed
        quantize=False,
    )
    logging.info("Validation generator created.")
    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = tfrecords_dir_train,
        shuffle=True,
        seed=seed, # Use dynamic seed
        quantize=False,
    )
    logging.info("Training generator created.")

    os.makedirs("trained_models", exist_ok=True)
    base_dir = f'/home/das214/work/users/das214/SmartPixels/SoftQuantize/trained_models/2t_N_{NOISE_MU}mu_{NOISE_SIGMA}sig_NoLog&Stdr/Transformer_model-{fingerprint}-checkpoints'
    logging.info(f"Base output directory: {base_dir}")
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # CRITICAL FIX: Changed .hdf5 to .weights.h5
    checkpoint_filepath = os.path.join(checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')
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
    all_callbacks = [mcp, csv_logger, scheduler_callback, quantizer_logger]

    # --- MODEL TRAINING ---
    logging.info("--- Starting model.fit() ---")
    model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=all_callbacks,
            epochs=1000,
            shuffle=False,
            verbose=1
        )
    logging.info("--- Model training finished for this run ---")

if __name__ == "__main__":
    logging.info("Script invoked directly. Starting main execution loop.")
    while True:
        try:
            # Generate a new random seed for each full run attempt
            run_seed = random.randint(0, 2**32 - 1)
            main(seed=run_seed)
            # break # Exit loop if main() completes successfully
        except Exception as e:
            logging.error(f"An exception occurred in main execution: {e}", exc_info=True)
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)

    logging.info("--- Training script completed successfully ---")