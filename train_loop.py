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
import logging  # Import the logging library

from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback
import csv

# --- LOGGING CONFIGURATION ---
# Configure logging to write to a file 'runLOG.txt' and also to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("runLOG.txt"), # Log to a file
        logging.StreamHandler()            # Also log to the console
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
# from models.models import CreateModel # Conv2D model

# %%
# Trasformer model

class PatchExtractor(layers.Layer):
  """Extract 2D patches from images."""
  def __init__(self, patch_size=(3,7)):
    super().__init__()
    self.patch_size = patch_size

  def call(self, images):
    # images: (batch, H, W, C)
    patch_h, patch_w = self.patch_size
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, patch_h, patch_w, 1),
        strides=(1, patch_h, patch_w, 1),
        rates=(1,1,1,1),
        padding='VALID'
    )
    # Now `patches` has shape: 
    #   (batch, H//patch_h, W//patch_w, patch_h*patch_w*C)
    # Flatten the 2D grid of patches => (batch, num_patches, patch_dim)
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
    # Linear projection of patch tokens
    projected = self.projection(patch_batch)
    # Add learnable positional embeddings
    return projected + self.pos_embed

def transformer_encoder(inputs,
                        head_size,
                        num_heads,
                        ff_dim,
                        dropout=0.1):
  # LayerNorm + Multi-head attention
  x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads,
                                key_dim=head_size,
                                dropout=dropout)(x, x)
  x = layers.Dropout(dropout)(x)
  res = x + inputs
  
  # Next LN + feed-forward
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
  # Input
  inp = layers.Input(shape=input_shape, name="raw_input")

  q_out = SoftQuantizeLayer(
      n_bits=2,
      initial_range=[-1.0, 1.0],
      trainable_levels=False,
      trainable_thresholds=True,
      initial_k=1.0,
      trainable_k=True,
      name="soft_quantizer_output"
  )(inp)

  # 1) Extract patches
  patches = PatchExtractor(patch_size=patch_size)(q_out)
  
  # Calculate how many patches we extracted:
  #   (H // patch_h) * (W // patch_w)
  # Must do it statically if possible:
  # e.g. 13//3=4, 21//7=3 => 12 patches total
  H, W, C = input_shape
  ph, pw  = patch_size
  num_patches = (H // ph) * (W // pw)

  # 2) Encode patches (linear projection + positional embedding)
  encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

  # 3) Apply multiple Transformer encoder blocks
  x = encoded_patches
  for _ in range(num_layers):
    x = transformer_encoder(x,
                            head_size=embed_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
  
  # 4) Flatten and final Dense
  x = layers.LayerNormalization(epsilon=1e-6)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(final_outputs, activation='linear')(x)

  # Create model
  model = keras.Model(inputs=inp, outputs=outputs)
  return model

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
# Log model summary
model.summary(print_fn=logging.info)


# %%
logging.info("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3, clipnorm=1.0),
    loss=custom_loss,
)
logging.info("Model compiled with Nadam optimizer and custom_loss.")


# %%
logging.info("--- DATASET CONFIGURATION ---")
dataset_base_dir = "/depot/cms/users/das214/datasets/dataset_3sr/dataset_3sr_16x16_50x12P5_parquets/"
logging.info(f"Dataset base directory: {dataset_base_dir}")

NOISE_MU = 0.0
NOISE_SIGMA = 400.0 # e-
logging.info(f"Noise parameters: MU={NOISE_MU}, SIGMA={NOISE_SIGMA} e-")


tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", f"2t_N_{NOISE_MU}mu_{NOISE_SIGMA}sig")

dataset_train_dir = os.path.join(dataset_base_dir, "train")
dataset_test_dir = os.path.join(dataset_base_dir, "test")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

logging.info(f"Training TFRecords directory: {tfrecords_dir_train}")
logging.info(f"Validation TFRecords directory: {tfrecords_dir_val}")


os.makedirs(dataset_train_dir, exist_ok=True)
os.makedirs(dataset_test_dir, exist_ok=True)
os.makedirs(tfrecords_dir_train, exist_ok=True)
os.makedirs(tfrecords_dir_val, exist_ok=True)


batch_size = 5000
val_batch_size = 5000
train_file_size = len(os.listdir(dataset_train_dir))
val_file_size = len(os.listdir(dataset_test_dir))
logging.info(f"Batch size: {batch_size} | Validation batch size: {val_batch_size}")
logging.info(f"Number of training files: {train_file_size} | Number of validation files: {val_file_size}")


# %%
# Loading pre-generated TFRecords
logging.info("Creating data generators from TFRecords...")
validation_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir= tfrecords_dir_val,
    shuffle=True,
    seed=42,
    quantize=False,
)
logging.info("Validation generator created.")

training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_dir_train,
    shuffle=True,
    seed=42,
    quantize=False,
)
logging.info("Training generator created.")


class SoftQuantizeLoggerCallback(Callback):
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
            

def main():
    fingerprint = '%08x' % random.randrange(16**8)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.info(f"--- Starting new training run ---")
    logging.info(f"Run Fingerprint: {fingerprint}")
    logging.info(f"Run Timestamp:   {timestamp}")
    
    os.makedirs("trained_models", exist_ok=True)

    base_dir = f'/depot/cms/users/das214/SmartPixels/SoftQuantize/trained_models/Transformer_model-{fingerprint}-checkpoints'
    logging.info(f"Base output directory: {base_dir}")

    checkpoints_dir = os.path.join(base_dir, 'checkpoints')

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True) 
    checkpoint_filepath = os.path.join(checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')
    logging.info(f"Checkpoints will be saved to: {checkpoint_filepath}")

    logging.info("Setting up Callbacks...")
    mcp = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq='epoch'
    )

    csv_log_path = f'{base_dir}/training_log.csv'
    csv_logger = CSVLogger(csv_log_path, append=True)
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

    logging.info("--- Starting model training ---")
    history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=all_callbacks,
            epochs=1000,
            shuffle=False,
            verbose=1
        )
    logging.info("--- Model training finished ---")


if __name__ == "__main__":
    import time
    logging.info("Script invoked directly. Starting main execution loop.")
    while True:
        try:
            main()
            break # Exit loop if main() completes successfully
        except Exception as e:
            # Log the full exception traceback for easier debugging
            logging.error(f"An exception occurred during training: {e}", exc_info=True)
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)

    logging.info("--- Training script completed successfully ---")