import tensorflow_probability as tfp  # noqa: F401  (must precede qkeras import, see HG_Convolution notebook)
from qkeras import *  # noqa: F401,F403

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator

dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

# --- ADC/CSA noise model: noise on time slice 1 (1ns) is i.i.d.; noise on time
# slice 2 (6ns) is correlated with slice 1 via rho (CSA-bandwidth-limited noise) ---
ENC = 80.0          # e- RMS, input-referred
CvG = 58e-6         # V/e-
NOISE_MU = 0.0
NOISE_SIGMA = CvG * ENC * 1000   # mV

fc = 55e6           # CSA bandwidth extracted from simulation, Hz
tau = 1 / (2*np.pi*fc)           # equivalent time constant, first-order system
dt_slice = 5e-9     # time-index 5 = 1ns, time-index 30 = 6ns -> 5ns gap
NOISE_RHO = np.exp(-dt_slice / tau)

print(f"NOISE_MU={NOISE_MU}, NOISE_SIGMA={NOISE_SIGMA:.4f} mV, NOISE_RHO={NOISE_RHO:.4f}")

# correlated noise + contained-cluster filter, using the new contained/{train,test} parquet pool
tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_1_6_noise_corr_contained")

dataset_train_dir = os.path.join(dataset_base_dir, "contained", "train")
dataset_test_dir = os.path.join(dataset_base_dir, "contained", "test")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")

os.makedirs(tfrecords_dir_train, exist_ok=True)
os.makedirs(tfrecords_dir_val, exist_ok=True)

batch_size = 5000
val_batch_size = 5000
train_file_size = len(os.listdir(dataset_train_dir))
val_file_size = len(os.listdir(dataset_test_dir))

start_time = time.time()
validation_generator = OptimizedDataGenerator(
    dataset_base_dir = dataset_test_dir,
    file_type = "parquet",
    data_format = "3D",
    batch_size = val_batch_size,
    file_count = val_file_size,
    to_standardize= False,
    select_contained = True,
    noise = [NOISE_MU, NOISE_SIGMA, NOISE_RHO], #[mean, sigma, rho]
    min_threshold = None,
    max_threshold = None,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    input_shape = (2,16,16),
    transpose = (0,2,3,1),
    shuffle = False,
    files_from_end=True,

    tfrecords_dir = tfrecords_dir_val,
    use_time_stamps = [5,30],
    max_workers = 2
)
print("--- Validation generator %s seconds ---" % (time.time() - start_time))

start_time = time.time()
training_generator = OptimizedDataGenerator(
    dataset_base_dir = dataset_train_dir,
    file_type = "parquet",
    data_format = "3D",
    batch_size = batch_size,
    file_count = train_file_size,
    to_standardize= False,
    select_contained = True,
    noise = [NOISE_MU, NOISE_SIGMA, NOISE_RHO], #[mean, sigma, rho]
    min_threshold = None,
    max_threshold = None,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    input_shape = (2,16,16),
    transpose = (0,2,3,1),
    shuffle = False,

    tfrecords_dir = tfrecords_dir_train,
    use_time_stamps = [5,30],
    max_workers = 2
)
print("--- Training generator %s seconds ---" % (time.time() - start_time))
