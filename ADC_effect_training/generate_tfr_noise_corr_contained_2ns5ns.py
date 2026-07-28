import tensorflow_probability as tfp  # noqa: F401  (must precede qkeras import, see HG_Convolution notebook)
from qkeras import *  # noqa: F401,F403

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator

dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

# --- ADC/CSA noise model: correlated noise across the two selected time slices ---
# time-index step: index 5 = 1ns, index 30 = 6ns -> 0.2 ns/index
# this case: time-index 10 = 2ns, time-index 25 = 5ns -> dt = 3ns
ENC = 80.0          # e- RMS, input-referred
CvG = 58e-6         # V/e-
NOISE_MU = 0.0
NOISE_SIGMA = CvG * ENC * 1000   # mV

fc = 55e6           # CSA bandwidth extracted from simulation, Hz
tau = 1 / (2*np.pi*fc)           # equivalent time constant, first-order system
dt_slice = 3e-9     # time-index 10 = 2ns, time-index 25 = 5ns -> 3ns gap
NOISE_RHO = np.exp(-dt_slice / tau)

print(f"NOISE_MU={NOISE_MU}, NOISE_SIGMA={NOISE_SIGMA:.4f} mV, NOISE_RHO={NOISE_RHO:.4f}")
print(f"tau={tau*1e9:.4f} ns, dt_slice={dt_slice*1e9:.1f} ns")

# correlated noise + contained-cluster filter, time slices 2ns (index 10) and 5ns (index 25)
tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_2_5_noise_corr_contained")

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
    use_time_stamps = [10, 25],
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
    use_time_stamps = [10, 25],
    max_workers = 2
)
print("--- Training generator %s seconds ---" % (time.time() - start_time))
