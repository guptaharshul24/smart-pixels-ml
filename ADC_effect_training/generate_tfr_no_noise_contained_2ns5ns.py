import tensorflow_probability as tfp  # noqa: F401  (must precede qkeras import, see HG_Convolution notebook)
from qkeras import *  # noqa: F401,F403

import os
import sys
import time
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator

dataset_base_dir = '/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'

# No noise injection at all (noise=-1, OptimizedDataGenerator_v3's off-sentinel) --
# everything else identical to generate_tfr_noise_corr_contained_2ns5ns.py: same
# contained-cluster filter, same 2ns/5ns time slices (index 10, index 25).
NOISE = -1

# no-noise + contained-cluster filter, time slices 2ns (index 10) and 5ns (index 25)
tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files_2_5_no_noise_contained")

dataset_train_dir = os.path.join(dataset_base_dir, "contained", "train")
dataset_test_dir = os.path.join(dataset_base_dir, "contained", "test")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")


def main():
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
        noise = NOISE,
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
        noise = NOISE,
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


if __name__ == "__main__":
    # ProcessPoolExecutor (used internally by OptimizedDataGenerator's
    # process_file_parallel) defaults to 'fork' on Linux, which doesn't play
    # well with pyarrow's background threads -- forked workers that then call
    # pd.read_parquet (pyarrow engine) crash with
    # concurrent.futures.process.BrokenProcessPool (observed directly after
    # pyarrow was added to this project-local pixi env; confirmed pyarrow
    # itself works fine single-process, so this is specifically a fork
    # interaction). 'spawn' gives each worker a fresh interpreter instead of
    # an inherited fork state. Requires the `if __name__ == "__main__":` guard
    # (spawn re-imports this file in each worker; without the guard, that
    # re-triggers top-level generator creation recursively).
    multiprocessing.set_start_method('spawn', force=True)
    main()
