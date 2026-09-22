"""
Generates NO-NOISE, containment-filtered TFRecords from the pixelAV-matched
parquet dataset built by filter_subsample_pixelav.py -- the exact same
parquet source (train/test dirs, unmodified) as
generate_tfr_pixelav_matched.py's noisy TFR set, so the underlying 195,000
selected events are identical between the two; only the noise setting
differs. This is deliberate (2026-09-08): thresholds are derived from a
noisy Stage 1 search (realistic ADC margins), but every stage after
threshold-search trains on no-noise data with those thresholds frozen in --
same convention decided for this dataset going forward.

Output directory name (`2t`, no noise suffix) mirrors pixelAV's own
no-noise TFR naming convention exactly
(datasets_16x16x20_charge/.../TFR_files/2t/, parallel to the noisy
variant's own `2t_N_0.0mu_80.0sig_NoLog_Stdr`).

See generate_tfr_pixelav_matched.py's docstring for everything else that
carries over unchanged: select_contained=True as a no-op (already applied
at the parquet stage), use_time_stamps=[0,19] (first/last of pixelAV's 20
raw slices, not our own dataset's [10,25] "2ns/5ns" pair), input_shape/
transpose for reshaping into a (16,16,2) channel-last image.
"""
import tensorflow_probability as tfp  # noqa: F401  (must precede qkeras import, see HG_Convolution notebook)
from qkeras import *  # noqa: F401,F403

import os
import sys
import time
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator

dataset_base_dir = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t")

# SAME parquet source as the noisy TFR generation -- not regenerated, not
# resubsampled. This is what guarantees identical underlying events between
# the noisy and no-noise TFR sets.
dataset_train_dir = os.path.join(dataset_base_dir, "train")
dataset_test_dir = os.path.join(dataset_base_dir, "test")
tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train")
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val")


def main():
    os.makedirs(tfrecords_dir_train, exist_ok=True)
    os.makedirs(tfrecords_dir_val, exist_ok=True)

    batch_size = 5000
    val_batch_size = 5000
    train_file_size = len(os.listdir(dataset_train_dir))
    val_file_size = len(os.listdir(dataset_test_dir))

    print(f"noise=-1 (off); train_file_size={train_file_size}, val_file_size={val_file_size}")

    start_time = time.time()
    validation_generator = OptimizedDataGenerator(
        dataset_base_dir = dataset_test_dir,
        file_type = "parquet",
        data_format = "3D",
        batch_size = val_batch_size,
        file_count = val_file_size,
        to_standardize= False,
        select_contained = True,
        noise = -1,
        min_threshold = None,
        max_threshold = None,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        input_shape = (2,16,16),
        transpose = (0,2,3,1),
        shuffle = False,
        files_from_end=True,

        tfrecords_dir = tfrecords_dir_val,
        use_time_stamps = [0, 19],
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
        noise = -1,
        min_threshold = None,
        max_threshold = None,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        input_shape = (2,16,16),
        transpose = (0,2,3,1),
        shuffle = False,

        tfrecords_dir = tfrecords_dir_train,
        use_time_stamps = [0, 19],
        max_workers = 2
    )
    print("--- Training generator %s seconds ---" % (time.time() - start_time))
    print("--- TFR generation complete ---")


if __name__ == "__main__":
    # See generate_tfr_pixelav_matched.py's __main__ block for why spawn is
    # required here (ProcessPoolExecutor default 'fork' + pyarrow crash).
    multiprocessing.set_start_method('spawn', force=True)
    main()
