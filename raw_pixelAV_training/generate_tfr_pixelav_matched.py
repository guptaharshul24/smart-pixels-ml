"""
Generates noisy, containment-filtered TFRecords from the pixelAV-matched
parquet dataset built by filter_subsample_pixelav.py
(/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence/
{train,test}/ -- 195,000 events total, |cotBeta|<2 + containment already
applied at the parquet-filtering stage).

Noise model: plain i.i.d. Gaussian, mu=0.0, sigma=80.0 -- matches the
existing pixelAV noisy TFR variant's own convention exactly
(datasets_16x16x20_charge/.../TFR_files/2t_N_0.0mu_80.0sig_NoLog_Stdr/,
noise=[0.0, 80.0]), in electrons (pixelAV's native charge unit -- this is
raw charge data, no CvG mV conversion applied anywhere upstream). NOT the
same as our own frontend pipeline's noise model (correlated, mV-scale,
derived from an ENC/CvG physical readout model in
generate_tfr_noise_corr_contained_2ns5ns.py) -- that model is specific to
our own already-converted-to-mV charge representation and doesn't apply to
raw pixelAV electron counts.

select_contained=True is passed for consistency with every other
generate_tfr_*.py script's call pattern, even though it's a no-op here:
every row already satisfies original_atEdge==False (containment was already
applied when the parquet files were built), and that column is still
present in the output (no columns were dropped during filtering).

use_time_stamps=[0,19] -- the FIRST and LAST of pixelAV's 20 raw time
slices, NOT the [10,25] "2ns/5ns" pair every _2ns5ns_ script in this repo
uses for our own frontend-effects dataset. Those indices are specific to
our own dataset's own (finer, 10ps-granularity) time binning and are not
meaningful here -- pixelAV's raw charge data only has 20 slices total
(indices 0-19), so index 25 would be out of range. input_shape=(2,16,16),
transpose=(0,2,3,1) stay the same as every other script: purely about
reshaping the 2 selected slices into a (16,16,2) channel-last image, not
tied to which 2 slices were picked. (Column layout verified directly,
2026-09-02: pixelAV is time-major, 5120 = 20*16*16 raw charge columns per
event, same convention as our own dataset's raw source.)

Output directory name mirrors the pixelAV source's own noisy-TFR naming
convention (2t_N_0.0mu_80.0sig_NoLog_Stdr) rather than this repo's own
_2ns5ns_mdmm-style naming, since this is pixelAV-native data, not one of
our frontend-effects variants.
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

NOISE_MU = 0.0
NOISE_SIGMA = 80.0  # electrons -- matches pixelAV's own noisy TFR set exactly

tfrecords_base_dir = os.path.join(dataset_base_dir, "TFR_files", "2t_N_0.0mu_80.0sig_NoLog_Stdr")

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

    print(f"NOISE_MU={NOISE_MU}, NOISE_SIGMA={NOISE_SIGMA} e- (i.i.d., no correlation)")
    print(f"train_file_size={train_file_size}, val_file_size={val_file_size}")

    start_time = time.time()
    validation_generator = OptimizedDataGenerator(
        dataset_base_dir = dataset_test_dir,
        file_type = "parquet",
        data_format = "3D",
        batch_size = val_batch_size,
        file_count = val_file_size,
        to_standardize= False,
        select_contained = True,
        noise = [NOISE_MU, NOISE_SIGMA],
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
        noise = [NOISE_MU, NOISE_SIGMA],
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
    # ProcessPoolExecutor (used internally by OptimizedDataGenerator's
    # process_file_parallel) defaults to 'fork' on Linux, which doesn't play
    # well with pyarrow's background threads -- forked workers that then call
    # pd.read_parquet (pyarrow engine) crash with
    # concurrent.futures.process.BrokenProcessPool. 'spawn' gives each worker
    # a fresh interpreter instead of an inherited fork state. Requires the
    # __main__ guard (spawn re-imports this file in each worker; without the
    # guard that re-triggers top-level generator creation recursively) -- see
    # generate_tfr_no_noise_contained_2ns5ns.py for the original occurrence.
    multiprocessing.set_start_method('spawn', force=True)
    main()
