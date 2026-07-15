# OptimizedDataGenerator_v3.py
#
# Copy of OptimizedDataGenerator_v2p5.py plus one new opt-in, default-off
# feature: hard digitization (map_to_levels), das214's validated Part-2
# pattern -- a real tf.raw_ops.Bucketize step applied to already-loaded
# batches in __getitem__, right after the existing `quantize` step. Default
# behavior (digitize=False) is byte-for-byte identical to v2p5; only callers
# that explicitly pass digitize=True/digitize_thresholds/digitize_levels see
# any difference. v2p5 itself is left untouched -- existing Part 1 scripts
# keep importing it unchanged.
import os
import gc
import math
import glob
import random
import logging
import datetime
import numpy as np
import pandas as pd
import json

from typing import Union, List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm
import tensorflow as tf
from qkeras import quantized_bits

import utils

# custom quantizer

# @tf.function
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    """
    Applies QKeras quantization.
    Args:
        data (tf.Tensor): Input data (tf.Tensor).
        bits (int): Number of bits for quantization.
        int_bits (int): Number of integer bits.
        alpha (float): (don't change)
    Returns::
        tf.Tensor: Quantized data (tf.Tensor).
    """
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)


class OptimizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
            dataset_base_dir: str = "./",
            batch_size: int = 32,
            optimize_batch_size: bool = False,
            file_count = None,
            labels_list: Union[List,str] = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            to_standardize: bool = False,
            input_shape: Tuple = (13,21),
            transpose = None,
            files_from_end = False,
            shuffle=False,

            # Added in Optimized datagenerators
            load_from_tfrecords_dir: str = None,
            tfrecords_dir: str = None,
            use_time_stamps = -1,
            select_contained = False, #If true, selects only clusters with original_atEdge==False
            noise = -1, #add gaussian noise (mu, sigma), set to -1 to turn off
            seed: int = None,
            min_threshold: float = None, #Zeros out charge<min_thresh
            max_threshold: float = None, #Zeros out charge>max_thresh
            quantize: bool = False,
            # v3: hard digitization (ADC-style bucketize), applied in __getitem__
            # right after `quantize`. Default off -- byte-identical to v2p5 unless
            # a caller opts in. digitize_thresholds: sorted list of N boundary
            # values (mV). digitize_levels: list of N+1 output level values.
            digitize: bool = False,
            digitize_thresholds=None,
            digitize_levels=None,
            max_workers: int = 1,
            label_scale_pctl: float = 99,
            norm_pos_pctl: float = 99.7,
            norm_neg_pctl: float = 99.7,
            tail_tol: float = 0.75,
            **kwargs,
            ):
        super().__init__()

        self.shuffle = shuffle
        self.seed = seed if seed is not None else 13
        if shuffle:
            self.rng = np.random.default_rng(seed = self.seed)

        if load_from_tfrecords_dir is None:
            # CREATOR MODE
            n_time, height, width = input_shape

            if use_time_stamps == -1:
                use_time_stamps = list(np.arange(0,20))
            assert len(use_time_stamps) == n_time, f"Expected {n_time} time steps, got {len(use_time_stamps)}"

            len_xy = height * width
            col_indices = [
                np.arange(t * len_xy, (t + 1) * len_xy).astype(str)
                for t in use_time_stamps
            ]
            self.recon_cols = np.concatenate(col_indices).tolist()

            self.max_workers = max_workers
            self.label_scale_pctl = label_scale_pctl
            self.norm_pos_pctl = norm_pos_pctl
            self.norm_neg_pctl = norm_neg_pctl


            self.files = sorted(glob.glob(os.path.join(dataset_base_dir, "part.*.parquet"), recursive=False))

            if file_count != None:
                if not files_from_end:
                    self.files = self.files[:file_count]
                else:
                    self.files = self.files[-file_count:]

            self.file_offsets = [0]
            self.dataset_mean = None
            self.dataset_std = None
            self.norm_factor_pos = None
            self.norm_factor_neg = None
            self.labels_scale = None

            self.labels_list = labels_list
            self.input_shape = input_shape
            self.transpose = transpose
            self.to_standardize = to_standardize
            self.noise = noise
            self.select_contained = select_contained
            self.min_threshold = min_threshold
            self.max_threshold = max_threshold
            if (max_threshold is not None) and (min_threshold is not None) and (max_threshold < min_threshold):
                raise ValueError("max_threshold < min_threshold!")

            self.process_file_parallel()


            if optimize_batch_size:
                original_bs = batch_size
                new_bs, residual = self.get_best_batch_size(self.file_offsets, original_bs)

                if new_bs != original_bs:
                    print(f"Batch size optimized from {original_bs} to {new_bs} "
                        f"to minimize final batch (residual: {residual} rows).")

                self.batch_size = new_bs
            else:
                self.batch_size = batch_size

            self.tail_tol = tail_tol
            self.batch_metadata = self.build_batch_metadata(
                batch_size=self.batch_size,
                file_offsets=self.file_offsets,
                tail_tol=self.tail_tol
            )


            self.current_file_index = None
            self.current_dataframes = None

            if tfrecords_dir is None:
                raise ValueError(f"tfrecords_dir is None")
            utils.safe_remove_directory(tfrecords_dir)

            self.tfrecords_dir = tfrecords_dir
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.save_batches_sequentially()
            del self.current_dataframes

            metadata_file_path = os.path.join(self.tfrecords_dir, "metadata.json")
            self.save_metadata(metadata_file_path)
            load_from_tfrecords_dir = self.tfrecords_dir

        # LOADER MODE
        self.file_offsets = [None]
        if not os.path.isdir(load_from_tfrecords_dir):
            raise ValueError(f"Directory {load_from_tfrecords_dir} does not exist.")

        self.tfrecords_dir = load_from_tfrecords_dir
        metadata_file_path = os.path.join(self.tfrecords_dir, "metadata.json")
        self.load_metadata(metadata_file_path)

        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tfrecords_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.digitize = digitize
        self.digitize_thresholds = digitize_thresholds
        self.digitize_levels = digitize_levels
        if self.digitize:
            if not self.digitize_thresholds or not self.digitize_levels:
                raise ValueError("digitize=True requires digitize_thresholds and digitize_levels.")
            if len(self.digitize_levels) != len(self.digitize_thresholds) + 1:
                raise ValueError("digitize_levels must have exactly one more entry than digitize_thresholds.")
        self.epoch_count = 0
        self.on_epoch_end()

    def save_metadata(self, metadata_file_path:str):
        """
        Saves the metadata of the dataset to a JSON file.
        Args:
            metadata_file_path (str): Path to save the metadata file.
        """
        metadata = {
            # Key configurations
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "recon_cols": self.recon_cols,
            "labels_list": self.labels_list,
            "to_standardize": self.to_standardize,
            "transpose": self.transpose,
            "shuffle": self.shuffle,
            "noise": self.noise,
            "select_contained": self.select_contained,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,

            "seed": self.seed,
            "label_scale_pctl": self.label_scale_pctl,
            "norm_pos_pctl": self.norm_pos_pctl,
            "norm_neg_pctl": self.norm_neg_pctl,
            "tail_tol": self.tail_tol,

            # Calculated statistics
            "dataset_mean": self.dataset_mean.tolist() if self.dataset_mean is not None else None,
            "dataset_std": self.dataset_std.tolist() if self.dataset_std is not None else None,
            "dataset_min": np.float64(self.dataset_min) if self.dataset_min is not None else None,
            "dataset_max": np.float64(self.dataset_max) if self.dataset_max is not None else None,
            "norm_factor_pos": self.norm_factor_pos,
            "norm_factor_neg": self.norm_factor_neg,
            "labels_scale": self.labels_scale.tolist() if self.labels_scale is not None else None,

            # Full batch plan
            "batch_metadata": self.batch_metadata


        }
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved successfully ast {metadata_file_path}")

    def load_metadata(self, metadata_file_path:str):
        """
        Loads the metadata of the dataset from a JSON file.
        Args:
            metadata_file_path (str): Path to the metadata file.
        """
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file {metadata_file_path} does not exist.\n"
                                    "Cannot initialiize genrator in load mode.")
        print(f"Loading metadata from {metadata_file_path}")
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)

        # Key configurations
        self.batch_size = metadata['batch_size']
        self.input_shape = tuple(metadata['input_shape'])
        self.recon_cols = metadata['recon_cols']
        self.labels_list = metadata['labels_list']
        self.to_standardize = metadata['to_standardize']
        self.select_contained = metadata['select_contained']
        self.label_scale_pctl = metadata['label_scale_pctl']
        self.norm_pos_pctl = metadata['norm_pos_pctl']
        self.norm_neg_pctl = metadata['norm_neg_pctl']
        self.tail_tol = metadata['tail_tol']

        # Calculated statistics
        self.dataset_mean = np.array(metadata['dataset_mean'])
        self.dataset_std = np.array(metadata['dataset_std'])
        self.dataset_min = np.float64(metadata['dataset_min'])
        self.dataset_max = np.float64(metadata['dataset_max'])
        self.norm_factor_pos = metadata['norm_factor_pos']
        self.norm_factor_neg = metadata['norm_factor_neg']
        self.labels_scale = np.array(metadata['labels_scale'])

        # Full batch plan
        self.batch_metadata = metadata['batch_metadata']


        # Optional parameters
        self.shuffle = metadata.get('shuffle', False)
        self.seed = metadata.get('seed', 13)
        self.transpose = metadata.get('transpose', None)
        self.noise = metadata.get('noise', -1)
        self.min_threshold = metadata.get('min_threshold', None)
        self.max_threshold = metadata.get('max_threshold', None)

        if self.shuffle:
            self.rng = np.random.default_rng(seed=self.seed)


    def process_file_parallel(self):
        file_infos = [(afile,
                    self.recon_cols, self.labels_list, self.noise, self.min_threshold, self.max_threshold, self.select_contained,
                    self.label_scale_pctl, self.norm_pos_pctl, self.norm_neg_pctl)
                    for afile in self.files
                    ]
        results = []
        with ProcessPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._process_file_single, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(file_infos), desc="Processing Files..."):
                results.append(future.result())

        for amean, avariance, amin, amax, num_rows, labels_scale, pos_scale, neg_scale in results:
            self.file_offsets.append(self.file_offsets[-1] + num_rows)

            if self.dataset_mean is None:
                self.dataset_max = amax
                self.dataset_min = amin
                self.dataset_mean = amean
                self.dataset_std = avariance
            else:
                self.dataset_max = max(self.dataset_max, amax)
                self.dataset_min = min(self.dataset_min, amin)
                self.dataset_mean += amean
                self.dataset_std += avariance

            if self.labels_scale is None:
                self.labels_scale = labels_scale
            else:
                self.labels_scale = np.maximum(self.labels_scale, labels_scale)

            self.norm_factor_pos = (pos_scale if self.norm_factor_pos is None
                                    else max(self.norm_factor_pos, pos_scale))
            self.norm_factor_neg = (neg_scale if self.norm_factor_neg is None
                                    else max(self.norm_factor_neg, neg_scale))

        self.dataset_mean = self.dataset_mean / len(self.files)
        self.dataset_std = np.sqrt(self.dataset_std / len(self.files))

        self.file_offsets = np.array(self.file_offsets)

    @staticmethod
    def _process_file_single(file_info):
        afile, recon_cols, labels_list, noise, min_threshold, max_threshold, select_contained, label_scale_pctl, norm_pos_pctl, norm_neg_pctl = file_info
        if select_contained:
            df = (pd.read_parquet(afile,
                                 columns=recon_cols + labels_list +['original_atEdge'])
                    .reset_index(drop=True))
            df = df.loc[df['original_atEdge'] == False]
        else:
            df = (pd.read_parquet(afile,
                                 columns=recon_cols + labels_list)
                    .reset_index(drop=True))
        # df = pd.read_parquet(afile, columns=recon_cols + labels_list).reset_index(drop=True)
        x = df[recon_cols].values

        if noise != -1:
            # Stats-only pass (mean/std/percentiles for `standardize()`, unused when
            # to_standardize=False) -- plain i.i.d. noise is fine here, correlation
            # between time slices only matters for the actual data in prepare_batch_data().
            bkg = np.random.normal(noise[0], noise[1], x.shape)
            x = x+bkg
        if min_threshold is not None:
            bellowthresh = x < min_threshold
            x[bellowthresh] = 0*x[bellowthresh]
        if max_threshold is not None:
            abovethresh = x > max_threshold
            x[abovethresh] = 0*x[abovethresh]


        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        amean, avariance = np.mean(x[nonzeros], keepdims=True), np.var(x[nonzeros], keepdims=True) + 1e-10
        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)
        amin, amax = np.min(centered), np.max(centered)

        pos_vals = np.abs(centered[centered  > 0])
        neg_vals = np.abs(centered[centered  < 0])

        pos_scale = (np.percentile(pos_vals, norm_pos_pctl)
                    if pos_vals.size else 1.0)
        neg_scale = (np.percentile(neg_vals, norm_neg_pctl)
                    if neg_vals.size else 1.0)

        len_adf = len(df)

        labels_values = df[labels_list].values
        labels_scale = np.percentile(np.abs(labels_values), label_scale_pctl, axis=0)

        del df
        gc.collect()

        return amean, avariance, amin, amax, len_adf, labels_scale, pos_scale, neg_scale

    def _sample_noise(self, shape):
        """
        Samples gaussian noise for `recon_values` (shape = (n_samples, n_time*len_xy),
        columns ordered by time slice per `use_time_stamps`, see __init__).
        `self.noise = [mu, sigma]` -> i.i.d. N(mu, sigma) everywhere (old behavior).
        `self.noise = [mu, sigma, rho]` -> only the first time slice's noise is drawn
        independently, n_1 ~ N(mu, sigma); every subsequent slice is correlated with
        the immediately preceding one (ADC/CSA bandwidth-limited noise is a Markov
        process in time): n_i = mu + rho*(n_{i-1}-mu) + sqrt(1-rho^2)*N(0, sigma).
        `rho = exp(-dt_slice/tau)`, `tau = 1/(2*pi*fc)`, assumes uniform spacing
        between consecutive `use_time_stamps` (single scalar rho for all gaps).
        """
        mu, sigma = self.noise[0], self.noise[1]
        n_time = self.input_shape[0]
        if len(self.noise) < 3:
            return np.random.normal(mu, sigma, shape)

        rho = self.noise[2]
        len_xy = shape[1] // n_time
        slices = [np.random.normal(mu, sigma, (shape[0], len_xy))]  # first slice: purely random
        for _ in range(1, n_time):
            prev = slices[-1]
            nxt = mu + rho * (prev - mu) + np.sqrt(1 - rho**2) * np.random.normal(0.0, sigma, (shape[0], len_xy))
            slices.append(nxt)
        return np.concatenate(slices, axis=1)

    def standardize(self, x):
        """
        Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized.
        """
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/self.norm_factor_pos
        out[out < 0] = out[out < 0]/self.norm_factor_neg
        out = np.clip(out, self.dataset_min, self.dataset_max)
        return out

    def save_batches_sequentially(self):
        num_batches = self.__len__()
        errors_found = []
        for i in tqdm(range(num_batches), desc="Saving batches as TFRecords"):
            result = self.save_single_batch(i)
            if "Error" in result:
                print(result)
                errors_found.append(result)

        if errors_found:
            logging.warning(f"Encountered {len(errors_found)} errors during sequential saving of TFRecords.")
        else:
            logging.info("All batches saved successfully in sequential mode.")


    def save_single_batch(self, batch_index):
        """
        Serializes and saves a single batch to a TFRecord file.
        Args:
            batch_index (int): Index of the batch to save.
        Returns:
            str: Path to the saved TFRecord file or an error message.
        """

        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tfrecords_dir, filename)
            X, y = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X, y)
            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        except Exception as e:
            return f"Error saving batch {batch_index}: {e}"

    @staticmethod
    def get_best_batch_size(file_offsets, target_bs=5000):
        """
        Find the best batch size that minimizes the residual when dividing the total number of rows.
        Args:
            file_offsets (np.ndarray): Array of file offsets.
            target_bs (int): Target batch size.
            tol (float): Tolerance for batch size deviation.
        Returns:
            int: Best batch size.
        """
        last_offset = file_offsets[-1]
        d_bs = int(0.5 * target_bs)
        batch_sizes = np.arange(target_bs - d_bs, target_bs + d_bs + 1)

        residuals = last_offset % batch_sizes
        min_res   = residuals.min()

        # All bs giving the minimal residual
        candidates = batch_sizes[residuals == min_res]

        # Prefer the one closest to the target
        idx = np.argmin(np.abs(candidates - target_bs))
        return int(candidates[idx]), min_res

    @staticmethod
    def _build_batching_plan(file_offsets, batch_size, tol = 0.75):
        """
        Pre-compute (row_start, row_end) for every batch.
        If the last batch < 0.5xbatch_size, merge the last two
        and split them evenly, so both new batches are within
        0.5x...1.0xbatch_size.
        """
        total = file_offsets[-1]
        b      = batch_size
        plan   = []
        start  = 0
        while start < total:
            end = min(start + b, total)
            plan.append((start, end))
            start = end

        # Re-balance if the tail is too short
        if len(plan) >= 2:
            last_len = plan[-1][1] - plan[-1][0]
            if last_len < tol * b:
                sec_start = plan[-2][0]
                comb_len  = plan[-1][1] - sec_start
                half      = math.ceil(comb_len / 2)
                plan[-2]  = (sec_start, sec_start + half)
                plan[-1]  = (sec_start + half, sec_start + comb_len)
        return plan

    @classmethod
    def build_batch_metadata(cls, batch_size: int, file_offsets: np.ndarray, tail_tol: float = 0.75) -> List[Dict[str, Any]]:
        """
        Builds optimized batch metadata using a pre-computed batch plan.
        This ensures that the final batch is not excessively small.
        """
        batching_plan = cls._build_batching_plan(file_offsets, batch_size, tail_tol)
        batch_metadata = []

        # 2. Loop through the generated plan instead of a simple range
        for batch_index, (start_evt, end_evt) in enumerate(batching_plan):

            # Create a new dictionary for the current batch
            current_batch_meta = {
                "batch_idx": batch_index,
                "target_batch_size": int(batch_size),
                # The actual size is now simply the difference from the plan
                "actual_batch_size": int(end_evt - start_evt),
                "segments": []
            }

            # 3. Use the same logic as before to find the file segments for the given range
            file_idx = np.searchsorted(file_offsets, start_evt, side="right") - 1
            evt_cursor = start_evt

            while evt_cursor < end_evt:
                file_start = file_offsets[file_idx]
                file_end = file_offsets[file_idx + 1]

                rel_start = evt_cursor - file_start
                rel_end = min(end_evt, file_end) - file_start

                # Append segment info to the current batch's metadata
                current_batch_meta["segments"].append({
                    "file_idx": int(file_idx),
                    "row_start": int(rel_start),
                    "row_end": int(rel_end - 1)
                })

                evt_cursor += (rel_end - rel_start)
                file_idx += 1

            batch_metadata.append(current_batch_meta)

        return batch_metadata

    def prepare_batch_data(self, batch_index):
        batch_plan = self.batch_metadata[batch_index]

        X_chunks = []
        y_chunks = []

        for segment in batch_plan["segments"]:
            file_idx = segment["file_idx"]
            rel_start = segment["row_start"]
            rel_end = segment["row_end"] + 1  # inclusive end

            if file_idx != self.current_file_index:
                parquet_file = self.files[file_idx]
                if self.select_contained:
                    all_columns_to_read = self.recon_cols + self.labels_list + ['original_atEdge']
                    df = (pd.read_parquet(parquet_file,
                                         columns = all_columns_to_read)
                            .dropna(subset=self.recon_cols)
                            .reset_index(drop=True))
                    df = df.loc[df['original_atEdge'] == False]
                else:
                    all_columns_to_read = self.recon_cols + self.labels_list
                    df =(pd.read_parquet(parquet_file,
                                         columns = all_columns_to_read)
                            .dropna(subset=self.recon_cols)
                            .reset_index(drop=True))
                # df = (pd.read_parquet(parquet_file,
                #                     columns=self.recon_cols + self.labels_list)
                #         .dropna(subset=self.recon_cols)
                #         .reset_index(drop=True))
                if self.shuffle:
                    df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                recon_df  = df[self.recon_cols]
                labels_df = df[self.labels_list]

                recon_values = recon_df.values
                if self.noise !=-1:
                    bkg = self._sample_noise(recon_values.shape)
                    recon_values = recon_values + bkg
                if self.min_threshold is not None:
                    bellowthresh = recon_values < self.min_threshold
                    recon_values[bellowthresh] = 0*recon_values[bellowthresh]
                if self.max_threshold is not None:
                    abovethresh = recon_values > self.max_threshold
                    recon_values[abovethresh] = 0*recon_values[abovethresh]

                # nonzeros = abs(recon_values) > 0
                # recon_values[nonzeros] = np.sign(recon_values[nonzeros]) * np.log1p(abs(recon_values[nonzeros])) / np.log(2)
                # if self.to_standardize:
                #     recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
                recon_values = recon_values.reshape((-1, *self.input_shape))
                if self.transpose is not None:
                    recon_values = recon_values.transpose(self.transpose)
                self.current_dataframes = (
                    recon_values,
                    labels_df.values,
                )
                self.current_file_index = file_idx
                del df
                gc.collect()

            recon_df, labels_df = self.current_dataframes
            X_chunk = recon_df[rel_start:rel_end]
            y_chunk = labels_df[rel_start:rel_end] / self.labels_scale

            X_chunks.append(X_chunk)
            y_chunks.append(y_chunk)



        X = np.concatenate(X_chunks, axis=0)
        y = np.concatenate(y_chunks, axis=0)

        return X, y


    def serialize_example(self, X, y):
        """
        Serializes a single example (featuresand labels) to TFRecord format.

        Args:
        - X: Training data
        - y: labelled data

        Returns:
        - string (serialized TFRecord example).
        """
        # X and y are float32 (maybe we can reduce this)
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)

        feature = {
            'X': self._bytes_feature(tf.io.serialize_tensor(X)),
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """
        Converts a string/byte value into a Tf feature of bytes_list

        Args:
        - string/byte value

        Returns:
        - tf.train.Feature object as a bytes_list containing the input value.
        """
        if isinstance(value, type(tf.constant(0))): # check if Tf tensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def map_to_levels(self, x):
        """
        Hard ADC-style digitization (das214's Part-2 pattern): buckets `x`
        against `self.digitize_thresholds` (real tf.raw_ops.Bucketize, a
        true step function -- not a soft/differentiable approximation) and
        maps each bucket to the corresponding value in `self.digitize_levels`.
        For N thresholds there are N+1 buckets/levels: values <= threshold[0]
        map to levels[0], threshold[0] < x <= threshold[1] map to levels[1],
        ..., x > threshold[-1] maps to levels[-1].
        """
        boundaries = [float(t) for t in self.digitize_thresholds]
        bucket_indices = tf.raw_ops.Bucketize(input=x, boundaries=boundaries)
        bucket_indices = tf.clip_by_value(bucket_indices, 0, len(self.digitize_levels) - 1)
        levels = tf.constant(self.digitize_levels, dtype=tf.float32)
        return tf.gather(levels, bucket_indices)

    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        quantization is done here: Helpful for pretraining without the quantization and the later training with quantized data.
        shuffling is also done here.
        TODO: prefetching (un-done)
        """
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Get the first (and only) batch from the dataset
        try:
            X_batch, y_batch = next(iter(parsed_dataset))
        except StopIteration:
            raise ValueError(f"No data found in TFRecord file: {tfrecord_path}")

        # Was `tf.reshape(X_batch, [-1, *X_batch.shape[1:]])` -- a reshape into the
        # tensor's own shape, meant as a no-op sanity pass. Two failed fix attempts
        # before this one (both crashed Part 3 for real, see git history/session notes
        # if this needs revisiting):
        #   1. Original `.shape[1:]` (Python-level static read): crashed with "Input to
        #      reshape is a tensor with 2560000 values, but the requested shape has
        #      20000" -- some call got a stale/wrong static shape baked in, likely
        #      because this __getitem__ runs under Keras's PyDatasetAdapter via
        #      tf.data.Dataset.from_generator + autograph, not plain eager Python.
        #   2. `tf.reshape(X_batch, tf.concat([[-1], tf.shape(X_batch)[1:]], axis=0))`
        #      (dynamic shape via a real op instead of the static attribute): survived
        #      longer (a full ~30-epoch attempt) but then crashed differently --
        #      "generator yielded an element of shape (2560000, 1) where an element of
        #      shape (None, 16, 16, 2) was expected". A tf.reshape whose target shape is
        #      itself a *dynamic* tensor (not a Python list of ints) produces an output
        #      with statically UNKNOWN shape (TensorShape(None)); something downstream
        #      in Keras/tf.data's output-signature matching mishandled that unknown
        #      shape and flattened the batch instead.
        # `tf.ensure_shape` fixes both: its target is a plain Python list evaluated at
        # trace time (no stale-attribute risk like #1), and -- unlike reshape -- it
        # *restores* concrete static shape info on its output rather than erasing it
        # (no unknown-shape risk like #2). It also fails fast with a clear message if a
        # batch is ever genuinely malformed, instead of a cryptic low-level TF error.
        # Confirmed via direct inspection: X out of parse_tensor here is channel-LAST
        # (rows, 16, 16, 2) -- do not use self.input_shape, which stores the
        # channel-FIRST (2, 16, 16) convention used elsewhere in this file (e.g. the
        # reconstruction reshape) and is not the on-disk layout for this tensor.
        X_batch = tf.ensure_shape(X_batch, [None, 16, 16, 2])
        y_batch = tf.ensure_shape(y_batch, [None, len(self.labels_list)])

        if self.quantize:
            X_batch = QKeras_data_prep_quantizer(X_batch, bits=4, int_bits=0, alpha=1)

        if self.digitize:
            X_batch = self.map_to_levels(X_batch)

        if self.shuffle:
            indices = tf.range(start=0, limit=tf.shape(X_batch)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices, seed=self.seed)
            X_batch = tf.gather(X_batch, shuffled_indices)
            y_batch = tf.gather(y_batch, shuffled_indices)

        del raw_dataset, parsed_dataset
        return X_batch, y_batch

    @staticmethod
    def _parse_tfrecord_fn(example):
        """
        Parses a single TFRecord example.

        Returns:
        - X: as a float32 tensor.
        - y: as a float32 tensor.
        """
        feature_description = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        X = tf.io.parse_tensor(example['X'], out_type=tf.float32)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
        return X, y

    def __len__(self):
        """
        Phase-aware length:
            during initial TFRecord creation: math on file_offsets
            after creation in same process: len(batch_metadata)
            when loading existing TFRecords: len(tfrecord_filenames)
        """
        # already have metadata?  Fastest answer.
        if self.batch_metadata:
            return len(self.batch_metadata)

        # still building batches, so compute from source rows.
        if len(self.file_offsets) > 1:         # have real offsets
            total_rows = self.file_offsets[-1]
            return math.ceil(total_rows / self.batch_size)

        # running in "load" mode.
        self.tfrecord_filenames = np.sort(
            np.array(tf.io.gfile.glob(
                os.path.join(self.tfrecords_dir, "*.tfrecord"))))
        return len(self.tfrecord_filenames)

    def on_epoch_end(self):
        '''
        This shuffles the file ordering so that it shuffles the ordering in which the TFRecord
        are loaded during the training for each epochs.
        '''
        gc.collect()
        self.epoch_count += 1
        # Log quantization status once
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.")

        if self.shuffle:
            self.rng.shuffle(self.tfrecord_filenames)
            self.seed += 1 # So that after each epoch the batch is shuffled with a different seed (deterministic)
