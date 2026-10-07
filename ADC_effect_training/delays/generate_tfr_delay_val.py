"""Delayed validation TFRecords for the Conv2D / QConv2D pass-through.

Drop-in replacement for TFR_files_2_5_no_noise_contained/TFR_val in which each
pixel's two samples are read from SHIFTED slice indices, and the resulting levels
are capped by the 12.5 ns auto-zero window.

    i       = round( (T + shift - d) / 0.2 )          T = 2 ns and 5 ns
    X_mV    = w[i]        (0 if T + shift - d <= 0, i.e. before the trace starts)
    cap     = highest k with t_k + d_k <= 12.5 ns
    level   = min( Bucketize(X_mV, thresholds), cap )

No interpolation anywhere. The waveforms are 200 ps native, downsampled from 10 ps
before being written to disk, and the DG has never interpolated: it selects slices
by integer index and the model only ever sees stored slice values. Rounding the
index also quantises the LUT delay to 200 ps, which is deliberate.

One delay per pixel, taken at the highest threshold the pixel reaches (d_kmax).
The LUT gives three delays per pixel; a single value is what makes a rigid slide
of the sampling well defined. It over-delays the lower thresholds slightly, which
is the conservative direction.

The 12.5 ns window is FIXED and does not move with the shift: it is a property of
the front-end reset, not of where the ADC samples. So the cap is identical across
all shifts and the three datasets differ in exactly one variable.

Output stores LEVELS, not mV, because a window cap cannot be expressed as a
voltage without fabricating one. Evaluate with digitize=False.

Events, order, labels and metadata are inherited from the source TFR set rather
than recomputed: y is copied from the source records and metadata.json is copied
verbatim, since it carries labels_scale which the evals un-scale predictions with.

Self-check: with shift = 0 and the delay disabled, the regenerated mV must equal
the source mV bit-for-bit. Asserted per batch; that is what proves the parquet-row
mapping is right.

Usage:
  python generate_tfr_delay_val.py --shift 0
  python generate_tfr_delay_val.py --shift 1.6
  python generate_tfr_delay_val.py --shift 3.4
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lut_loader as L                                                    # noqa: E402
from run_delay_study import (CvG, DT, NT, NPIX, THRESHOLDS_MV, WINDOW,    # noqa: E402
                             DATA_BASE)

DATASET_BASE = os.path.dirname(DATA_BASE)          # .../shuffled_3d
SRC_TFR = os.path.join(DATASET_BASE, 'TFR_files_2_5_no_noise_contained', 'TFR_val')
PARQUET_DIR = os.path.join(DATA_BASE, 'test')
SAMPLE_TIMES_NS = [2.0, 5.0]
RECON_COLS = [str(c) for c in range(NT * NPIX)]


def dst_dir(shift):
    # Always tag the shift, including 0, so the three sets are self-describing and
    # none of them collides with the pre-fix output generated from the holed LUT.
    tag = f'delay_shift{shift:g}'.replace('.', 'p')
    return os.path.join(DATASET_BASE, f'TFR_files_2_5_no_noise_contained_{tag}', 'TFR_val')


def load_parquet(path):
    df = pd.read_parquet(path, columns=RECON_COLS + ['original_atEdge'],
                         engine='fastparquet')
    keep = ~df['original_atEdge'].astype(bool).values
    w = df[RECON_COLS].values[keep].astype(np.float32).reshape(-1, NT, NPIX)
    del df
    return w


def pixel_delay_and_cap(w, points):
    """-> (d, cap). d is the LUT delay at the pixel's highest reached threshold."""
    qin = w.max(axis=1) / CvG
    d = np.zeros(qin.shape)
    t_fire = []
    for (ch, dl), th_mv in zip(points, THRESHOLDS_MV):
        above = w >= th_mv
        reaches = above.any(axis=1)
        dk = L.nearest_row(ch, dl, qin)
        d = np.where(reaches, dk, d)                       # later k overwrites
        t_k = np.where(reaches, np.argmax(above, axis=1) * DT, np.inf)
        t_fire.append(t_k + dk)
    cap = np.zeros(qin.shape, dtype=np.int8)
    for k in range(3):
        cap = np.where((t_fire[k] <= WINDOW) & (cap == k), k + 1, cap)
    return d, cap


def build_levels(w, d, cap, shift, apply_delay=True):
    """-> (levels (nev,16,16,2) float32, mV (nev,16,16,2) for the control check)."""
    dd = d if apply_delay else np.zeros_like(d)
    chans_mv, chans_lv = [], []
    for T in SAMPLE_TIMES_NS:
        t = T + shift - dd
        i = np.clip(np.rint(t / DT).astype(np.int64), 0, NT - 1)
        mv = np.where(t > 0, np.take_along_axis(w, i[:, None, :], axis=1)[:, 0, :], 0.0)
        lv = np.digitize(mv, THRESHOLDS_MV)                # 0..3
        if apply_delay:
            lv = np.minimum(lv, cap)
        chans_mv.append(mv.reshape(-1, 16, 16))
        chans_lv.append(lv.reshape(-1, 16, 16))
    return (np.stack(chans_lv, -1).astype(np.float32),
            np.stack(chans_mv, -1).astype(np.float32))


def read_record(path):
    raw = next(iter(tf.data.TFRecordDataset(path)))
    ex = tf.io.parse_single_example(raw, {'X': tf.io.FixedLenFeature([], tf.string),
                                          'y': tf.io.FixedLenFeature([], tf.string)})
    return (tf.io.parse_tensor(ex['X'], tf.float32).numpy(),
            tf.io.parse_tensor(ex['y'], tf.float32).numpy())


def write_record(path, X, y):
    def _b(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    feat = {'X': _b(tf.io.serialize_tensor(tf.cast(X, tf.float32)).numpy()),
            'y': _b(tf.io.serialize_tensor(tf.cast(y, tf.float32)).numpy())}
    with tf.io.TFRecordWriter(path) as wr:
        wr.write(tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shift', type=float, required=True,
                    help='readout shift in ns, must be a multiple of 0.2')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if abs(args.shift / DT - round(args.shift / DT)) > 1e-9:
        raise SystemExit(f'shift {args.shift} is not a multiple of the {DT} ns slice step')

    DST = dst_dir(args.shift)
    if os.path.abspath(DST) == os.path.abspath(SRC_TFR):
        raise SystemExit('destination equals source, refusing')
    existing = glob.glob(os.path.join(DST, '*.tfrecord'))
    if existing and not args.force:
        raise SystemExit(f'{DST} already holds {len(existing)} tfrecord(s); '
                         'move them aside or pass --force')

    cols, points = L.build(THRESHOLDS_MV)
    meta = json.load(open(os.path.join(SRC_TFR, 'metadata.json')))
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, 'part.*.parquet')))
    os.makedirs(DST, exist_ok=True)

    print(f'source      : {SRC_TFR}')
    print(f'destination : {DST}')
    print(f'LUT         : {os.path.basename(L.DEFAULT_LUT)}  columns {cols}')
    print(f'thresholds  : {THRESHOLDS_MV} mV')
    print(f'shift       : {args.shift} ns  -> nominal slices '
          f'{[int(round((T + args.shift) / DT)) for T in SAMPLE_TIMES_NS]}')
    print(f'window      : {WINDOW} ns (fixed, independent of shift)\n')

    cache_idx, cache_w, tot = None, None, 0
    lv_hist = np.zeros((2, 4), dtype=np.int64)
    for bm in meta['batch_metadata']:
        b = bm['batch_idx']
        chunks = []
        for seg in bm['segments']:
            if seg['file_idx'] != cache_idx:
                cache_w = load_parquet(files[seg['file_idx']])
                cache_idx = seg['file_idx']
            chunks.append(cache_w[seg['row_start']:seg['row_end'] + 1])
        w = np.concatenate(chunks, axis=0)

        X_src, y_src = read_record(os.path.join(SRC_TFR, f'batch_{b}.tfrecord'))
        zero = np.zeros((w.shape[0], NPIX))
        _, ctrl_mv = build_levels(w, zero, zero.astype(np.int8), 0.0, apply_delay=False)
        if ctrl_mv.shape != X_src.shape or not np.array_equal(ctrl_mv, X_src):
            raise SystemExit(f'batch {b}: zero-shift rebuild does not match source mV. '
                             'Row mapping is wrong, refusing to write.')

        d, cap = pixel_delay_and_cap(w, points)
        X, _ = build_levels(w, d, cap, args.shift)
        write_record(os.path.join(DST, f'batch_{b}.tfrecord'), X, y_src)

        for ch in range(2):
            lv_hist[ch] += np.bincount(X[..., ch].astype(int).ravel(), minlength=4)
        hit = d > 0
        print(f'  batch {b}: {w.shape[0]:5d} events  control OK  '
              f'median d = {np.median(d[hit]) if hit.any() else 0:.2f} ns')
        tot += w.shape[0]
        del w, chunks

    shutil.copy2(os.path.join(SRC_TFR, 'metadata.json'),
                 os.path.join(DST, 'metadata.json'))
    print(f'\nwrote {tot} events to {DST}')
    for ch, T in enumerate(SAMPLE_TIMES_NS):
        n = lv_hist[ch].sum()
        print(f'  channel {ch} ({T} ns): L0..L3 = {list(lv_hist[ch])}  '
              f'lit {n - lv_hist[ch][0]} ({(n - lv_hist[ch][0]) / n * 100:.2f}%)')
    print('metadata.json copied verbatim (labels_scale must not be recomputed)')
    print('evaluate with digitize=False: these are levels, not mV')


if __name__ == '__main__':
    main()
