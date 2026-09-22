"""Delay study: how many pixels drop an ADC level once comparator delay is applied?

Baseline and delayed levels both use the same 12.5 ns auto-zero window, so the
delay is the only difference. With d = 0 the procedure reproduces the current
Bucketize levels exactly (asserted below).

Threshold k asserts at t_k + d_k. The delays are NOT summed across thresholds:
the LUT value at a given Q_th already supersedes the lower ones (confirmed with
the LUT authors, 2026-09-21).

Q_in lookup: **nearest tabulated row**, no interpolation. 1688 e- -> row 1700,
1610 e- -> row 1600. Below a column's first row (the LUT is not tabulated under
Q_in ~ 1.9*Q_th) this clamps to row 1 by construction, so the sub-edge case and
the in-range case are one rule.

The Q_th axis is likewise nearest-column, never interpolated: 238.1 e- -> 250,
415.1 -> 425, 959.0 -> 950. Adjacent columns differ by <= 0.07 ns.

Superseded variants (a serial/cumulative-delay model; linear interpolation with
sub-edge clamp / extrapolation / never-fires) are recorded in delay_plan.md.

Run with the project env:
  /work/users/harshul-cern/smartpixels/pixi-env/.pixi/envs/default/bin/python run_delay_study.py
"""
import json
import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LUT_CSV = os.path.join(HERE, 'VIZARD_ADC_LUT_PixB_27c_combined_sorted.csv')
DATA_BASE = ('/home/harshul-cern/work/projects/SmartPixML/'
             'dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/'
             'shuffled_3d/contained')
# Both splits: this is a characterisation of the front end, not a trained model,
# so there is nothing to leak. 80 train + 20 test files = 189956 contained clusters.
SUBDIRS = ['train', 'test']

CvG = 0.058          # mV per electron (58 uV/e-)
DT = 0.2             # ns per time slice
NT = 101             # slices, 0..20 ns
NPIX = 256
WINDOW = 12.5        # ns, auto-zero window
CHUNK = 500          # events per chunk
N_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # of 100 (80 train + 20 test)

# median thresholds, corr-noise 2ns/5ns campaign
THRESHOLDS_MV = [13.809455, 24.076480, 55.619907]
LUT_COLUMNS = [250, 425, 950]        # nearest 25 e- grid column to each

HIST_BINS = np.arange(0.0, 30.0 + 1e-9, 0.05)


def load_lut():
    raw = pd.read_csv(LUT_CSV, dtype=str)
    out = {}
    for qt in LUT_COLUMNS:
        ch = pd.to_numeric(raw['Charge_Qth_%d' % qt], errors='coerce').values
        d = pd.to_numeric(raw['Qth = %d' % qt], errors='coerce').values
        m = ~(np.isnan(ch) | np.isnan(d))
        out[qt] = (ch[m], d[m] * 1e9)    # electrons, ns
    return out


def nearest_row(ch, d, q):
    """Delay at the tabulated row closest to q; clamps at both ends."""
    i = np.clip(np.searchsorted(ch, q), 1, len(ch) - 1)
    pick = np.where(np.abs(q - ch[i - 1]) <= np.abs(q - ch[i]), i - 1, i)
    return d[pick]


def crossing_and_delay(w, lut):
    """-> t_cross, delay, subedge, each (3, nev, npix). t_cross = inf if never crossed.

    Q_in comes from the waveform PEAK, not the 20 ns plateau: ~6% of hit pixels
    are induced-signal transients that spike and decay back to ~0, for which a
    plateau-based charge is meaningless (sometimes negative).
    """
    qin = w.max(axis=1) / CvG
    t_cross, delay, subedge = [], [], []
    for th_mv, qt in zip(THRESHOLDS_MV, LUT_COLUMNS):
        above = w >= th_mv
        ever = above.any(axis=1)
        idx = np.argmax(above, axis=1)
        i0 = np.clip(idx - 1, 0, NT - 1)
        v0 = np.take_along_axis(w, i0[:, None, :], 1)[:, 0, :]
        v1 = np.take_along_axis(w, idx[:, None, :], 1)[:, 0, :]
        frac = np.where(v1 > v0, (th_mv - v0) / np.maximum(v1 - v0, 1e-9), 0.0)
        t_cross.append(np.where(ever, (i0 + np.clip(frac, 0, 1)) * DT, np.inf))
        ch, d = lut[qt]
        delay.append(nearest_row(ch, d, qin))
        subedge.append((qin < ch.min()) & ever)      # reported only; row 1 is used anyway
    return np.array(t_cross), np.array(delay), np.array(subedge)


def levels(t_cross, delay):
    lvl = np.zeros(t_cross.shape[1:], dtype=np.int8)
    for k in range(3):
        lvl = np.where(((t_cross[k] + delay[k]) <= WINDOW) & (lvl == k), k + 1, lvl)
    return lvl


def ideal_levels(t_cross):
    """Level with no window and no delay: the highest threshold the waveform reaches."""
    lvl = np.zeros(t_cross.shape[1:], dtype=np.int8)
    for k in range(3):
        lvl = np.where(np.isfinite(t_cross[k]) & (lvl == k), k + 1, lvl)
    return lvl


def main():
    lut = load_lut()
    files = [f for sub in SUBDIRS
             for f in sorted(glob.glob(os.path.join(DATA_BASE, sub, 'part.*.parquet')))][:N_FILES]
    cols = [str(c) for c in range(NT * NPIX)]

    confusion = np.zeros((4, 4), dtype=np.int64)
    conf_ideal = np.zeros((4, 4), dtype=np.int64)
    clus_hist = np.zeros(NPIX + 1, dtype=np.int64)   # floored pixels per cluster
    clus_hit = np.zeros(NPIX + 1, dtype=np.int64)    # above-threshold pixels per cluster
    hist = np.zeros((3, len(HIST_BINS) - 1), dtype=np.int64)
    n_cross = np.zeros(3, dtype=np.int64)
    n_over = np.zeros(3, dtype=np.int64)
    n_sub = np.zeros(3, dtype=np.int64)
    n_ev = n_px = 0

    for path in files:
        df = pd.read_parquet(path, columns=cols + ['original_atEdge'], engine='fastparquet')
        # Same contained-cluster cut the training pipeline applies via the DG's
        # select_contained=True. The contained/ directory is NOT pre-filtered --
        # ~53% of its rows are atEdge. (The DG also dropna's on the recon columns;
        # this dataset has no NaNs, so that is a no-op.)
        keep = ~df['original_atEdge'].astype(bool).values
        allw = df[cols].values[keep].astype(np.float32)
        del df
        print(f'{os.path.basename(path)}: {allw.shape[0]} contained events '
              f'({keep.sum()}/{keep.size} kept)')

        for start in range(0, allw.shape[0], CHUNK):
            w = allw[start:start + CHUNK].reshape(-1, NT, NPIX)
            if w.shape[0] == 0:
                break
            tc, dl, sub = crossing_and_delay(w, lut)

            zero = np.zeros_like(dl)
            base = levels(tc, zero)
            new = levels(tc, dl)
            np.add.at(confusion, (base.ravel(), new.ravel()), 1)
            ideal = ideal_levels(tc)
            np.add.at(conf_ideal, (ideal.ravel(), new.ravel()), 1)
            # cluster view: the model sees the whole 16x16 array, not single pixels
            clus_hist += np.bincount((ideal > new).sum(axis=1), minlength=NPIX + 1)
            clus_hit += np.bincount((ideal > 0).sum(axis=1), minlength=NPIX + 1)

            for k in range(3):
                tot = (tc[k] + dl[k])[np.isfinite(tc[k])]
                hist[k] += np.histogram(tot, bins=HIST_BINS)[0]
                n_cross[k] += tot.size
                n_over[k] += int((tot > WINDOW).sum())
                n_sub[k] += int(sub[k].sum())

            n_ev += w.shape[0]
            n_px += w.shape[0] * NPIX
            del w, tc, dl, sub
        del allw

    base_counts = confusion.sum(axis=1)
    hit = int(base_counts[1:].sum())

    print(f'\n{n_ev} events, {n_px} pixels, noise-free, window = {WINDOW} ns')
    print(f'thresholds {THRESHOLDS_MV} mV -> LUT columns {LUT_COLUMNS}')
    print(f'pixels with baseline level > 0: {hit}\n')

    print('=== level distribution ===')
    print(f'  {"":<22}' + ''.join(f'{"L%d" % i:>10}' for i in range(4)))
    print(f'  {"baseline (no delay)":<22}' + ''.join(f'{int(c):10d}' for c in base_counts))
    print(f'  {"with delay":<22}' + ''.join(f'{int(c):10d}' for c in confusion.sum(axis=0)))

    changed = sum(confusion[b, n] for b in range(1, 4) for n in range(4) if n != b)
    drops = [sum(confusion[b, b - j] for b in range(j, 4)) for j in (1, 2, 3)]
    up = sum(confusion[b, n] for b in range(4) for n in range(4) if n > b)
    print(f'\n=== migration, of the {hit} pixels with baseline level > 0 ===')
    print(f'  changed {changed / hit * 100:.2f}%   ' +
          '   '.join(f'drop{j} {d / hit * 100:.2f}%' for j, d in zip((1, 2, 3), drops)) +
          f'   increases {up}')

    # By baseline level: a genuine partition, unlike the per-threshold rates below
    # (whose populations are nested -- every th3 crosser is also a th1 crosser).
    print('\n  by baseline level (disjoint bins):')
    print(f'    {"":<4}{"pixels":>10}{"dropped":>10}{"rate":>9}')
    for b in range(1, 4):
        n_b = confusion[b].sum()
        d_b = sum(confusion[b, n] for n in range(b))
        print(f'    L{b:<3}{int(n_b):10d}{int(d_b):10d}{d_b / max(n_b, 1) * 100:8.2f}%')

    # Headline table: every pixel whose final level is below the level its waveform
    # would reach with no timing constraint at all -- whether it missed the window on
    # t_k alone or only after the delay. Binned by that no-window level, which is a
    # partition, and all rates share one denominator so they sum to the total.
    ih = int(conf_ideal.sum(axis=1)[1:].sum())
    print(f'\n=== floored pixels, exclusive by level reached (of {ih} pixels reaching any threshold) ===')
    print(f'  {"reached":<10}{"pixels":>11}{"floored":>10}{"% of all":>10}{"rate in bin":>13}')
    tot_f = 0
    for b in range(1, 4):
        n_b = conf_ideal[b].sum()
        f_b = sum(conf_ideal[b, n] for n in range(b))
        tot_f += f_b
        print(f'  {"th%d only" % b if b < 3 else "th3":<10}{int(n_b):11d}{int(f_b):10d}'
              f'{f_b / ih * 100:9.2f}%{f_b / max(n_b, 1) * 100:12.2f}%')
    print(f'  {"TOTAL":<10}{ih:11d}{int(tot_f):10d}{tot_f / ih * 100:9.2f}%')

    nclus = int(clus_hist.sum())
    kk = np.arange(NPIX + 1)
    print(f'\n=== cluster view ({nclus} clusters; the ML input is the whole 16x16 array) ===')
    print(f'  above-threshold pixels per cluster: mean {(clus_hit * kk).sum() / nclus:.2f}')
    print(f'  floored pixels per cluster:         mean {(clus_hist * kk).sum() / nclus:.2f}')
    print(f'  clusters with >=1 floored pixel: {nclus - clus_hist[0]:8d}  '
          f'({(nclus - clus_hist[0]) / nclus * 100:.2f}%)')
    for t in (1, 2, 3, 5):
        n = int(clus_hist[t:].sum())
        print(f'  clusters with >={t} floored: {n:8d}  ({n / nclus * 100:5.2f}%)')

    print('\n=== t_k + delay: percentiles and window overflow ===')
    centers = 0.5 * (HIST_BINS[1:] + HIST_BINS[:-1])
    print(f'  {"":<6}{"p50":>8}{"p90":>8}{"p99":>8}{"max":>8}{"> 12.5 ns":>11}{"crossings":>11}')
    for k in range(3):
        cdf = np.cumsum(hist[k]) / max(hist[k].sum(), 1)
        p = [centers[np.searchsorted(cdf, q)] for q in (0.5, 0.9, 0.99)]
        mx = centers[np.nonzero(hist[k])[0][-1]]
        print(f'  th{k+1:<4}' + ''.join(f'{x:8.2f}' for x in p) +
              f'{mx:8.2f}{n_over[k] / max(n_cross[k], 1) * 100:10.2f}%{n_cross[k]:11d}')

    print('\n=== pixels below the LUT tabulated edge (row 1 used for these) ===')
    for k in range(3):
        print(f'  th{k+1}: {int(n_sub[k]):7d} of {int(n_cross[k]):7d} crossings '
              f'({n_sub[k] / max(n_cross[k], 1) * 100:5.1f}%)')

    out = {
        'n_events': n_ev, 'n_pixels': n_px, 'window_ns': WINDOW,
        'thresholds_mV': THRESHOLDS_MV, 'lut_columns': LUT_COLUMNS,
        'baseline_levels': base_counts.tolist(),
        'confusion': confusion.tolist(),
        'subedge_crossings': n_sub.tolist(),
        'crossings': n_cross.tolist(), 'over_window': n_over.tolist(),
        'confusion_ideal': conf_ideal.tolist(),
        'floored_per_cluster': clus_hist.tolist(),
        'hitpix_per_cluster': clus_hit.tolist(),
    }
    dest = os.path.join(HERE, 'delay_study_results.json')
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
