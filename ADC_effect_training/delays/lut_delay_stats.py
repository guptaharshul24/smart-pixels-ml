"""Min / median comparator delay from the LUT, for a given set of thresholds.

Used to pick the global sample-time shift for the delay pass-throughs: the two
cases are to advance the readout by the minimum delay the LUT can produce, and by
the median.

Columns are chosen from the **25 e- grid only**, deliberately. The LUT also carries
finer off-grid columns sitting closer to arbitrary threshold values, but those are
near-threshold patches rather than full columns (Qth_984 holds two cells), and
mixing them in would make the threshold the delay model assumes drift with Q_in.
One consistent column per threshold, same as the earlier runs.

The statistics here are over LUT CELLS and are therefore unweighted: every
tabulated Q_in counts once regardless of how often that charge occurs in the data.
For the delay actually assigned to pixels, which is what a sample-time shift is
really compensating, run the per-pixel version against the dataset instead.

Usage:
  python lut_delay_stats.py 13.001228 21.901985 57.135010
  python lut_delay_stats.py 224.2 377.6 985.1 --unit e
  python lut_delay_stats.py 13.0 21.9 57.1 --lut VIZARD_ADC_LUT_PixB_27c_global.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CvG = 0.058          # mV per electron (58 uV/e-)


def load_grid(path):
    """-> (qin, qth, delay_ns) from the 2D global-format LUT. NaN where untabulated."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith('Qth_')]
    if not cols or 'Qin_e' not in df.columns:
        raise SystemExit(f'{path} is not in the 2D global format '
                         '(expected a Qin_e column and Qth_* columns)')
    qth = np.array([int(c.split('_')[1]) for c in cols])
    order = np.argsort(qth)
    return df['Qin_e'].values.astype(float), qth[order], df[cols].values[:, order] * 1e9


def pick_column(qth, target_e, grid25=True):
    """Nearest column to target_e, restricted to the 25 e- grid unless told otherwise."""
    allowed = np.array([t for t in qth if t % 25 == 0]) if grid25 else qth
    if not len(allowed):
        raise SystemExit('no columns available after filtering')
    return int(allowed[np.argmin(np.abs(allowed - target_e))])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('thresholds', nargs=3, type=float, help='three thresholds, low to high')
    p.add_argument('--unit', choices=['mV', 'e'], default='mV',
                   help='units of the thresholds given (default mV)')
    p.add_argument('--lut', default='VIZARD_ADC_LUT_PixB_27c_global.csv')
    p.add_argument('--any-column', action='store_true',
                   help='allow off-grid columns too (not recommended, see module docstring)')
    args = p.parse_args()

    qin, qth, D = load_grid(os.path.join(HERE, args.lut) if not os.path.isabs(args.lut)
                            else args.lut)

    th_mv = [t if args.unit == 'mV' else t * CvG for t in args.thresholds]
    th_e = [t / CvG if args.unit == 'mV' else t for t in args.thresholds]

    print(f'LUT   : {args.lut}   ({len(qin)} Qin rows x {len(qth)} Qth columns)')
    print(f'CvG   : {CvG} mV/e-')
    print(f'columns restricted to the 25 e- grid: {not args.any_column}\n')

    print(f'{"th":<5}{"mV":>11}{"e-":>9}{"col":>7}{"cells":>7}{"Qin range":>16}'
          f'{"min":>9}{"median":>9}{"max":>9}')
    pooled = []
    for k, (mv, e) in enumerate(zip(th_mv, th_e), 1):
        col = pick_column(qth, e, grid25=not args.any_column)
        v = D[:, list(qth).index(col)]
        ok = np.isfinite(v)
        if not ok.any():
            print(f'th{k:<4}{mv:11.3f}{e:9.1f}{col:7d}{0:7d}   (column empty)')
            continue
        d, q = v[ok], qin[ok]
        pooled.append(d)
        rng = f'{q.min():.0f}..{q.max():.0f}'
        print(f'th{k:<4}{mv:11.3f}{e:9.1f}{col:7d}{ok.sum():7d}{rng:>16}'
              f'{d.min():9.3f}{np.median(d):9.3f}{d.max():9.3f}')

    if pooled:
        a = np.concatenate(pooled)
        print(f'\n{"pooled over the three columns":<40}{a.size:6d} cells')
        print(f'  MIN    = {a.min():.3f} ns')
        print(f'  MEDIAN = {np.median(a):.3f} ns')
        print('\nThese are per-cell statistics, unweighted by how often each Q_in occurs.')


if __name__ == '__main__':
    main()
