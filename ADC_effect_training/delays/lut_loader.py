"""Loader for the 2D global-format VIZARD delay LUT.

The newer table (VIZARD_ADC_LUT_PixB_27c_global.csv) is a proper grid: a Qin_e
index column and one Qth_<n> column per threshold, blank where untabulated. That
replaces the older 120 ragged column-pair layout.

Two things it changes that matter:

  * Coverage reaches much closer to threshold. For the columns we use the first
    tabulated Q_in drops to 250 / 390 / 980 e- (ratios 1.11 / 1.03 / 1.01), where
    the old table stopped around 1.6-1.9x overdrive. Sub-edge clamping goes from
    the dominant effect to a corner case.
  * Q_in steps 10 e- from 230 to 1000 instead of 100 e-, which is where the delay
    gradient is steepest (about 16.6 ns per 1000 e- near the edge of column 225).

Columns are taken from the **25 e- grid only**. The table also carries finer
off-grid columns closer to arbitrary threshold values, but they are near-threshold
patches rather than full columns (Qth_984 holds two cells, spanning 990..1000 e-),
and mixing them with grid columns would make the threshold the delay model assumes
drift with Q_in. One consistent column per threshold, matching the earlier runs.

Holes are interior, not just at the edges: Qth_975 is filled at 980, 990, 1000 and
then not again until 1800. Lookups therefore operate on each column's filled rows
only, which is the same nearest-row semantics the old loader had.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LUT = os.path.join(HERE, 'VIZARD_ADC_LUT_PixB_27c_global.csv')
CvG = 0.058          # mV per electron (58 uV/e-)


def load_grid(path=DEFAULT_LUT):
    """-> (qin_e, qth_e, delay_ns) with NaN where untabulated; columns sorted."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith('Qth_')]
    if 'Qin_e' not in df.columns or not cols:
        raise SystemExit(f'{path} is not in the 2D global format')
    qth = np.array([int(c.split('_')[1]) for c in cols])
    order = np.argsort(qth)
    qin = df['Qin_e'].values.astype(float)
    return qin, qth[order], df[cols].values[:, order].astype(float) * 1e9


def pick_column(qth, target_e, grid25=True):
    """Nearest tabulated threshold column to target_e, on the 25 e- grid by default."""
    allowed = np.array([t for t in qth if t % 25 == 0]) if grid25 else np.asarray(qth)
    return int(allowed[np.argmin(np.abs(allowed - target_e))])


def column_points(qin, qth, D, col):
    """Filled (Q_in, delay) pairs for one column, ascending in Q_in."""
    v = D[:, int(np.where(qth == col)[0][0])]
    ok = np.isfinite(v)
    if not ok.any():
        raise SystemExit(f'column Qth_{col} is empty')
    return qin[ok], v[ok]


def nearest_row(ch, d, q):
    """Delay at the tabulated row closest to q; clamps at both ends."""
    i = np.clip(np.searchsorted(ch, q), 1, len(ch) - 1)
    pick = np.where(np.abs(q - ch[i - 1]) <= np.abs(q - ch[i]), i - 1, i)
    return d[pick]


def build(thresholds_mV, path=DEFAULT_LUT, grid25=True):
    """-> (columns, points) ready for per-pixel lookup.

    columns[k] is the Qth column used for threshold k; points[k] is its
    (Q_in, delay_ns) arrays.
    """
    qin, qth, D = load_grid(path)
    columns, points = [], []
    for mv in thresholds_mV:
        col = pick_column(qth, mv / CvG, grid25=grid25)
        columns.append(col)
        points.append(column_points(qin, qth, D, col))
    return columns, points


if __name__ == '__main__':
    TH = [13.001228, 21.901985, 57.135010]        # MDMM campaign medians
    cols, pts = build(TH)
    print(f'{"th":<5}{"mV":>10}{"e-":>9}{"col":>7}{"cells":>7}{"Qin range":>16}'
          f'{"d(min Qin)":>12}{"d min":>9}{"d max":>9}')
    for k, (mv, col, (ch, d)) in enumerate(zip(TH, cols, pts), 1):
        print(f'th{k:<4}{mv:10.3f}{mv / CvG:9.1f}{col:7d}{len(ch):7d}'
              f'{f"{ch.min():.0f}..{ch.max():.0f}":>16}{d[0]:12.3f}{d.min():9.3f}{d.max():9.3f}')

    # Cross-check against the old ragged-format table on cells both tables hold.
    old = os.path.join(HERE, 'VIZARD_ADC_LUT_PixB_27c_combined_sorted.csv')
    if os.path.exists(old):
        raw = pd.read_csv(old, dtype=str)
        print('\ncross-check vs old LUT (shared cells):')
        worst = 0.0
        for col, (ch, d) in zip(cols, pts):
            oc = pd.to_numeric(raw.get(f'Charge_Qth_{col}'), errors='coerce').values
            od = pd.to_numeric(raw.get(f'Qth = {col}'), errors='coerce').values * 1e9
            m = ~(np.isnan(oc) | np.isnan(od))
            oc, od = oc[m], od[m]
            shared = np.intersect1d(ch, oc)
            if not len(shared):
                print(f'  col {col}: no shared rows'); continue
            a = np.array([d[ch == q][0] for q in shared])
            b = np.array([od[oc == q][0] for q in shared])
            diff = np.abs(a - b).max(); worst = max(worst, diff)
            print(f'  col {col}: {len(shared):3d} shared rows, max |diff| = {diff:.2e} ns')
        print(f'  worst disagreement overall: {worst:.2e} ns'
              f'  -> {"identical" if worst < 1e-9 else "DIFFERS, investigate"}')
