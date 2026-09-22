"""
Visualizes the y-midplane / cotBeta containment-induced shift for the
ADC-effects (2ns5ns) dataset's contained/ source -- the same effect
quantified numerically for both this dataset and the pixelAV-matched one in
this session's discussion (2026-09-09): containment (original_atEdge==False)
is not symmetric in cotBeta, so it drags the surviving sample's mean cotBeta
away from 0, which -- via the fixed relation y-midplane = y-entry - 50*cotBeta
(z-entry is a dataset-wide constant 100um, so sensor_thickness/2 - z-entry =
-50 exactly) -- pushes mean y-midplane positive.

Same panel style as raw_pixelAV_training/plotting/plot_distributions_pixelav_matched.py
(single-color step histogram + mean/std/range text box per panel), but laid
out as cotBeta/y-midplane (rows) x raw/contained/contained+|cotBeta|<2
(columns) to show the shift across the three selection stages side by side.

Reads only the 5 needed columns (column-pruned parquet read) across all 100
files (80 train + 20 test) for the full-sample distribution, not a subsample.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/contained"
COLS = ["cotBeta", "y-midplane", "y-entry", "z-entry", "original_atEdge"]
here = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting"

files = sorted(glob.glob(f"{BASE}/train/*.parquet")) + sorted(glob.glob(f"{BASE}/test/*.parquet"))
print(f"reading {len(files)} files...")
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
print(f"total rows: {len(df):,}")

raw = df
contained = df[~df["original_atEdge"]]
contained_cotb = contained[contained["cotBeta"].abs() < 2.0]

stages = [("raw (no cuts)", raw), ("+ containment", contained), ("+ containment + |cotBeta|<2", contained_cotb)]
rows = [("cotBeta", r"$\cot\beta$"), ("y-midplane", "y-midplane [um]")]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for i, (col, label) in enumerate(rows):
    for j, (stage_name, d) in enumerate(stages):
        ax = axes[i][j]
        v = d[col].to_numpy()
        ax.hist(v, bins=80, histtype="step", color="tab:blue")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.set_title(f"{stage_name} (n={len(d):,})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"mean={v.mean():+.3f}\nstd={v.std():.3f}\nrange=[{v.min():+.3f},{v.max():+.3f}]",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

fig.suptitle("ADC-effects (2ns5ns) dataset: cotBeta/y-midplane across selection stages\n"
             "(y-midplane = y-entry - 50*cotBeta, z-entry constant at 100um)", y=1.0)
plt.tight_layout()
out = f"{here}/ymidplane_containment_2ns5ns.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved to {out}")
