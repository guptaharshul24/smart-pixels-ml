"""
Verifies the pixelAV-matched subsample (built by
../filter_subsample_pixelav.py) actually has the properties it was selected
for: cotBeta restricted to |cotBeta|<2, pT still flat, and the realized
cotAlpha/x-midplane/y-midplane shapes -- self-checks on the output alone, not
a comparison against the ADC-effects (frontend) dataset (that dataset was
built by a different process and isn't a valid reference here, per
2026-09-02 discussion).

Reads directly from the parquet labels (dataset_3srb_16x16_50x12P5_
centeredIncidence/{train,test}/*.parquet) -- these 5 quantities are already
final at this stage, unaffected by the separate noisy-TFR generation step.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
COLS = ["cotBeta", "cotAlpha", "pt", "x-midplane", "y-midplane"]
here = os.path.dirname(os.path.abspath(__file__))

files = (sorted(glob.glob(os.path.join(DATA_DIR, "train", "part.*.parquet"))) +
         sorted(glob.glob(os.path.join(DATA_DIR, "test", "part.*.parquet"))))
df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
print(f"n = {len(df):,} across {len(files)} files (train+test combined)")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flat
specs = [
    ("cotBeta", r"$\cot\beta$", None),
    ("cotAlpha", r"$\cot\alpha$", None),
    ("pt", r"$p_T$", None),
    ("x-midplane", "x-midplane [um]", None),
    ("y-midplane", "y-midplane [um]", None),
]
for ax, (col, label, xlim) in zip(axes, specs):
    v = df[col].to_numpy()
    ax.hist(v, bins=80, histtype="step", color="tab:blue")
    ax.set_xlabel(label)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f"mean={v.mean():+.3f}\nstd={v.std():.3f}\nrange=[{v.min():+.3f},{v.max():+.3f}]",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    if xlim:
        ax.set_xlim(*xlim)
axes[-1].axis("off")

fig.suptitle(f"pixelAV-matched subsample (n={len(df):,}): distribution self-check", y=1.0)
plt.tight_layout()
out = os.path.join(here, "distributions_pixelav_matched.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved to {out}")

# explicit pT flatness re-check
from scipy.stats import kstest
pt = df["pt"].to_numpy()
u = (pt - pt.min()) / (pt.max() - pt.min())
ks = kstest(u, "uniform")
print(f"pT KS-vs-uniform: stat={ks.statistic:.4f} p={ks.pvalue:.3g}")
print(f"cotBeta range: [{df['cotBeta'].min():+.4f}, {df['cotBeta'].max():+.4f}]  "
      f"(cut was |cotBeta|<2.0)")
