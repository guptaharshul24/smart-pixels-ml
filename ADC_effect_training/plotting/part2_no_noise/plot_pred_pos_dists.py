"""Distributions of the raw predicted x/y position vs truth for a given run.

The position counterpart to plot_pred_angle_dists_*.py -- same pattern, same
predictions.csv, but for x-midplane / y-midplane instead of cotA/cotB. Values are
in the normalized (LABELS_SCALE-divided) units the model works in, matching the
angle plot's convention; multiply by LABELS_SCALE to get um.

Drop this file next to an eval script and run it after that eval has produced
predictions.csv for the same --fingerprint. Reads from, and writes into,
<script dir>/<fingerprint>/.

Usage: python plot_pred_pos_dists.py --fingerprint <fp>
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--fingerprint", type=str, required=True)
parser.add_argument("--title", type=str, default="",
                    help="run description for the figure title")
args = parser.parse_args()
fingerprint = args.fingerprint

here = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(here, fingerprint, "predictions.csv")
if not os.path.exists(pred_path):
    raise SystemExit(f"No predictions.csv for {fingerprint} -- run the eval "
                     f"script for this fingerprint first.")
df = pd.read_csv(pred_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, name in [(axes[0], "x", "x-midplane"), (axes[1], "y", "y-midplane")]:
    lo = min(df[var + "true"].min(), df[var].quantile(0.01))
    hi = max(df[var + "true"].max(), df[var].quantile(0.99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(df[var + "true"], bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"true (std {df[var+'true'].std():.3f})")
    ax.hist(df[var], bins=bins, histtype="stepfilled", color="tab:blue", alpha=0.6,
            label=(f"predicted (std {df[var].std():.3f}, "
                   f"corr {np.corrcoef(df[var], df[var+'true'])[0,1]:.3f})"))
    ax.set_xlabel(f"{name} (normalized)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Predicted vs true position distributions: {args.title or fingerprint}")
plt.tight_layout()
out = os.path.join(here, fingerprint, "pred_pos_dists.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
