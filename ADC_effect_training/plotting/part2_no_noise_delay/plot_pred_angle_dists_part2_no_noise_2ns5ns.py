"""Distributions of the raw predicted cotA/cotB vs truth for a given Part 2
no-noise run. Reads the predictions.csv saved by
eval_part2_no_noise_2ns5ns.py (must be run first for the same --fingerprint).
See plotting/part2/plot_pred_angle_dists_part2_2ns5ns.py for the full
explanation.

Usage: python plot_pred_angle_dists_part2_no_noise_2ns5ns.py --fingerprint <fp>
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
args = parser.parse_args()
fingerprint = args.fingerprint

here = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(here, fingerprint, "predictions.csv")
if not os.path.exists(pred_path):
    raise SystemExit(f"No predictions.csv for {fingerprint} -- run "
                      f"eval_part2_no_noise_2ns5ns.py --fingerprint {fingerprint} first.")
df = pd.read_csv(pred_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, name in [(axes[0], "cotA", r"$\cot\alpha$"), (axes[1], "cotB", r"$\cot\beta$")]:
    lo = min(df[var + "true"].min(), df[var].quantile(0.01))
    hi = max(df[var + "true"].max(), df[var].quantile(0.99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(df[var + "true"], bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"true (std {df[var+'true'].std():.3f})")
    ax.hist(df[var], bins=bins, histtype="stepfilled", color="tab:blue", alpha=0.6,
            label=f"predicted (std {df[var].std():.3f}, corr {np.corrcoef(df[var], df[var+'true'])[0,1]:.3f})")
    ax.set_xlabel(f"{name} (normalized)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Predicted vs true angle distributions: Part 2 no-noise non-quantized Conv2D 2ns/5ns ({fingerprint})")
plt.tight_layout()
out = os.path.join(here, fingerprint, "pred_angle_dists.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
