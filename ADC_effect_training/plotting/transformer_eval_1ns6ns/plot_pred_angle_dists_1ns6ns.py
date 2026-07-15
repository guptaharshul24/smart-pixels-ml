"""Distributions of the raw predicted cotA/cotB for the pre-MDMM 1ns6ns best run
(3b9c78f7) -- visualizes the angle collapse: predictions pile up at a constant
while the truth spans its full range. Reads the predictions.csv saved by
eval_transformer_1ns6ns.py."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
fingerprint = "3b9c78f7"
df = pd.read_csv(os.path.join(here, fingerprint, "predictions.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, name in [(axes[0], "cotA", r"$\cot\alpha$"), (axes[1], "cotB", r"$\cot\beta$")]:
    bins = np.linspace(-1.5, 1.5, 80)
    ax.hist(df[var + "true"], bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"true (std {df[var+'true'].std():.3f})")
    ax.hist(df[var], bins=bins, histtype="stepfilled", color="tab:red", alpha=0.6,
            label=f"predicted (std {df[var].std():.4f})")
    ax.set_xlabel(f"{name} (normalized)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Predicted vs true angle distributions: pre-MDMM 1ns/6ns transformer ({fingerprint})\n"
             "predictions collapse to a constant while truth spans the full range")
plt.tight_layout()
out = os.path.join(here, fingerprint, "pred_angle_dists.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
