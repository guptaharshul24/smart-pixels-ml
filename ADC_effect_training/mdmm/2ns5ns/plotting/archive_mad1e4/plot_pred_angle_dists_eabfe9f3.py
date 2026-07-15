"""Distributions of the raw predicted cotA/cotB for the MAD-constraint 1e4
run eabfe9f3 (2ns/5ns, in-progress best ckpt) -- checks whether the MAD-satisfying
spread is honest bulk movement or a wide, truth-uncorrelated spray. Reads the
predictions.csv saved by eval_transformer_2ns5ns_mdmm.py --fingerprint eabfe9f3."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
fingerprint = "eabfe9f3"
df = pd.read_csv(os.path.join(here, fingerprint, "predictions.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, var, name in [(axes[0], "cotA", r"$\cot\alpha$"), (axes[1], "cotB", r"$\cot\beta$")]:
    lo = min(df[var + "true"].min(), df[var].quantile(0.01))
    hi = max(df[var + "true"].max(), df[var].quantile(0.99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(df[var + "true"], bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"true (std {df[var+'true'].std():.3f})")
    ax.hist(df[var], bins=bins, histtype="stepfilled", color="tab:purple", alpha=0.6,
            label=f"predicted (std {df[var].std():.2f}, MAD {np.abs(df[var]-df[var].mean()).mean():.2f})")
    ax.set_xlabel(f"{name} (normalized)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Predicted vs true angle distributions: MDMM MAD-constraint 1e4 2ns/5ns transformer, "
             f"in-progress ({fingerprint})\n"
             "MAD target satisfied via honest bulk spread, but one-sided and truth-uncorrelated")
plt.tight_layout()
out = os.path.join(here, fingerprint, "pred_angle_dists.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
