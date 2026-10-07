"""Predicted-vs-true distribution plots, for both angles and position.

Called from inside every eval so these cannot be forgotten. Residual width alone
does not reveal prediction collapse: a model can show a plausible residual spread
while its predicted distribution is far narrower than truth, or even
anti-correlated with it. Overlaying predicted against true is what exposes that,
and the printed std ratio and correlation make it quantitative.

Reads <plot_dir>/predictions.csv, which every eval writes, and emits
<plot_dir>/pred_angle_dists.png and <plot_dir>/pred_pos_dists.png.

Values are in the normalized (LABELS_SCALE-divided) units the model works in,
matching the convention the standalone plotting scripts used; multiply by
LABELS_SCALE for physical units.
"""
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = {
    "angle": [("cotA", r"$\cot\alpha$"), ("cotB", r"$\cot\beta$")],
    "pos": [("x", "x-midplane"), ("y", "y-midplane")],
}


def _panel(ax, df, var, name):
    lo = min(df[var + "true"].min(), df[var].quantile(0.01))
    hi = max(df[var + "true"].max(), df[var].quantile(0.99))
    bins = np.linspace(lo, hi, 80)
    corr = np.corrcoef(df[var], df[var + "true"])[0, 1]
    ratio = df[var].std() / df[var + "true"].std()
    ax.hist(df[var + "true"], bins=bins, histtype="step", color="gray", lw=1.5,
            label=f"true (std {df[var+'true'].std():.3f})")
    ax.hist(df[var], bins=bins, histtype="stepfilled", color="tab:blue", alpha=0.6,
            label=f"predicted (std {df[var].std():.3f}, corr {corr:.3f})")
    # A collapsed prediction shows up as ratio << 1; an inverted one as corr < 0.
    ax.set_xlabel(f"{name} (normalized)   std ratio {ratio:.2f}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def make_pred_dists(plot_dir, title=""):
    """Write pred_angle_dists.png and pred_pos_dists.png into plot_dir."""
    pred_path = os.path.join(plot_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        print(f"[pred_dists] no predictions.csv in {plot_dir}, skipping")
        return
    df = pd.read_csv(pred_path)
    for kind, pairs in PAIRS.items():
        missing = [v for v, _ in pairs if v not in df or v + "true" not in df]
        if missing:
            print(f"[pred_dists] {kind}: columns {missing} absent, skipping")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (var, name) in zip(axes, pairs):
            _panel(ax, df, var, name)
        label = "angle" if kind == "angle" else "position"
        # Case tags run long (architecture, condition, shift), so wrap and shrink
        # rather than letting the suptitle run off the canvas.
        head = f"Predicted vs true {label} distributions"
        full = f"{head}: {title}" if title else f"{head}: {os.path.basename(plot_dir)}"
        fig.suptitle("\n".join(textwrap.wrap(full, 95)), fontsize=10)
        plt.tight_layout()
        out = os.path.join(plot_dir, f"pred_{kind}_dists.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved to {out}")
