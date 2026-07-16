"""
Stage 2 (ViT, Part 1.5) vs Stage 3 (plain Conv2D, Part 2) comparison, 2ns/5ns.
Overlays both models' residuals+uncertainty bands and pull distributions on
shared axes -- adapted from das's performance_plots.ipynb multi-model overlay
pattern (residual_plot/residual_plot_deg binned-mean + sigma-band style).

Stage 4 (QConv2D, Part 2.5) is deliberately excluded for now -- no valid
converged run exists yet (all 10/10 attempts stuck), so there's nothing
meaningful to overlay. Add it here once a real result exists.

Reads the predictions.csv already written by each stage's own eval script;
does not re-run inference.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

pi = 3.14159265359
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

STAGE2_CSV = os.path.join(repo_root, "ADC_effect_training/plotting/part1p5/00bbfea6/predictions.csv")
STAGE3_CSV = os.path.join(repo_root, "ADC_effect_training/plotting/part2/986827aa/predictions.csv")
STAGE2_LABEL = "Stage 2: ViT (Part 1.5)"
STAGE3_LABEL = "Stage 3: plain Conv2D (Part 2)"
STAGE2_COLOR = "tab:red"
STAGE3_COLOR = "tab:blue"

# Same 2ns5ns val set for both -- confirmed identical labels_scale printed by
# both stages' own eval runs.
LABELS_SCALE = [123.6133301, 31.00504472, 6.51238097, 1.84647953]

df2 = pd.read_csv(STAGE2_CSV)
df3 = pd.read_csv(STAGE3_CSV)

# ------------------------------------------------------- residual+uncertainty
def residual_plot(ax, df, var1, var2, name, color, label, scaling=1.0, nbins=15):
    v1 = df[var1] * scaling
    v2 = df[var2] * scaling
    resid = v1 - v2
    xmin, xmax = np.min(v1), np.max(v1)
    step = (xmax - xmin) / nbins
    sns.regplot(x=v1, y=resid, x_bins=np.linspace(xmin, xmax, nbins),
                fit_reg=None, marker='.', ax=ax, color=color, label=label)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    means, upbar, downbar = [], [], []
    for i in range(nbins):
        sel = (v1 > xmin + i * step) & (v1 < xmin + (i + 1) * step)
        means.append(np.mean(resid[sel]))
        band = np.mean(df['sigma' + var2][sel] * scaling)
        upbar.append(means[i] + band)
        downbar.append(means[i] - band)
    ax.fill_between(x=np.linspace(xmin, xmax, nbins), y1=upbar, y2=downbar, alpha=0.2, color=color)

def inverse_cot(cota):
    a = np.arctan(1.0 / cota)
    a[np.where(a < 0)] = a[np.where(a < 0)] + pi
    return a

def residual_plot_deg(ax, df, var1, var2, name, color, label, scaling=1.0, nbins=15):
    angle = inverse_cot(df[var2].values * scaling) * 180 / pi
    angleup = abs(inverse_cot((df[var2].values + df['sigma' + var2].values) * scaling) * 180 / pi - angle)
    angledown = abs(inverse_cot((df[var2].values - df['sigma' + var2].values) * scaling) * 180 / pi - angle)
    angletrue = inverse_cot(df[var1].values * scaling) * 180 / pi
    xmin, xmax = np.min(angletrue), np.max(angletrue)
    step = (xmax - xmin) / nbins
    resid = angletrue - angle
    sns.regplot(x=angletrue, y=resid, x_bins=np.linspace(xmin, xmax, nbins),
                fit_reg=None, marker='.', ax=ax, color=color, label=label)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    means, upbar, downbar = [], [], []
    for i in range(nbins):
        sel = (angletrue > xmin + i * step) & (angletrue < xmin + (i + 1) * step)
        means.append(np.mean(resid[sel]))
        upbar.append(means[i] + np.mean(angleup[sel]))
        downbar.append(means[i] - np.mean(angledown[sel]))
    ax.fill_between(x=np.linspace(xmin, xmax, nbins), y1=upbar, y2=downbar, alpha=0.2, color=color)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.tight_layout(pad=4.5)

residual_plot(axes[0][0], df2, 'xtrue', 'x', r'$x$ [um]', STAGE2_COLOR, STAGE2_LABEL, scaling=LABELS_SCALE[0])
residual_plot(axes[0][0], df3, 'xtrue', 'x', r'$x$ [um]', STAGE3_COLOR, STAGE3_LABEL, scaling=LABELS_SCALE[0])
axes[0][0].axhline(0, alpha=0.4, ls='dashed', color='gray')
axes[0][0].set_title(r'$x$ residual + uncertainty')
axes[0][0].legend(loc='upper left', fontsize=8)

residual_plot(axes[0][1], df2, 'ytrue', 'y', r'$y$ [um]', STAGE2_COLOR, STAGE2_LABEL, scaling=LABELS_SCALE[1])
residual_plot(axes[0][1], df3, 'ytrue', 'y', r'$y$ [um]', STAGE3_COLOR, STAGE3_LABEL, scaling=LABELS_SCALE[1])
axes[0][1].axhline(0, alpha=0.4, ls='dashed', color='gray')
axes[0][1].set_title(r'$y$ residual + uncertainty')
axes[0][1].legend(loc='upper left', fontsize=8)

residual_plot_deg(axes[1][0], df2, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', STAGE2_COLOR, STAGE2_LABEL, scaling=LABELS_SCALE[2])
residual_plot_deg(axes[1][0], df3, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', STAGE3_COLOR, STAGE3_LABEL, scaling=LABELS_SCALE[2])
axes[1][0].axhline(0, alpha=0.4, ls='dashed', color='gray')
axes[1][0].set_title(r'$\alpha$ residual + uncertainty')
axes[1][0].legend(loc='upper left', fontsize=8)

residual_plot_deg(axes[1][1], df2, 'cotBtrue', 'cotB', r'$\beta$ [deg]', STAGE2_COLOR, STAGE2_LABEL, scaling=LABELS_SCALE[3])
residual_plot_deg(axes[1][1], df3, 'cotBtrue', 'cotB', r'$\beta$ [deg]', STAGE3_COLOR, STAGE3_LABEL, scaling=LABELS_SCALE[3])
axes[1][1].axhline(0, alpha=0.4, ls='dashed', color='gray')
axes[1][1].set_title(r'$\beta$ residual + uncertainty')
axes[1][1].legend(loc='upper left', fontsize=8)

fig.suptitle("Stage 2 (ViT) vs Stage 3 (plain Conv2D): residuals + uncertainty, 2ns/5ns", y=1.0)
out1 = os.path.join(here, "residuals_stage2_vs_stage3.png")
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close()
print(f"saved to {out1}")

# ------------------------------------------------------------------ pull plot
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def pull_plot(ax, df, var, name, color, label):
    h = ax.hist(df[var], bins=np.linspace(-5, 5, 50), histtype='step', color=color, label=label)
    ax.set_xlabel(name)
    ax.set_yscale('log')
    ydata, xdata = h[0], h[1][:-1] + 5.0 / 50.
    try:
        pars, _ = curve_fit(gauss, xdata, ydata, p0=[ydata.max(), 0.0, 1.0])
        xbins = np.linspace(-5, 5, 100)
        ax.plot(xbins, gauss(xbins, *pars), color=color, linestyle='--')
    except Exception as e:
        print(f"pull fit failed for {label}/{var}: {e}")
    ax.set_ylim(0.5, None)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(11, 8))
for var, name, ax in [('pullx', r'$x$ pull', axes[0][0]), ('pully', r'$y$ pull', axes[0][1]),
                       ('pullcotA', r'$\cot\alpha$ pull', axes[1][0]), ('pullcotB', r'$\cot\beta$ pull', axes[1][1])]:
    pull_plot(ax, df2, var, name, STAGE2_COLOR, STAGE2_LABEL)
    pull_plot(ax, df3, var, name, STAGE3_COLOR, STAGE3_LABEL)
fig.suptitle("Stage 2 (ViT) vs Stage 3 (plain Conv2D): pulls, 2ns/5ns")
plt.tight_layout()
out2 = os.path.join(here, "pulls_stage2_vs_stage3.png")
plt.savefig(out2, dpi=120)
plt.close()
print(f"saved to {out2}")
