"""
Evaluate the pixelAV-matched Stage 2 (non-quantized Conv2D, frozen
hard-digitized thresholds, no-noise TFRs, MDMM) run: residuals, pulls, and
summary plots. Same pattern as eval_pixelav_matched_part1p5_mdmm.py, but
loading CreateNonQuantizedModel from train_conv2d_part2_pixelav_matched_mdmm.py
instead of building a ViT.
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

import tensorflow as tf
import tensorflow_probability as tfp  # must precede `from qkeras import *` -- see losses.loss import-order note

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "raw_pixelAV_training"))

import utils
utils.check_GPU()

from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator
from losses.loss import custom_loss
from train_conv2d_part2_pixelav_matched_mdmm import CreateNonQuantizedModel

pi = 3.14159265359
minval = 1e-9

# ---------------------------------------------------------------- configuration
dataset_base_dir = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
rnd_thr_dir = os.path.join(dataset_base_dir, "trained_models_rnd_thr_mdmm")
part2_output_dir = os.path.join(rnd_thr_dir, "part2_conv2d")
tfrecords_dir_val = os.path.join(dataset_base_dir, "TFR_files", "2t", "TFR_val")  # no-noise
out_base = os.path.dirname(os.path.abspath(__file__))
CASE_TAG = "pixelAV-matched, Stage 2 non-quantized Conv2D, frozen thresholds, no_noise, MDMM"

# ----------------------------------------------------------- run selection
parser = argparse.ArgumentParser()
parser.add_argument('--fingerprint', type=str, default=None,
                    help="evaluate this run instead of the best-val-loss one")
args = parser.parse_args()

summary_paths = glob.glob(os.path.join(part2_output_dir, "**", "summary.json"), recursive=True)
if not summary_paths:
    raise SystemExit(f"No Stage 2 summary.json found under {part2_output_dir} -- has a run completed yet?")
summaries = [json.load(open(p)) for p in summary_paths]
if args.fingerprint:
    record = next(r for r in summaries if r["fingerprint"] == args.fingerprint)
else:
    record = min(summaries, key=lambda r: r["best_val_loss"])

fingerprint = record["fingerprint"]
print(f"Evaluating run {fingerprint} (seed={record['seed']}, "
      f"best_val_loss={record['best_val_loss']:.1f}, "
      f"fixed_thresholds={[round(t,2) for t in record['fixed_thresholds']]})")

plot_dir = os.path.join(out_base, fingerprint)
os.makedirs(plot_dir, exist_ok=True)

# --------------------------------------------------- build model + load weights
model = CreateNonQuantizedModel(shape=(16, 16, 2), output=14, n_filters=5, pool_size=3)
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)

checkpoints_dir = os.path.join(record["checkpoint_dir"], "checkpoints")

def extract_val_metric(fname):
    try:
        return float(fname.split("-v")[-1].replace(".weights.h5", ""))
    except Exception:
        return float("inf")

ckpts = glob.glob(os.path.join(checkpoints_dir, "weights.*.weights.h5"))
best_ckpt = min(ckpts, key=lambda f: extract_val_metric(os.path.basename(f)))
print(f"Loading best checkpoint (val_loss={extract_val_metric(os.path.basename(best_ckpt)):.2f}): {os.path.basename(best_ckpt)}")
model.load_weights(best_ckpt)

# ------------------------------------------------------------- data + predict
# Same digitize=True/digitize_thresholds/digitize_levels as training's own
# validation_generator; shuffle=False here so model.predict() and the
# truth-collection loop see identical example order.
test_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir=tfrecords_dir_val,
    shuffle=False,
    quantize=False,
    digitize=True,
    digitize_thresholds=record["fixed_thresholds"],
    digitize_levels=record["fixed_levels"],
)
labels_scale = test_generator.labels_scale
print(f"labels_scale = {labels_scale}")

p_test = model.predict(test_generator)

complete_truth = None
for _, y in test_generator:
    y = np.asarray(y)
    complete_truth = y if complete_truth is None else np.concatenate((complete_truth, y), axis=0)

# --------------------------------------------- dataframe with sigmas (das convention)
df = pd.DataFrame(p_test, columns=['x','M11','y','M22','cotA','M33','cotB','M44',
                                   'M21','M31','M32','M41','M42','M43'])
df['xtrue']    = complete_truth[:,0]
df['ytrue']    = complete_truth[:,1]
df['cotAtrue'] = complete_truth[:,2]
df['cotBtrue'] = complete_truth[:,3]

df['M11'] = minval + np.maximum(df['M11'], 0)
df['M22'] = minval + np.maximum(df['M22'], 0)
df['M33'] = minval + np.maximum(df['M33'], 0)
df['M44'] = minval + np.maximum(df['M44'], 0)

df['sigmax']    = abs(df['M11'])
df['sigmay']    = np.sqrt(df['M21']**2 + df['M22']**2)
df['sigmacotA'] = np.sqrt(df['M31']**2 + df['M32']**2 + df['M33']**2)
df['sigmacotB'] = np.sqrt(df['M41']**2 + df['M42']**2 + df['M43']**2 + df['M44']**2)

df['pullx']    = (df['xtrue']    - df['x'])    / df['sigmax']
df['pully']    = (df['ytrue']    - df['y'])    / df['sigmay']
df['pullcotA'] = (df['cotAtrue'] - df['cotA']) / df['sigmacotA']
df['pullcotB'] = (df['cotBtrue'] - df['cotB']) / df['sigmacotB']

df.to_csv(os.path.join(plot_dir, "predictions.csv"), header=True, index=False)

# ---------------------------------------------------- predicted angle distributions
# Standard collapse diagnostic (same pattern as every other
# plot_pred_angle_dists_*.py in this repo): predicted vs true cotA/cotB
# distributions -- a collapsed/shrunk prediction shows up here as a narrow
# spike vs. the true distribution's spread.
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
fig.suptitle(f"Predicted vs true angle distributions: {fingerprint} ({CASE_TAG})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "pred_angle_dists.png"), dpi=120)
plt.close()

# ------------------------------------------------------------ residual summary
VARS = [('x', 'xtrue', labels_scale[0], r'$x$ [um]'),
        ('y', 'ytrue', labels_scale[1], r'$y$ [um]'),
        ('cotA', 'cotAtrue', labels_scale[2], r'$\cot\alpha$'),
        ('cotB', 'cotBtrue', labels_scale[3], r'$\cot\beta$')]

summary_lines = [f"run {fingerprint} ({CASE_TAG}), checkpoint {os.path.basename(best_ckpt)}"]
for var, tvar, scale, name in VARS:
    res = (df[tvar] - df[var]) * scale
    summary_lines.append(f"{var:5s}: residual mean = {np.mean(res):+.4f}, std = {np.std(res):.4f}"
                         f"  (scale {scale:.4f})")
print("\n".join(summary_lines))
with open(os.path.join(plot_dir, "residual_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines) + "\n")

# ------------------------------------------------------------ residual hists
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (var, tvar, scale, name) in zip(axes.flat, VARS):
    res = (df[tvar] - df[var]) * scale
    ax.hist(res, bins=50, histtype='step')
    ax.set_xlabel(f"True - predicted {name}")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
fig.suptitle(f"Residuals: {fingerprint} ({CASE_TAG})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "residual_hists.png"), dpi=120)
plt.close()

# --------------------------------------------------------------- sigma hists
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sigma_ranges = {'sigmax': (0, 35), 'sigmay': (0, 6), 'sigmacotA': (0, 1.0), 'sigmacotB': (0, 0.6)}
for ax, ((var, tvar, scale, name), (svar, srange)) in zip(axes.flat, zip(VARS, sigma_ranges.items())):
    ax.hist(df[svar] * scale, bins=np.linspace(*srange, 50), histtype='step')
    ax.set_xlabel(f"predicted sigma {name}")
    ax.grid(True, alpha=0.3)
fig.suptitle(f"Predicted uncertainties: {fingerprint} ({CASE_TAG})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "sigma_hists.png"), dpi=120)
plt.close()

# ------------------------------------------------------------------ pull plot
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def pull_plot(ax, var, name):
    h = ax.hist(df[var], bins=np.linspace(-5, 5, 50), histtype='step')
    ax.set_xlabel(name)
    ax.set_yscale('log')
    ydata = h[0]
    xdata = h[1][:-1] + 5.0/50.
    try:
        pars, _ = curve_fit(gauss, xdata, ydata, p0=[ydata.max(), 0.0, 1.0])
        xbins = np.linspace(-5, 5, 100)
        ax.plot(xbins, gauss(xbins, *pars), color='black')
        ax.text(-4.8, ydata.max()/5,  rf"$\mu$={pars[1]:.2f}")
        ax.text(-4.8, ydata.max()/20, rf"$\sigma$={abs(pars[2]):.2f}")
    except Exception as e:
        print(f"pull fit failed for {var}: {e}")
    ax.set_ylim(0.5, None)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 7))
pull_plot(axes[0][0], 'pullx',    r'$x$ pull')
pull_plot(axes[0][1], 'pully',    r'$y$ pull')
pull_plot(axes[1][0], 'pullcotA', r'$\cot\alpha$ pull')
pull_plot(axes[1][1], 'pullcotB', r'$\cot\beta$ pull')
fig.suptitle(f"Pulls: {fingerprint} ({CASE_TAG})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "pull.png"), dpi=120)
plt.close()

# -------------------------------------------------------- summary (money) plot
def residual_plot(ax, thisdf, var1, var2, name, scaling=1.0):
    nbins = 15
    var1_scaled = thisdf[var1] * scaling
    var2_scaled = thisdf[var2] * scaling
    residual_scaled = var1_scaled - var2_scaled
    xmin, xmax = np.min(var1_scaled), np.max(var1_scaled)
    step = (xmax - xmin) / nbins
    sns.regplot(x=var1_scaled, y=residual_scaled,
                x_bins=np.linspace(xmin, xmax, nbins), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    means, upbar, downbar = [], [], []
    for i in range(nbins):
        sel = (var1_scaled > xmin + i*step) & (var1_scaled < xmin + (i+1)*step)
        means.append(np.mean(residual_scaled[sel]))
        band = np.mean(thisdf['sigma'+var2][sel] * scaling)
        upbar.append(means[i] + band)
        downbar.append(means[i] - band)
    ax.fill_between(x=np.linspace(xmin, xmax, nbins), y1=upbar, y2=downbar, alpha=0.2)

def inverse_cot(cota):
    a = np.arctan(1.0/cota)
    a[np.where(a < 0)] = a[np.where(a < 0)] + pi
    return a

def residual_plot_deg(ax, thisdf, var1, var2, name, scaling=1.0):
    angle     = inverse_cot(thisdf[var2].values * scaling) * 180/pi
    angleup   = abs(inverse_cot((thisdf[var2].values + thisdf['sigma'+var2].values) * scaling) * 180/pi - angle)
    angledown = abs(inverse_cot((thisdf[var2].values - thisdf['sigma'+var2].values) * scaling) * 180/pi - angle)
    angletrue = inverse_cot(thisdf[var1].values * scaling) * 180/pi

    nbins = 15
    xmin, xmax = np.min(angletrue), np.max(angletrue)
    step = (xmax - xmin) / nbins
    resid = angletrue - angle
    sns.regplot(x=angletrue, y=resid,
                x_bins=np.linspace(xmin, xmax, nbins), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    means, upbar, downbar = [], [], []
    for i in range(nbins):
        sel = (angletrue > xmin + i*step) & (angletrue < xmin + (i+1)*step)
        means.append(np.mean(resid[sel]))
        upbar.append(means[i] + np.mean(angleup[sel]))
        downbar.append(means[i] - np.mean(angledown[sel]))
    ax.fill_between(x=np.linspace(xmin, xmax, nbins), y1=upbar, y2=downbar, alpha=0.2)

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
fig.tight_layout(pad=4.0)
residual_plot(axes[0][0], df, 'xtrue', 'x', r'$x$ [um]', scaling=labels_scale[0])
axes[0][0].axvline(-25, color='gray', linestyle=':')
axes[0][0].axvline( 25, color='gray', linestyle=':')
residual_plot(axes[0][1], df, 'ytrue', 'y', r'$y$ [um]', scaling=labels_scale[1])
axes[0][1].axvline(-6.25, color='gray', linestyle=':')
axes[0][1].axvline( 6.25, color='gray', linestyle=':')
residual_plot_deg(axes[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', scaling=labels_scale[2])
axes[1][0].axvline(90, color='gray', linestyle=':')
residual_plot_deg(axes[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', scaling=labels_scale[3])
axes[1][1].axvline(90, color='gray', linestyle=':')
fig.suptitle(f"Summary: {fingerprint} ({CASE_TAG})", y=1.0)
plt.savefig(os.path.join(plot_dir, "summary.png"), dpi=120, bbox_inches='tight')
plt.close()

print(f"All plots saved to {plot_dir}")
