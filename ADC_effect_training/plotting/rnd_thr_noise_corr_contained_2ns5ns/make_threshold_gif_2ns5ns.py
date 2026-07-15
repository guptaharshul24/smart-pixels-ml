"""
Animated GIF of threshold convergence across ALL runs in the original (no
MDMM) Part-1 threshold-search campaign: replays training epoch by epoch,
threshold value on x, epoch on y (growing downward). Same data/styling as
plot_thresholds_rnd_thr_noise_corr_contained_2ns5ns.py (faint per-run lines +
bold cross-run median) but animated instead of a single static frame.

Usage: python make_threshold_gif_2ns5ns.py [--frames 200] [--fps 20]
"""
import os
import csv
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=200, help="target number of animation frames")
parser.add_argument("--fps", type=int, default=20)
args = parser.parse_args()

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6_noise_corr_contained_2ns5ns"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns.jsonl")

stuck_fingerprints = set()
if os.path.exists(threshold_runs_path):
    for line in open(threshold_runs_path):
        if line.strip():
            r = json.loads(line)
            if r.get("stuck", False):
                stuck_fingerprints.add(r["fingerprint"])

ckpt_dirs = glob.glob(os.path.join(trained_models_dir, "2t_rnd_thr_noise_corr_contained_2ns5ns_5000ep_*", "Transformer_model-*-checkpoints"))
ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] not in stuck_fingerprints]
ckpt_dirs = sorted(ckpt_dirs, key=os.path.getctime)

thr_names = ["threshold_0", "threshold_1", "threshold_2"]
colors = {"threshold_0": "tab:blue", "threshold_1": "tab:orange", "threshold_2": "tab:green"}

per_run_data, fps_list = [], []
for d in ckpt_dirs:
    path = os.path.join(d, "soft_quantizer_state_log.csv")
    if not os.path.exists(path):
        continue
    data = {thr: [] for thr in thr_names}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for thr in thr_names:
                data[thr].append(float(row[thr]))
    if not data["threshold_0"]:
        continue
    per_run_data.append(data)
    fps_list.append(os.path.basename(d).split("-")[1])

if not per_run_data:
    raise SystemExit("No runs with threshold data found.")

n_runs = len(per_run_data)
lengths = [len(d["threshold_0"]) for d in per_run_data]
max_len = max(lengths)
epochs_full = list(range(max_len))

# precompute the cross-run median at every epoch once (same logic as the
# static plot): median over whichever runs have data at that epoch
median_vals = {thr: [] for thr in thr_names}
if n_runs > 1:
    for thr in thr_names:
        for i in range(max_len):
            vals = [d[thr][i] for d in per_run_data if len(d[thr]) > i]
            median_vals[thr].append(np.median(vals) if vals else np.nan)

xmin = min(min(d[t]) for d in per_run_data for t in thr_names)
xmax = max(max(d[t]) for d in per_run_data for t in thr_names)
pad = 0.05 * (xmax - xmin)

step = max(1, max_len // args.frames)
frame_epochs = list(range(step, max_len, step)) + [max_len - 1]

fig, ax = plt.subplots(figsize=(11, 9))
faint_lines = [{t: ax.plot([], [], color=colors[t], alpha=0.15, lw=1)[0] for t in thr_names}
               for _ in per_run_data]
bold_lines = {t: ax.plot([], [], color=colors[t], lw=2.5,
                         label=(f"{t} median" if n_runs > 1 else t))[0]
              for t in thr_names}
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(max_len, 0)
ax.set_xlabel("Threshold value (mV)")
ax.set_ylabel("Epoch")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

def update(i):
    upto = frame_epochs[i]
    for run_i, d in enumerate(per_run_data):
        run_upto = min(upto, len(d["threshold_0"]) - 1)
        for t in thr_names:
            faint_lines[run_i][t].set_data(d[t][:run_upto + 1], epochs_full[:run_upto + 1])
    for t in thr_names:
        if n_runs > 1:
            bold_lines[t].set_data(median_vals[t][:upto + 1], epochs_full[:upto + 1])
        else:
            run_upto = min(upto, len(per_run_data[0][t]) - 1)
            bold_lines[t].set_data(per_run_data[0][t][:run_upto + 1], epochs_full[:run_upto + 1])
    ax.set_title(f"Threshold convergence: {n_runs} run(s), 2ns/5ns (no MDMM), epoch {upto}")
    return [l for fl in faint_lines for l in fl.values()] + list(bold_lines.values())

anim = FuncAnimation(fig, update, frames=len(frame_epochs), blit=False)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_convergence.gif")
anim.save(out, writer=PillowWriter(fps=args.fps))
print(f"saved to {out} ({len(frame_epochs)} frames, {n_runs} run(s): {fps_list})")
