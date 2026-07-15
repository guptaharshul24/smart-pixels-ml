import os
import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6_noise_corr"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr.jsonl")

# Known-stuck fingerprints (from the JSONL) get excluded -- they'd otherwise drag min_len
# down for everyone else and hide the converged plateau.
stuck_fingerprints = set()
if os.path.exists(threshold_runs_path):
    for line in open(threshold_runs_path):
        if line.strip():
            r = json.loads(line)
            if r.get("stuck", False):
                stuck_fingerprints.add(r["fingerprint"])

# Discover checkpoint dirs directly (not from the JSONL) so in-progress runs show up too.
# Sort by creation time so run ordering matches actual chronological run order.
ckpt_dirs = glob.glob(os.path.join(trained_models_dir, "2t_rnd_thr_noise_corr_5000ep_*", "Transformer_model-*-checkpoints"))
ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] not in stuck_fingerprints]
ckpt_dirs = sorted(ckpt_dirs, key=os.path.getctime)

thr_names = ["threshold_0", "threshold_1", "threshold_2"]
colors = {"threshold_0": "tab:blue", "threshold_1": "tab:orange", "threshold_2": "tab:green"}

per_run_data = []
for d in ckpt_dirs:
    path = os.path.join(d, "soft_quantizer_state_log.csv")
    data = {thr: [] for thr in thr_names}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for thr in thr_names:
                data[thr].append(float(row[thr]))
    per_run_data.append(data)

n_runs = len(per_run_data)
min_len = min(len(d["threshold_0"]) for d in per_run_data)  # in-progress runs have fewer epochs so far
epochs = list(range(min_len))

fig, ax = plt.subplots(figsize=(11, 9))

for thr in thr_names:
    for d in per_run_data:
        ax.plot(d[thr][:min_len], epochs, color=colors[thr], alpha=0.15, lw=1)

if n_runs > 1:
    for thr in thr_names:
        arr = np.array([d[thr][:min_len] for d in per_run_data])
        median = np.median(arr, axis=0)
        ax.plot(median, epochs, color=colors[thr], lw=2.5, label=f"{thr} median (->{median[-1]:.1f})")
else:
    for thr in thr_names:
        ax.plot(per_run_data[0][thr][:min_len], epochs, color=colors[thr], lw=2.5,
                 label=f"{thr} (->{per_run_data[0][thr][min_len-1]:.1f} so far)")

ax.set_xlabel("threshold value")
ax.set_ylabel("epoch (training progresses downward)")
ax.invert_yaxis()
ax.set_title(f"correlated noise + 5000ep: threshold convergence -- {n_runs} run(s), "
             f"{min_len} epochs (faint) + median/trend (bold)", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_noise_corr_1ns6ns/run_thresholds_rnd_thr_noise_corr_1ns6ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")
