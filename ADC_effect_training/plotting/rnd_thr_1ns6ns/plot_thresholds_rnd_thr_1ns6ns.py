import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr.jsonl")

records = [json.loads(l) for l in open(threshold_runs_path) if l.strip()]
runs = [r for r in records if not r.get("stuck", False)]

thr_names = ["threshold_0", "threshold_1", "threshold_2"]
colors = {"threshold_0": "tab:blue", "threshold_1": "tab:orange", "threshold_2": "tab:green"}

per_run_data = []
for r in runs:
    path = os.path.join(r["checkpoint_dir"], "soft_quantizer_state_log.csv")
    data = {thr: [] for thr in thr_names}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for thr in thr_names:
                data[thr].append(float(row[thr]))
    per_run_data.append(data)

n_runs = len(per_run_data)
min_len = min(len(d["threshold_0"]) for d in per_run_data)
epochs = list(range(min_len))

fig, ax = plt.subplots(figsize=(10, 9))

for thr in thr_names:
    for d in per_run_data:
        ax.plot(d[thr][:min_len], epochs, color=colors[thr], alpha=0.15, lw=1)

for thr in thr_names:
    arr = np.array([d[thr][:min_len] for d in per_run_data])
    median = np.median(arr, axis=0)
    ax.plot(median, epochs, color=colors[thr], lw=2.5, label=f"{thr} median (->{median[-1]:.1f})")

ax.set_xlabel("threshold value")
ax.set_ylabel("epoch (training progresses downward)")
ax.invert_yaxis()
ax.set_title(f"random initial thresholds: threshold convergence -- all {n_runs} seeds (faint) + median (bold)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_1ns6ns/run_thresholds_rnd_thr_1ns6ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")
