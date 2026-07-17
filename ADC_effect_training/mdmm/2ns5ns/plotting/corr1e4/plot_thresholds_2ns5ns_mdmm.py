import os
import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_2_5_noise_corr_contained_2ns5ns_mdmm"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl")

# the JSONL is an event journal (started/completed/failed/abandoned) -- stuck
# flags only exist on completed records
stuck_fingerprints = set()
if os.path.exists(threshold_runs_path):
    for line in open(threshold_runs_path):
        if line.strip():
            r = json.loads(line)
            if r.get("status", "completed") == "completed" and r.get("stuck", False):
                stuck_fingerprints.add(r["fingerprint"])

ckpt_dirs = glob.glob(os.path.join(trained_models_dir, "2t_rnd_thr_noise_corr_contained_2ns5ns_mdmm_5000ep_*", "Transformer_model-*-checkpoints"))
ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] not in stuck_fingerprints]
ckpt_dirs = sorted(ckpt_dirs, key=os.path.getctime)

thr_names = ["threshold_0", "threshold_1", "threshold_2"]
colors = {"threshold_0": "tab:blue", "threshold_1": "tab:orange", "threshold_2": "tab:green"}

per_run_data = []
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

n_runs = len(per_run_data)
lengths = [len(d["threshold_0"]) for d in per_run_data]
min_len = min(lengths)
max_len = max(lengths)
epochs_full = list(range(max_len))

fig, ax = plt.subplots(figsize=(11, 9))

for thr in thr_names:
    for d in per_run_data:
        n = len(d[thr])
        ax.plot(d[thr], list(range(n)), color=colors[thr], alpha=0.15, lw=1)

if n_runs > 1:
    for thr in thr_names:
        median_vals = []
        for i in range(max_len):
            vals = [d[thr][i] for d in per_run_data if len(d[thr]) > i]
            median_vals.append(np.median(vals))
        ax.plot(median_vals, epochs_full, color=colors[thr], lw=2.5,
                 label=f"{thr} median (->{median_vals[-1]:.1f})")
else:
    for thr in thr_names:
        ax.plot(per_run_data[0][thr], epochs_full, color=colors[thr], lw=2.5,
                 label=f"{thr} (->{per_run_data[0][thr][-1]:.1f} so far)")

ax.set_xlabel("threshold value")
ax.set_ylabel("epoch (training progresses downward)")
ax.invert_yaxis()
epoch_str = f"{max_len} epochs" if min_len == max_len else f"completed: {max_len} epochs, in-progress: {min_len} epochs"
ax.set_title(f"MDMM + corr noise + contained + 2ns/5ns + 5000ep: threshold convergence\n"
             f"{n_runs} run(s), {epoch_str} (faint) + median/trend (bold)",
             fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_thresholds_2ns5ns_mdmm.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
