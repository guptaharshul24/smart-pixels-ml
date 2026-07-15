import os
import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"

CASES = {
    "1ns/6ns": {
        "trained_models_dir": os.path.join(base, "trained_models_1_6_noise_corr_contained"),
        "jsonl": "threshold_runs_rnd_thr_noise_corr_contained.jsonl",
        "glob": "2t_rnd_thr_noise_corr_contained_5000ep_*",
        "colors": ["#9ecae1", "#4292c6", "#08519c"],   # light -> dark blue for thr0/1/2
    },
    "2ns/5ns": {
        "trained_models_dir": os.path.join(base, "trained_models_1_6_noise_corr_contained_2ns5ns"),
        "jsonl": "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns.jsonl",
        "glob": "2t_rnd_thr_noise_corr_contained_2ns5ns_5000ep_*",
        "colors": ["#fdae6b", "#f16913", "#a63603"],   # light -> dark orange for thr0/1/2
    },
}

thr_names = ["threshold_0", "threshold_1", "threshold_2"]

fig, ax = plt.subplots(figsize=(11, 9))

for case_label, cfg in CASES.items():
    jsonl_path = os.path.join(cfg["trained_models_dir"], cfg["jsonl"])
    stuck_fingerprints = set()
    if os.path.exists(jsonl_path):
        for line in open(jsonl_path):
            if line.strip():
                r = json.loads(line)
                if r.get("stuck", False):
                    stuck_fingerprints.add(r["fingerprint"])

    ckpt_dirs = glob.glob(os.path.join(cfg["trained_models_dir"], cfg["glob"], "Transformer_model-*-checkpoints"))
    ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] not in stuck_fingerprints]

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
        if data["threshold_0"]:
            per_run_data.append(data)

    n_runs = len(per_run_data)
    max_len = max(len(d["threshold_0"]) for d in per_run_data)
    epochs_full = list(range(max_len))

    for i, thr in enumerate(thr_names):
        median_vals = []
        for e in range(max_len):
            vals = [d[thr][e] for d in per_run_data if len(d[thr]) > e]
            median_vals.append(np.median(vals))
        ax.plot(median_vals, epochs_full, color=cfg["colors"][i], lw=2.5,
                label=f"{case_label} {thr} (->{median_vals[-1]:.1f}, {n_runs} runs)")

ax.set_xlabel("threshold value")
ax.set_ylabel("epoch (training progresses downward)")
ax.invert_yaxis()
ax.set_title("corr noise + contained + 5000ep: median threshold convergence\n"
             "1ns/6ns (blues) vs 2ns/5ns (oranges)",
             fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_noise_corr_contained_compare/median_thresholds_compare_1ns6ns_2ns5ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")
