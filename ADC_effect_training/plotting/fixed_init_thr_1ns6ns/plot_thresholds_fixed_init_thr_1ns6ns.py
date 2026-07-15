import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6/2t_rnd_thr_NoLog_Stdr_4p0_ThOf0.0_ThL35.0_ThH150.0"

# (label, fingerprint, color) -- shared across all plots; stuck/aborted run (6fb4eb30) excluded
RUNS = [
    ("Run 1", "473ab943", "tab:blue"),
    ("Run 2", "a2951979", "tab:orange"),
    ("Run 3", "ad1e76bd", "tab:red"),
    ("Run 4", "a21ce75c", "tab:purple"),
]
linestyles = {"threshold_0": "-", "threshold_1": "--", "threshold_2": ":"}

fig, ax = plt.subplots(figsize=(12, 7))

for label, fp, color in RUNS:
    path = os.path.join(base, f"Transformer_model-{fp}-checkpoints", "soft_quantizer_state_log.csv")
    data = {"epoch": [], "threshold_0": [], "threshold_1": [], "threshold_2": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["threshold_0"].append(float(row["threshold_0"]))
            data["threshold_1"].append(float(row["threshold_1"]))
            data["threshold_2"].append(float(row["threshold_2"]))
    for thr in ["threshold_0", "threshold_1", "threshold_2"]:
        final_val = data[thr][-1]
        ax.plot(data[thr], data["epoch"], label=f"{label} ({fp}) {thr} ({final_val:.2f} mV)",
                color=color, linestyle=linestyles[thr])

ax.set_xlabel("threshold value")
ax.set_ylabel("epoch")
ax.invert_yaxis()
ax.set_title("Per-epoch threshold evolution for all runs")
ax.legend(ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/fixed_init_thr_1ns6ns/run_thresholds_fixed_init_thr_1ns6ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")
