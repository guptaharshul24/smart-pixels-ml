import os
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr.jsonl")

records = [json.loads(l) for l in open(threshold_runs_path) if l.strip()]
runs = [r for r in records if not r.get("stuck", False)]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
RUNS = [(f"Run {i+1}", r["fingerprint"], r["checkpoint_dir"], colors[i % len(colors)])
        for i, r in enumerate(runs)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, fp, ckpt_dir, color in RUNS:
    path = os.path.join(ckpt_dir, "training_log.csv")
    epochs, losses, val_losses = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            val_losses.append(float(row["val_loss"]))
    legend_label = f"{label} ({fp})"
    axes[0].plot(epochs, losses, label=legend_label, color=color)
    axes[1].plot(epochs, val_losses, label=legend_label, color=color)

axes[0].set_title("Training loss")
axes[1].set_title("Validation loss")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_1ns6ns/run_losses_rnd_thr_1ns6ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")

# Zoomed view
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, fp, ckpt_dir, color in RUNS:
    path = os.path.join(ckpt_dir, "training_log.csv")
    epochs, losses, val_losses = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            val_losses.append(float(row["val_loss"]))
    legend_label = f"{label} ({fp})"
    axes[0].plot(epochs, losses, label=legend_label, color=color)
    axes[1].plot(epochs, val_losses, label=legend_label, color=color)

axes[0].set_title("Training loss (zoomed)")
axes[1].set_title("Validation loss (zoomed)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_ylim(-50000, 5000)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_1ns6ns/run_losses_zoom_rnd_thr_1ns6ns.png"
plt.savefig(out2, dpi=120)
print(f"saved to {out2}")
