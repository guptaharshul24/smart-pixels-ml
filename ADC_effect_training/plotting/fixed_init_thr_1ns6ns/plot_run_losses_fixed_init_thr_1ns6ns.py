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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, fp, color in RUNS:
    path = os.path.join(base, f"Transformer_model-{fp}-checkpoints", "training_log.csv")
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
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/fixed_init_thr_1ns6ns/run_losses_fixed_init_thr_1ns6ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")

# Zoomed view
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, fp, color in RUNS:
    path = os.path.join(base, f"Transformer_model-{fp}-checkpoints", "training_log.csv")
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
out2 = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/fixed_init_thr_1ns6ns/run_losses_zoom_fixed_init_thr_1ns6ns.png"
plt.savefig(out2, dpi=120)
print(f"saved to {out2}")
