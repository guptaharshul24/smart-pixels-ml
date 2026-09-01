import os
import csv
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_2_5_noise_corr_contained"
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

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
RUNS = [(f"Run {i+1}", os.path.basename(d).split("-")[1], d, colors[i % len(colors)])
        for i, d in enumerate(ckpt_dirs)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, fp, ckpt_dir, color in RUNS:
    path = os.path.join(ckpt_dir, "training_log.csv")
    if not os.path.exists(path):
        continue
    epochs, losses, val_losses = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            val_losses.append(float(row["val_loss"]))
    if not epochs:
        continue
    legend_label = f"{label} ({fp}, {len(epochs)} epochs)"
    axes[0].plot(epochs, losses, label=legend_label, color=color)
    axes[1].plot(epochs, val_losses, label=legend_label, color=color)

axes[0].set_title("Training loss")
axes[1].set_title("Validation loss")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Corr noise + contained, 5000ep target (2ns/5ns)")
plt.tight_layout()
out = "/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/plotting/rnd_thr_noise_corr_contained_2ns5ns/run_losses_rnd_thr_noise_corr_contained_2ns5ns.png"
plt.savefig(out, dpi=120)
print(f"saved to {out}")
