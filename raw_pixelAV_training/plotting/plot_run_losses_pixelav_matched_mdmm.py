"""Train/val loss curves for the pixelAV-matched Stage 1 (ViT, trainable
threshold) runs. Same pattern as
ADC_effect_training/plotting/rnd_thr_noise_corr_contained_2ns5ns/
plot_run_losses_rnd_thr_noise_corr_contained_2ns5ns.py, pointed at
dataset_3srb_16x16_50x12P5_centeredIncidence/trained_models_rnd_thr/.
"""
import os
import csv
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence/trained_models_rnd_thr_mdmm"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_pixelav_matched_mdmm.jsonl")
here = os.path.dirname(os.path.abspath(__file__))

# The MDMM journal is an event log (started / completed / failed / abandoned),
# unlike the non-MDMM one where every line was a finished run. Select dirs by
# the fingerprints of COMPLETED, non-stuck runs rather than globbing and
# excluding -- an abandoned seed's "started" record carries a fingerprint too.
good_fingerprints = set()
if os.path.exists(threshold_runs_path):
    for line in open(threshold_runs_path):
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "completed" and not r.get("stuck", False):
                good_fingerprints.add(r["fingerprint"])

ckpt_dirs = glob.glob(os.path.join(trained_models_dir, "1t_rnd_thr_pixelav_matched_mdmm_5000ep_*", "Transformer_model-*-checkpoints"))
ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] in good_fingerprints]
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

fig.suptitle("pixelAV-matched (containment + |cotBeta|<2, noise=[0,80]e-), Stage 1 ViT rnd_thr + MDMM, 5000ep target")
plt.tight_layout()
out = os.path.join(here, "run_losses_pixelav_matched_mdmm.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
