"""
Train/val loss curves for Part 2.5 (2ns5ns, QConv2D) runs -- same two-panel
layout as Part 1/Part 1.5's loss plots. Thresholds are frozen constants in
Part 2.5 (nothing analogous to Part 1's threshold-evolution plot) -- loss is
the only per-epoch trajectory worth tracking here. "loss_obj" (pure NLL,
present on MDMM runs) is plotted instead of "loss" (NLL + constraint
penalties) when present, so curves stay comparable across MDMM and
non-MDMM runs. Discovers runs via checkpoint dirs directly (not
summary.json, only written on completion) so in-progress runs show up.

Only plots MDMM runs -- the pre-MDMM stalled attempt (fp 61e72f40, killed
before MDMM was added) is excluded rather than shown alongside real runs.
Every Part 2.5 run from here on uses MDMM, so this filter naturally stays
correct without needing to track excluded fingerprints.
"""
import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset_base_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
part2p5_output_dir = os.path.join(
    dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm", "part2p5_qconv2d")
out = os.path.dirname(os.path.abspath(__file__))

log_paths = glob.glob(os.path.join(part2p5_output_dir, "**", "QConv2D_model-*-checkpoints",
                                    "training_log.csv"), recursive=True)
if not log_paths:
    raise SystemExit(f"No Part 2.5 training_log.csv found under {part2p5_output_dir}")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plotted_any = False
for i, log_path in enumerate(sorted(log_paths, key=os.path.getctime)):
    fp = os.path.basename(os.path.dirname(log_path)).split("-")[1]
    rows = list(csv.DictReader(open(log_path)))
    if not rows or "pen_corr_x" not in rows[0]:
        continue  # skip non-MDMM (legacy/superseded) runs
    plotted_any = True
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["loss_obj"]) for r in rows]
    val_losses = [float(r["val_loss"]) for r in rows]
    color = colors[i % len(colors)]
    tag = f"{fp} ({len(epochs)} epochs)"
    axes[0].plot(epochs, losses, label=tag, color=color)
    axes[1].plot(epochs, val_losses, label=tag, color=color)

if not plotted_any:
    raise SystemExit("No MDMM runs found under Part 2.5 output dir yet.")

axes[0].set_title("Training loss_obj (NLL)")
axes[1].set_title("Validation loss (NLL)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Part 2.5, QConv2D, frozen hard-digitized thresholds, 2ns/5ns")
plt.tight_layout()
out_path = os.path.join(out, "run_losses_part2p5_2ns5ns.png")
plt.savefig(out_path, dpi=120)
print(f"saved to {out_path}")
