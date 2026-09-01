"""
Train/val loss curves for Part 2 no-noise (2ns5ns) runs -- see
plotting/part2/plot_run_losses_part2_2ns5ns.py for the full explanation
(same script, points at part2_conv2d_no_noise/ instead of part2_conv2d/).

Attempt 1 (fp 692b1b40) hit the NLL-clip stuck-at-init trap (see
losses/loss.py's custom_loss docstring) and sat frozen at loss_obj ~103620
for all 536 of its epochs -- not real training. Excluded here (same
convention as Part 1.5's pre-MDMM-run exclusion) since its flat ~100000
plateau forces the y-axis wide and squashes the actual convergence detail
of the successful attempt 2 (64d9b19b) into an unreadable sliver. Data isn't
lost -- training_log.csv for 692b1b40 is still on disk -- just not plotted.
"""
import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset_base_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
part2_output_dir = os.path.join(
    dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm", "part2_conv2d_no_noise")
out = os.path.dirname(os.path.abspath(__file__))

# Stuck-at-init attempts (see module docstring) -- excluded from the plot.
EXCLUDED_FINGERPRINTS = {"692b1b40"}

log_paths = glob.glob(os.path.join(part2_output_dir, "**", "Conv2D_model-*-checkpoints",
                                    "training_log.csv"), recursive=True)
if not log_paths:
    raise SystemExit(f"No Part 2 no-noise training_log.csv found under {part2_output_dir}")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plotted_any = False
for i, log_path in enumerate(sorted(log_paths, key=os.path.getctime)):
    fp = os.path.basename(os.path.dirname(log_path)).split("-")[1]
    if fp in EXCLUDED_FINGERPRINTS:
        continue
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
    raise SystemExit("No MDMM runs found under Part 2 no-noise output dir yet.")

axes[0].set_title("Training loss_obj (NLL)")
axes[1].set_title("Validation loss (NLL)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Part 2 no-noise, non-quantized Conv2D, frozen hard-digitized thresholds, 2ns/5ns")
plt.tight_layout()
out_path = os.path.join(out, "run_losses_part2_no_noise_2ns5ns.png")
plt.savefig(out_path, dpi=120)
print(f"saved to {out_path}")
