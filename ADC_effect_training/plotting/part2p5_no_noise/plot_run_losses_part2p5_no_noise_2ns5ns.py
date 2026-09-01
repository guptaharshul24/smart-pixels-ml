"""
Train/val loss curves for Part 2.5 no-noise, cold-start (2ns5ns, QConv2D)
runs -- see plotting/part2p5/plot_run_losses_part2p5_2ns5ns.py for the full
explanation (same script, points at part2p5_qconv2d_no_noise/ instead of
part2p5_qconv2d/). All runs here use MDMM (no separate no-MDMM exclusion
needed).

Uses an INCLUDED (allowlist) filter rather than an exclusion list: every
Part 2.5 no-noise attempt ever run -- pre- and post-fix alike -- writes into
the same on-disk output tree (part2p5_qconv2d_no_noise/), and the pre-fix
ones are all stuck runs from before the QKeras/Keras-3 gradient bug was
found (see losses/loss.py and models/models.py). They sit at a completely
different scale (frozen ~99k-103k at the clip floor, or collapsed at a
~15600-16300 plateau) from the real result, so plotting them together forces
the y-axis wide and hides all the relevant detail.

Only fp e61b24cc is plotted: the first and so far only genuine QConv2D
convergence (best_val_loss=-20864.39 at epoch 1193, EarlyStopping at 1293,
attempt 1/10 with no retries), obtained once TF_USE_LEGACY_KERAS=1 was set.
Nothing is lost -- every other run's training_log.csv is still on disk, and
the superseded eval outputs are archived in wrong_qconv_fixes/ (untracked).
"""
import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset_base_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
part2p5_output_dir = os.path.join(
    dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm", "part2p5_qconv2d_no_noise")
out = os.path.dirname(os.path.abspath(__file__))

# Allowlist -- see module docstring. Every other fingerprint in this output
# tree is a pre-Keras-fix stuck/collapsed attempt.
INCLUDED_FINGERPRINTS = {"e61b24cc"}

log_paths = glob.glob(os.path.join(part2p5_output_dir, "**", "QConv2D_model-*-checkpoints",
                                    "training_log.csv"), recursive=True)
if not log_paths:
    raise SystemExit(f"No Part 2.5 no-noise training_log.csv found under {part2p5_output_dir}")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plotted_any = False
for i, log_path in enumerate(sorted(log_paths, key=os.path.getctime)):
    fp = os.path.basename(os.path.dirname(log_path)).split("-")[1]
    if fp not in INCLUDED_FINGERPRINTS:
        continue
    rows = list(csv.DictReader(open(log_path)))
    if not rows or "pen_corr_x" not in rows[0]:
        continue  # skip non-MDMM (legacy/superseded) runs
    plotted_any = True
    epochs = [int(r["epoch"]) for r in rows]
    # MDMM logs both: "loss" = NLL + constraint penalties, "loss_obj" = pure NLL.
    losses = [float(r["loss_obj"]) for r in rows]
    val_losses = [float(r["val_loss"]) for r in rows]
    color = colors[i % len(colors)]
    tag = f"{fp} ({len(epochs)} epochs)"
    axes[0].plot(epochs, losses, label=tag, color=color)
    axes[1].plot(epochs, val_losses, label=tag, color=color)

if not plotted_any:
    raise SystemExit("No MDMM runs found under Part 2.5 no-noise output dir yet.")

axes[0].set_title("Training loss_obj (NLL)")
axes[1].set_title("Validation loss (NLL)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Part 2.5 no-noise, cold-start, QConv2D, frozen hard-digitized thresholds, 2ns/5ns")
plt.tight_layout()
out_path = os.path.join(out, "run_losses_part2p5_no_noise_2ns5ns.png")
plt.savefig(out_path, dpi=120)
print(f"saved to {out_path}")
