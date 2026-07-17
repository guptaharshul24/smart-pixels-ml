"""
MDMM constraint-penalty curves for Part 2.5 (2ns5ns, QConv2D) runs -- the
analog of Part 1's threshold-evolution plot, but tracking the correlation-
constraint penalty per parameter instead (thresholds are frozen constants
in Part 2.5, nothing to plot there). Reads the pen_corr_x/y/cotA/cotB columns
MDMM's train_step adds to training_log.csv. Runs without MDMM are skipped.
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
params = ["x", "y", "cotA", "cotB"]

any_mdmm = False
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for log_path in sorted(log_paths, key=os.path.getctime):
    fp = os.path.basename(os.path.dirname(log_path)).split("-")[1]
    rows = list(csv.DictReader(open(log_path)))
    if not rows or "pen_corr_x" not in rows[0]:
        continue
    any_mdmm = True
    epochs = [int(r["epoch"]) for r in rows]
    for ax, p in zip(axes.flat, params):
        pen = [float(r[f"pen_corr_{p}"]) for r in rows]
        ax.plot(epochs, pen, label=fp, alpha=0.8)

if not any_mdmm:
    raise SystemExit("No MDMM runs found under Part 2.5 output dir yet (no pen_corr_* "
                      "columns in any training_log.csv).")

for ax, p in zip(axes.flat, params):
    ax.set_title(f"corr_{p} penalty")
    ax.set_xlabel("epoch")
    ax.set_ylabel("penalty (-> 0 once constraint satisfied)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Part 2.5 (QConv2D) 2ns/5ns: MDMM correlation-constraint penalties per parameter")
plt.tight_layout()
out_path = os.path.join(out, "mdmm_state_part2p5_2ns5ns.png")
plt.savefig(out_path, dpi=120)
print(f"saved to {out_path}")
