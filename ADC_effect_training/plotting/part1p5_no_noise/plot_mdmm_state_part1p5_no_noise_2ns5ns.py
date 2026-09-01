"""
MDMM constraint-penalty curves for Part 1.5 no-noise (2ns5ns) runs -- see
plotting/part1p5/plot_mdmm_state_part1p5_2ns5ns.py for the full explanation
(same script, points at part1p5_no_noise/ instead of part1p5_vit/).
"""
import os
import glob
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset_base_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
part1p5_output_dir = os.path.join(
    dataset_base_dir, "trained_models_2_5_noise_corr_contained_mdmm", "part1p5_no_noise")
out = os.path.dirname(os.path.abspath(__file__))

log_paths = glob.glob(os.path.join(part1p5_output_dir, "**", "Transformer_model-*-checkpoints",
                                    "training_log.csv"), recursive=True)
params = ["x", "y", "cotA", "cotB"]
colors = {"x": "tab:blue", "y": "tab:orange", "cotA": "tab:green", "cotB": "tab:red"}

any_mdmm = False
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for log_path in sorted(log_paths, key=os.path.getctime):
    fp = os.path.basename(os.path.dirname(log_path)).split("-")[1]
    rows = list(csv.DictReader(open(log_path)))
    if not rows or f"pen_corr_x" not in rows[0]:
        continue
    any_mdmm = True
    epochs = [int(r["epoch"]) for r in rows]
    for ax, p in zip(axes.flat, params):
        pen = [float(r[f"pen_corr_{p}"]) for r in rows]
        ax.plot(epochs, pen, label=fp, alpha=0.8)

if not any_mdmm:
    raise SystemExit("No MDMM runs found under Part 1.5 no-noise output dir yet (no pen_corr_* "
                      "columns in any training_log.csv).")

for ax, p in zip(axes.flat, params):
    ax.set_title(f"corr_{p} penalty")
    ax.set_xlabel("epoch")
    ax.set_ylabel("penalty (-> 0 once constraint satisfied)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Part 1.5 no-noise (ViT) 2ns/5ns: MDMM correlation-constraint penalties per parameter")
plt.tight_layout()
out_path = os.path.join(out, "mdmm_state_part1p5_no_noise_2ns5ns.png")
plt.savefig(out_path, dpi=120)
print(f"saved to {out_path}")
