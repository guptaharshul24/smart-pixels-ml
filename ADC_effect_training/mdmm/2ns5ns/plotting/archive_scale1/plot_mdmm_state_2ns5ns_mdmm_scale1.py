"""
RECONSTRUCTED (2026-07-13): this is the std-constraint version of
plot_mdmm_state_2ns5ns_mdmm.py that produced mdmm_state_2ns5ns_mdmm_scale1.png (the scale=1.0 std-constraint campaign)
in this directory. It was overwritten in place (via Write, not Edit) when the
constraint metric moved std -> MAD -> correlation, so the exact original bytes
are not preserved -- this is a faithful reconstruction from the same authoring
pattern used throughout the campaign (confirmed consistent with the archived
scale1 script and the png this produced). Functionally identical script also
applies to the archive_std1e4/ campaign (same std-constraint log format,
different scale/config at runtime only -- that campaign used scale=1e4).

MDMM diagnostics: per-run Lagrange multipliers (lambda) and predicted output
std (the constrained metric in this campaign).
"""
import os
import csv
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_2_5_noise_corr_contained_mdmm_scale1"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm_std1e4.jsonl")

# must match MDMM_MIN_STD in the training script (0.8 * true std per parameter)
MIN_STD = {"x": 0.36, "y": 0.36, "cotA": 0.43, "cotB": 0.36}
PARAMS = ["x", "y", "cotA", "cotB"]
colors = {"x": "tab:blue", "y": "tab:orange", "cotA": "tab:green", "cotB": "tab:red"}

stuck_fingerprints = set()
if os.path.exists(threshold_runs_path):
    for line in open(threshold_runs_path):
        if line.strip():
            r = json.loads(line)
            if r.get("status", "completed") == "completed" and r.get("stuck", False):
                stuck_fingerprints.add(r["fingerprint"])

ckpt_dirs = glob.glob(os.path.join(trained_models_dir, "2t_rnd_thr_noise_corr_contained_2ns5ns_mdmm_5000ep_*", "Transformer_model-*-checkpoints"))
ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d).split("-")[1] not in stuck_fingerprints]
ckpt_dirs = sorted(ckpt_dirs, key=os.path.getctime)

per_run = []
for d in ckpt_dirs:
    path = os.path.join(d, "mdmm_state_log.csv")
    if not os.path.exists(path):
        continue
    data = {"epoch": [], **{f"lmbda_{p}": [] for p in PARAMS}, **{f"std_{p}": [] for p in PARAMS}}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            for p in PARAMS:
                data[f"lmbda_{p}"].append(float(row[f"lmbda_std_{p}"]))
                data[f"std_{p}"].append(float(row[f"pred_std_{p}"]))
    if data["epoch"]:
        per_run.append((os.path.basename(d).split("-")[1], data))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for fp, data in per_run:
    for p in PARAMS:
        axes[0].plot(data["epoch"], data[f"lmbda_{p}"], color=colors[p], alpha=0.4, lw=1)
        axes[1].plot(data["epoch"], data[f"std_{p}"], color=colors[p], alpha=0.4, lw=1)

for p in PARAMS:
    axes[0].plot([], [], color=colors[p], label=p)
    axes[1].axhline(MIN_STD[p], color=colors[p], ls="--", lw=1, alpha=0.8)
    axes[1].plot([], [], color=colors[p], label=f"{p} (target {MIN_STD[p]})")

axes[0].set_title("Constraint multipliers")
axes[0].set_ylabel(r"$\lambda$")
axes[1].set_title("Predicted output spreads (dashed = constraint targets)")
axes[1].set_ylabel("predicted std (deterministic, val batch)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(f"MDMM state (std constraints, scale=1.0 ARCHIVE): corr noise + contained + 2ns/5ns, {len(per_run)} run(s)")
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdmm_state_2ns5ns_mdmm_scale1.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
