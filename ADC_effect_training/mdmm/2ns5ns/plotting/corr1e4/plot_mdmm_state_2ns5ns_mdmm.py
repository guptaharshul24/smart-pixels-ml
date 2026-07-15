"""
MDMM diagnostics: per-run Lagrange multipliers (lambda), predicted correlation
with truth (the constrained metric) and predicted std (kept as a diagnostic:
a healthy correlated prediction has std close to the true label's std, so std
blowing up out of proportion flags an outlier-salting-style gaming attempt
again, same idea as the earlier std vs MAD check).

NOTE: this version reads the correlation-constraint log format (lmbda_corr_*,
pred_std_*, pred_corr_*). Earlier campaigns logged different formats -- use the
archived plots/scripts for those: archive_scale1/, archive_std1e4/ (std format:
lmbda_std_*), archive_mad1e4/ (MAD format: lmbda_mad_*, pred_mad_*).
"""
import os
import csv
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trained_models_dir = "/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/trained_models_1_6_noise_corr_contained_2ns5ns_mdmm"
threshold_runs_path = os.path.join(trained_models_dir, "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl")

# must match MDMM_MIN_CORR in the training script
MIN_CORR = {"x": 0.5, "y": 0.5, "cotA": 0.5, "cotB": 0.5}
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
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "lmbda_corr_x" not in reader.fieldnames:
            continue  # old-format log (std or MAD constraint campaign)
        data = {"epoch": [], **{f"lmbda_{p}": [] for p in PARAMS},
                **{f"std_{p}": [] for p in PARAMS}, **{f"corr_{p}": [] for p in PARAMS}}
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            for p in PARAMS:
                data[f"lmbda_{p}"].append(float(row[f"lmbda_corr_{p}"]))
                data[f"std_{p}"].append(float(row[f"pred_std_{p}"]))
                data[f"corr_{p}"].append(float(row[f"pred_corr_{p}"]))
    if data["epoch"]:
        per_run.append((os.path.basename(d).split("-")[1], data))

# distinguish runs by linestyle (color still encodes parameter); ctime order
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

for i, (fp, data) in enumerate(per_run):
    ls = LINESTYLES[i % len(LINESTYLES)]
    for p in PARAMS:
        axes[0].plot(data["epoch"], data[f"lmbda_{p}"], color=colors[p], alpha=0.6, lw=1.3, linestyle=ls)
        axes[1].plot(data["epoch"], data[f"corr_{p}"], color=colors[p], alpha=0.6, lw=1.3, linestyle=ls)
        axes[2].plot(data["epoch"], data[f"std_{p}"], color=colors[p], alpha=0.6, lw=1.3, linestyle=ls)

for p in PARAMS:
    axes[0].plot([], [], color=colors[p], label=p)
    axes[1].axhline(MIN_CORR[p], color=colors[p], ls="--", lw=1, alpha=0.8)
    axes[1].plot([], [], color=colors[p], label=f"{p} (target {MIN_CORR[p]})")
    axes[2].plot([], [], color=colors[p], label=p)

axes[0].set_title("Constraint multipliers")
axes[0].set_ylabel(r"$\lambda$")
axes[1].set_title("Predicted correlation with truth (dashed = targets)")
axes[1].set_ylabel("corr(pred, true) on val batch")
axes[1].set_ylim(-1.05, 1.05)
axes[2].set_title("Predicted std (diagnostic: outlier-salting flag)")
axes[2].set_ylabel("predicted std")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend(loc="lower left" if ax is axes[1] else "upper left")
    ax.grid(True, alpha=0.3)

# run-identity legend (linestyle -> fingerprint) as a SECOND legend on
# axes[2]: keep the param-color legend already placed there, add this one
# via add_artist so the next .legend() call below doesn't evict it
if per_run:
    from matplotlib.lines import Line2D
    param_legend = axes[2].get_legend()
    axes[2].add_artist(param_legend)
    run_handles = [Line2D([0], [0], color="gray", lw=1.5,
                          linestyle=LINESTYLES[i % len(LINESTYLES)],
                          label=f"run {i+1}: {fp}")
                   for i, (fp, _) in enumerate(per_run)]
    axes[2].legend(handles=run_handles, loc="lower right", fontsize=9, title="linestyle = run")

fig.suptitle(f"MDMM state (correlation constraints): corr noise + contained + 2ns/5ns, {len(per_run)} run(s)")
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdmm_state_2ns5ns_mdmm.png")
plt.savefig(out, dpi=120)
print(f"saved to {out}")
