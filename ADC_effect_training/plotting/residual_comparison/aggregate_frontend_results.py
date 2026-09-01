"""
Aggregates our own (post-CSA "frontend effects" dataset) per-run predictions.csv
files into the exact same JSON structure/key-naming as the prior team's real
dataset3sr/residuals.json (fetched via scp -- see
convert_dataset3sr_residuals.py's docstring), so both sides can be plotted
through the same code path in plot_residual_comparison.py with no format
translation needed.

Per quantity v in {x, y, cotA, cotB, A, B} (A/B = alpha/beta in degrees,
cotA/cotB = the same angles in their raw scaled-cotangent form): mean_<v>,
std_<v>, up68_<v>, down68_<v> (measured residual: mean, plain std, and the
asymmetric shortest-68%-interval half-widths -- the same robust metric
upstream's read-all-models-2s.ipynb uses). Plus mean_upsigma<v>/
mean_downsigma<v> (no underscore before the letter, matching their exact
key spelling) -- the model's own mean PREDICTED uncertainty, a second,
independent stat from the measured spread.

Architecture keys match theirs directly: "transformer" (our ViT) and
"max_2dconv" (our Max Conv2D) -- not "vit"/"max_conv2d". Variant keys use
their numbered convention ("3-input_dig_2t", etc.) rather than our own
descriptive names.

Only includes (architecture, variant) combos we actually have a valid
trained result for -- no fabricated/placeholder entries. As of writing:
  - transformer / 3-input_dig_2t : Stage 1.5 (frozen thresholds, MDMM), fp 00bbfea6
  - max_2dconv  / 3-input_dig_2t : Stage 2 (non-quantized Conv2D, MDMM), fp 986827aa
  - (no_noise condition) transformer / 3-input_dig_2t : Stage 1.5 no-noise, fp fc8976dc
  - (no_noise condition) max_2dconv  / 3-input_dig_2t : Stage 2 no-noise, fp 64d9b19b
    (attempt 2/10 -- attempt 1, fp 692b1b40, hit the NLL-clip stuck-at-init
    trap in losses/loss.py's custom_loss and never trained; not used here)
  - (no_noise condition) max_2dconv  / 4-quantized : Stage 2.5 (QConv2D) no-noise,
    fp e61b24cc -- attempt 1/10, best_val_loss=-20864.39 (epoch 1193), EarlyStopping
    at epoch 1293. First valid QConv2D result, after root-causing the prior 0/20+
    failure rate to a QKeras/Keras-3 incompatibility that silently dropped the
    gradient for one of each layer's kernel/bias quantizer (fixed via
    TF_USE_LEGACY_KERAS=1 -- see models/models.py and models/mdmm.py).
Missing on purpose (not written to the JSON at all):
  - transformer / 4-quantized : no QViT stage exists
  - max_2dconv / 1-noquant_20t, 2-noquant_2t : neither architecture has a
    genuine full-precision variant per direct collaborator confirmation --
    Stage 1's soft/trainable ADC (fp a43ed7b9) is still considered a
    digitized-input case, not full-precision, so it isn't relabeled into
    either slot; Max Conv2D was never trained on non-digitized input at all.
  - (frontend condition) max_2dconv / 4-quantized : Stage 2.5 (QConv2D) has no
    valid noisy-frontend result yet -- only the no_noise run has succeeded so far.

Writes one JSON per dataset condition (residuals_<condition>.json), not just
one -- "frontend" is our standard result (correlated-noise TFRs, same as
everything else in the pipeline), "no_noise" is the no-noise rerun (same
frozen thresholds, contained clusters, 2ns/5ns -- only noise is toggled off)
for both transformer (fp fc8976dc) and max_2dconv (fp 64d9b19b).
"""
import os
import json
import numpy as np
import pandas as pd

pi = np.pi
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

# 2ns5ns LABELS_SCALE, confirmed identical across Stage 1/1.5/2 val sets
# (see comparison_stage2_stage3/compare_stage2_stage3.py)
LABELS_SCALE = {"x": 123.6133301, "y": 31.00504472, "cotA": 6.51238097, "cotB": 1.84647953}

CONDITIONS = {
    "frontend": {
        "transformer": {
            "3-input_dig_2t": "ADC_effect_training/plotting/part1p5/00bbfea6/predictions.csv",
        },
        "max_2dconv": {
            "3-input_dig_2t": "ADC_effect_training/plotting/part2/986827aa/predictions.csv",
        },
    },
    "no_noise": {
        "transformer": {
            "3-input_dig_2t": "ADC_effect_training/plotting/part1p5_no_noise/fc8976dc/predictions.csv",
        },
        "max_2dconv": {
            "3-input_dig_2t": "ADC_effect_training/plotting/part2_no_noise/64d9b19b/predictions.csv",
            "4-quantized": "ADC_effect_training/plotting/part2p5_no_noise/e61b24cc/predictions.csv",
        },
    },
}


def inverse_cot(cota):
    a = np.arctan(1.0 / cota)
    a = np.where(a < 0, a + pi, a)
    return a


def shortest_interval_68(data, center_type="mean"):
    """Shortest window containing 68% of the data -- a robust, non-Gaussian-safe
    stand-in for +/-1 sigma. Ported from upstream's read-all-models-2s.ipynb."""
    data = np.sort(data)
    n = len(data)
    ci_size = int(np.floor(0.68 * n))
    min_width = float("inf")
    min_i = 0
    for i in range(n - ci_size):
        width = data[i + ci_size] - data[i]
        if width < min_width:
            min_width = width
            min_i = i
    low = data[min_i]
    high = data[min_i + ci_size]
    center = np.mean(data) if center_type == "mean" else np.median(data)
    return {"error_low": float(center - low), "error_high": float(high - center)}


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    stats = {}

    # cotA, cotB: residuals in their raw scaled-cotangent form (LABELS_SCALE
    # applied, no degree conversion)
    for cot_v in ["cotA", "cotB"]:
        resid = (df[cot_v] - df[cot_v + "true"]) * LABELS_SCALE[cot_v]
        interval = shortest_interval_68(resid.values)
        stats[f"mean_{cot_v}"] = float(np.mean(resid))
        stats[f"std_{cot_v}"] = float(np.std(resid))
        stats[f"up68_{cot_v}"] = interval["error_high"]
        stats[f"down68_{cot_v}"] = interval["error_low"]

    # A, B: alpha/beta in degrees (cotA/cotB run through inverse_cot)
    angle = {}
    for cot_v, deg_v in [("cotA", "A"), ("cotB", "B")]:
        ang = inverse_cot(df[cot_v].values * LABELS_SCALE[cot_v]) * 180 / pi
        angletrue = inverse_cot(df[cot_v + "true"].values * LABELS_SCALE[cot_v]) * 180 / pi
        angle[deg_v] = ang
        resid = ang - angletrue
        interval = shortest_interval_68(resid)
        stats[f"mean_{deg_v}"] = float(np.mean(resid))
        stats[f"std_{deg_v}"] = float(np.std(resid))
        stats[f"up68_{deg_v}"] = interval["error_high"]
        stats[f"down68_{deg_v}"] = interval["error_low"]

    # predicted-uncertainty bands: symmetric for x/y/cotA/cotB, asymmetric for
    # A/B (propagated through inverse_cot on the +/- side)
    for v in ["x", "y"]:
        mean_sigma = float(np.mean(df["sigma" + v])) * LABELS_SCALE[v]
        stats[f"mean_upsigma{v}"] = mean_sigma
        stats[f"mean_downsigma{v}"] = mean_sigma
    for cot_v in ["cotA", "cotB"]:
        mean_sigma = float(np.mean(df["sigma" + cot_v])) * LABELS_SCALE[cot_v]
        stats[f"mean_upsigma{cot_v}"] = mean_sigma
        stats[f"mean_downsigma{cot_v}"] = mean_sigma
    for cot_v, deg_v in [("cotA", "A"), ("cotB", "B")]:
        cot_pred = df[cot_v].values * LABELS_SCALE[cot_v]
        sigma_cot = df["sigma" + cot_v].values * LABELS_SCALE[cot_v]
        angle_up = inverse_cot(cot_pred + sigma_cot) * 180 / pi
        angle_down = inverse_cot(cot_pred - sigma_cot) * 180 / pi
        stats[f"mean_upsigma{deg_v}"] = float(np.mean(np.abs(angle_up - angle[deg_v])))
        stats[f"mean_downsigma{deg_v}"] = float(np.mean(np.abs(angle_down - angle[deg_v])))

    # x, y: linear residuals in um
    for v in ["x", "y"]:
        resid = (df[v] - df[v + "true"]) * LABELS_SCALE[v]
        interval = shortest_interval_68(resid.values)
        stats[f"mean_{v}"] = float(np.mean(resid))
        stats[f"std_{v}"] = float(np.std(resid))
        stats[f"up68_{v}"] = interval["error_high"]
        stats[f"down68_{v}"] = interval["error_low"]

    return stats


def main():
    for condition, runs in CONDITIONS.items():
        out = {}
        for arch, variants in runs.items():
            out[arch] = {}
            for variant, rel_path in variants.items():
                csv_path = os.path.join(repo_root, rel_path)
                if not os.path.exists(csv_path):
                    print(f"SKIP {condition}/{arch}/{variant}: {csv_path} not found")
                    continue
                out[arch][variant] = summarize(csv_path)
                print(f"{condition}/{arch}/{variant}: computed")

        out_path = os.path.join(here, f"residuals_{condition}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
