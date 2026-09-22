"""
Aggregates our pixelAV-matched-dataset predictions.csv files into the same
residuals_<condition>.json structure aggregate_frontend_results.py builds for
our own frontend-effects dataset, so plot_residual_comparison.py can plot
both through the same code path.

Separate script (not folded into aggregate_frontend_results.py) because the
label scale is dataset-specific -- read directly from each run's own
OptimizedDataGenerator (test_generator.labels_scale, printed by the eval
script and pasted in here), NOT aggregate_frontend_results.py's LABELS_SCALE
dict, which is specific to the 2ns5ns frontend-effects dataset's own
normalization and does not apply to the pixelAV-matched dataset.

Results, all pixelAV-matched dataset (dataset_3srb_16x16_50x12P5_centeredIncidence),
no-noise TFRs (TFR_files/2t/) with thresholds frozen from the noisy Stage 1
MDMM campaign's median:
  - transformer / 3-input_dig_2t : Stage 1.5 (ViT, frozen thresholds,
    MDMM), fp 399ab9d5 -- raw_pixelAV_training/plotting/eval_pixelav_matched_part1p5_mdmm.py
  - max_2dconv / 3-input_dig_2t : Stage 2 (non-quantized Conv2D, frozen
    thresholds, MDMM), fp eded8400, attempt 1/10, ran the full 5000 epochs --
    raw_pixelAV_training/plotting/eval_pixelav_matched_part2_mdmm.py
  - max_2dconv / 4-quantized : Stage 2.5 (QConv2D, frozen thresholds, MDMM),
    fp 17f79cba, attempt 1/10, best_val_loss=-23646.77 (epoch 796),
    EarlyStopping -- raw_pixelAV_training/plotting/eval_pixelav_matched_part2p5_mdmm.py
"""
import os
import json
import numpy as np
import pandas as pd

pi = np.pi
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

# pixelAV-matched dataset's own label scale (dataset_3srb_16x16_50x12P5_centeredIncidence,
# TFR_files/2t/ no-noise val set) -- read from test_generator.labels_scale at
# eval time (eval_pixelav_matched_part1p5_mdmm.py's printed
# "labels_scale = [...]"), NOT the 2ns5ns frontend dataset's LABELS_SCALE.
LABELS_SCALE = {"x": 122.89345547, "y": 30.94789415, "cotA": 6.50295834, "cotB": 1.8710731}

CONDITIONS = {
    "pixelav_matched": {
        "transformer": {
            "3-input_dig_2t": "raw_pixelAV_training/plotting/399ab9d5/predictions.csv",
        },
        "max_2dconv": {
            "3-input_dig_2t": "raw_pixelAV_training/plotting/eded8400/predictions.csv",
            "4-quantized": "raw_pixelAV_training/plotting/17f79cba/predictions.csv",
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

    for cot_v in ["cotA", "cotB"]:
        resid = (df[cot_v] - df[cot_v + "true"]) * LABELS_SCALE[cot_v]
        interval = shortest_interval_68(resid.values)
        stats[f"mean_{cot_v}"] = float(np.mean(resid))
        stats[f"std_{cot_v}"] = float(np.std(resid))
        stats[f"up68_{cot_v}"] = interval["error_high"]
        stats[f"down68_{cot_v}"] = interval["error_low"]

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
