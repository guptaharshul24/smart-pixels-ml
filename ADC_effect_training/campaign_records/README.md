# campaign_records

Synced copies of each threshold-search campaign's `median_thresholds_*.json` and
`threshold_runs_*.jsonl`, so seed/threshold provenance lives in the repo rather than only in
`trained_models_*` on scratch. Written by `../sync_campaign_records.py <trained_models_dir>
<dest_subdir>`.

## Use MDMM records only

**The live threshold set is `mdmm_2ns5ns/corr1e4/`:**

```
median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json
    -> 13.001228 / 21.901985 / 57.13501 mV
```

Every Stage 1.5 / 2 / 2.5 script reads that file, and their output dirs are tagged
`fixed_thr_13.00_21.90_57.14`. Without MDMM the angle predictions collapse — the non-MDMM Stage 1
control (`raw_pixelAV_training/plotting/47f2d3a6_failed-no_mdmm`) has x-residual std 29.77 against
6.25 for its MDMM twin, worst on the angles. A non-MDMM number is not a weaker result, it is a
broken one.

**Do not resolve thresholds by grepping this directory.** Read the path out of the training script
(`train_*_mdmm_corr1e4.py` names the exact file). The non-MDMM file differs from the MDMM one only
by a missing `_mdmm` suffix and describes itself identically — "corr-noise, contained, 2ns/5ns" —
so the two are easy to confuse. That is exactly how the delay study initially got built on the wrong
set (2026-09-23).

## Contents

| Directory | Case | Status |
|---|---|---|
| `mdmm_2ns5ns/corr1e4/` | 2ns/5ns, corr-noise, contained, **MDMM** (Pearson correlation constraint) | **Live.** The only campaign with adopted medians. |
| `mdmm_2ns5ns/archive_mad1e4/`, `archive_scale1/`, `archive_std1e4/` | same case, other MDMM constraint formulations | Superseded. `threshold_runs` only, no adopted medians. |
| `mdmm_2ns5ns/part1p5_vit/`, `part2p5_qconv2d/` | downstream run summaries | Provenance for the frozen-threshold stages. |
| `noise_corr_contained_2ns5ns_BAD-ANGLES-DO-NOT-USE/` | 2ns/5ns, corr-noise, contained, **no MDMM** | **Do not use.** Renamed 2026-09-23. Retained only so the earlier runs stay reproducible. |
| `noise_corr_1ns6ns/`, `noise_corr_contained_1ns6ns/`, `noiseless_1ns6ns/` | 1ns/6ns cases, pre-MDMM | Historical. The 1ns/6ns case is superseded by 2ns/5ns; these predate MDMM. |
