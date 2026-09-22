# raw_pixelAV_training

Runs the same four-stage pipeline as `ADC_effect_training/`, but on **pixelAV's raw charge data**
instead of our own frontend-effects dataset. The point is a controlled comparison: if the two
pipelines are given the same selection, the same angular range and the same number of events, then a
difference in residual widths is attributable to the front end itself rather than to the data.

Results from here feed `ADC_effect_training/plotting/residual_comparison/` as the `pixelav_matched`
condition.

## The dataset

`filter_subsample_pixelav.py` builds the subsample. Source is
`/work/projects/SmartPixML/datasets_16x16x20_charge/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/{train,test}/`
— 1,441,454 + 360,799 = 1,802,253 raw-charge events, in **electrons**, not noise-injected (noise and
normalization happen later at TFRecord-generation time, as everywhere else in this repo).

pixelAV's own train/test split is discarded. All eligible rows from both dirs are pooled, one global
random draw of 195,000 is taken, shuffled once, then sliced: first 155,000 → train, remaining
40,000 → test. Selection is `original_atEdge == False` (containment) **and** `|cotBeta| < 2.0`,
applied at selection time rather than deferred to TFR-gen so the row counts land exactly.

**Known discrepancy.** Those targets were meant to match the ADC-effects dataset's counts but were
derived as `file_count × nominal batch_size` (31 × 5,000 and 8 × 5,000) from
`TFR_files_2_5_noise_corr_contained`. The true counts there are **152,037 + 37,919 = 189,956**,
because `_build_batching_plan`'s `tail_tol=0.75` splits the remainder instead of emitting a runt
batch, so the last two files per split hold ~3,500 rows. This subsample is therefore 5,044 events
(2.6 %) larger than the set it matches. Left as-is — immaterial for residual widths, and the 31/8
file structure does match. See the docstring in `filter_subsample_pixelav.py` for the full
arithmetic. **If you ever need these matched properly, sum `actual_batch_size` over
`batch_metadata` in `metadata.json`; never multiply file count by `batch_size`.**

Two known selection side-effects, both documented in that docstring and deliberately not corrected:
containment (not the cotBeta cut) narrows cotAlpha's realized range and shifts y-midplane positive;
and pT is not flat, with a dip at pT≈0 that containment flips into a spike.

## Stages

Same structure as `ADC_effect_training/`, so the two are directly comparable:

| Stage | Script | What it is |
|---|---|---|
| 1 | `train_vit_rnd_thr_pixelav_matched.py` | Threshold search, soft/differentiable ADC (`SoftQuantizeLayer`) |
| 1.5 | `train_vit_part1p5_pixelav_matched_mdmm.py` | Same ViT, thresholds **frozen and hard-digitized** — isolates the cost of freezing |
| 2 | `train_conv2d_part2_pixelav_matched_mdmm.py` | Chip-sized plain Conv2D, every QKeras layer swapped for its Keras equivalent — isolates architecture shrink from quantization |
| 2.5 | `train_qconv2d_part2p5_pixelav_matched_mdmm.py` | QKeras-quantized Conv2D (`QConv2D_Max`) — adds quantization on top |

`train_loop_pixelav_matched_mdmm.py` and `run_orchestrator_pixelav_matched_mdmm.py` drive the Stage 1
MDMM campaign (multiple runs, median thresholds). `wrapper_submit_part2_part2p5.py` chains Stage 2 →
Stage 2.5, launching 2.5 only if 2 exits cleanly — unlike
`ADC_effect_training/wrapper3_submit_part2p5.py`, which polls.

## TFRecords

| File | What it does |
|---|---|
| `generate_tfr_pixelav_matched.py` | Noisy TFRs from the subsampled parquets |
| `generate_tfr_pixelav_matched_no_noise.py` | No-noise TFRs from the **same** parquet source, so the underlying 195,000 events are identical and only the noise setting differs |

The no-noise set is deliberate (2026-09-08): thresholds are derived from one condition and reused in
the other, so the two must share events exactly.

## Gotchas

- **QConv2D needs Keras 2.** `TF_USE_LEGACY_KERAS=1` must be set *before* any TF/Keras/QKeras import
  — QKeras 0.9.0 does not work on Keras 3. Applies to Stage 2.5 training and its eval script.
- **`LABELS_SCALE` is this dataset's own**, distinct from the frontend dataset's. Any cross-dataset
  residual comparison has to un-scale with the right one.

## `plotting/`

| File | What it does |
|---|---|
| `eval_pixelav_matched_rnd_thr.py`, `eval_pixelav_matched_mdmm.py` | Stage 1 eval (non-MDMM / MDMM). MDMM version picks the best of many `threshold_runs.jsonl` entries |
| `eval_pixelav_matched_part1p5_mdmm.py` | Stage 1.5 eval. No `SoftQuantizeLayer` — digitization happens in the data generator via `digitize=True`; reads a single `summary.json` |
| `eval_pixelav_matched_part2_mdmm.py` | Stage 2 eval, loads `CreateNonQuantizedModel` from the Stage 2 training script |
| `eval_pixelav_matched_part2p5_mdmm.py` | Stage 2.5 eval, QConv2D — needs the legacy-Keras env var |
| `plot_run_losses_*.py`, `plot_mdmm_state_*.py` | Read `training_log.csv` directly, so they work mid-training |
| `plot_thresholds_*.py`, `plot_pred_angle_dists_*.py` | Threshold evolution; predicted-angle distributions from an eval's `predictions.csv` |
| `plot_distributions_pixelav_matched.py` | Input/label distributions for the subsampled dataset |

Each eval writes to `plotting/<fingerprint>/` — `predictions.csv`, `residual_hists.png`, `pull.png`,
`sigma_hists.png`, `pred_angle_dists.png`, `residual_summary.txt`. Current fingerprints:

| Fingerprint | Run | x-residual std |
|---|---|---|
| `1e65a8aa` | Stage 1, ViT rnd_thr, MDMM | 6.25 |
| `399ab9d5` | Stage 1.5, ViT frozen thresholds, no-noise, MDMM | 6.37 |
| `eded8400` | Stage 2, plain Conv2D, frozen thresholds, no-noise, MDMM | 11.92 |
| `17f79cba` | Stage 2.5, QConv2D, frozen thresholds, no-noise, MDMM | 14.93 |
| `47f2d3a6_failed-no_mdmm` | Stage 1, ViT rnd_thr, **no MDMM** — kept deliberately | 29.77 |

(x-residual std in the label's own units, scale 122.8935; full numbers in each
`residual_summary.txt`.)

The progression is the point of the whole exercise: freezing thresholds costs almost nothing
(6.25 → 6.37), shrinking to a chip-sized Conv2D roughly doubles the residual (→ 11.92), and
quantizing adds a further ~25 % (→ 14.93). `47f2d3a6` is retained as the negative control — the same
Stage 1 configuration without MDMM collapses to 29.77, which is why MDMM angle-collapse protection
is on everywhere else.
