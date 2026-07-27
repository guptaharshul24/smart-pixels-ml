# ADC Effect Training

Studies the accuracy cost of moving from an idealized, fully-trainable pixel readout down to
something that could actually run on-chip: a fixed 2-bit ADC, a compact convolutional
architecture, and quantized weights. Each stage below removes exactly one idealization at a time,
so the accuracy drop at each step can be attributed to a single, specific cause rather than several
tangled together.

Two timing cases are studied throughout: **1ns/6ns** (early/late split) and **2ns/5ns** (tighter,
more centered). Most of the mature results below are for 2ns/5ns; 1ns/6ns has threshold-search
results but hasn't been carried through the later stages yet (see [Status](#status--known-gaps)).

## The four stages

| Stage | Script | Architecture | Threshold | What's isolated |
|---|---|---|---|---|
| **1** | `train_vit_part1_rnd_thr_*.py` (+ `mdmm/*/train_loop_rnd_thr_*_mdmm.py`) | ViT | soft, trainable (`SoftQuantizeLayer`, annealed) | learns the thresholds — nothing frozen yet |
| **1.5** | `train_vit_part1p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | ViT (same as Stage 1) | frozen, hard-digitized | cost of no longer optimizing thresholds during training |
| **2** | `train_conv2d_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | plain Conv2D (1,869 params) | frozen, hard-digitized | cost of shrinking to a chip-sized architecture |
| **2.5** | `train_qconv2d_part2p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | QKeras QConv2D (4-bit conv / 8-bit dense) | frozen, hard-digitized | cost of quantization-aware training — the actual chip-deployable model |

Stage 1 produces the frozen thresholds (`median_thresholds_*.json`, median over 5 independent runs)
that every later stage trains against. Stages 1.5/2/2.5 are otherwise identical in dataset, MDMM
config, and retry/floor-gate logic — only the model itself changes.

## Results (2ns/5ns)

| Stage | Run | best_val_loss | x corr | y corr | cotA corr | cotB corr |
|---|---|---|---|---|---|---|
| 1.5 (ViT) | `00bbfea6` | -35,906 | 0.991 | 0.987 | 0.994 | 0.975 |
| 2 (plain Conv2D) | `986827aa` | -25,749 | 0.973 | 0.973 | 0.987 | 0.960 |
| 2.5 (QConv2D) | — | **all 10/10 attempts failed** | — | — | — | — |

**Stage 2.5 has never produced a valid result.** Every attempt, across 10 different seeds, gets
stuck at essentially the identical val_loss (~98,980) and never clears the acceptance floor. The
leading hypothesis is that 4-bit weight quantization (`quantized_bits(4,0,1,alpha=1)`, which assumes
weights fill ~[-1,1]) crushes small Glorot-initialized weights toward zero at init, killing gradient
flow — supported by: (a) Stage 2's *identical* architecture, unquantized, converges cleanly on
attempt 1; (b) different seeds land on the exact same stuck value, suggesting many different
initializations collapse to the same degenerate point rather than genuinely different local minima.
Not definitively proven (no direct gradient/weight-histogram inspection has been done).
`plotting/comparison_stage2_stage3/` has the Stage 2 vs Stage 1.5 comparison; Stage 2.5 will be
added there once a real result exists.

## MDMM: preventing angle-prediction collapse

Every stage from 1.5 onward wraps its model in `models/mdmm.py`'s `MDMM` class. Without it, angle
predictions (cotA/cotB) collapse to a near-constant value — this minimizes the NLL loss without
actually tracking the true signal, and correlation with truth drops to near zero (observed directly:
a pre-MDMM Stage 1.5 attempt landed cotA/cotB corr ~0.03/0.05). MDMM adds a Lagrangian penalty
(`MinCorrConstraint`) enforcing a minimum correlation between predictions and truth; the penalty
grows while violated and shrinks to ~0 once satisfied, so it's self-tuning — no manual weight to
hand-pick. It's training-time only: the Lagrange multipliers are discarded after training, so the
deployed model's architecture and weights are completely unaffected (`MDMM.save_weights`/
`load_weights` delegate straight to the wrapped model).

## Directory guide

```
ADC_effect_training/
├── train_vit_part1_rnd_thr_*.py          Stage 1: threshold search (soft ADC, various dataset variants)
├── train_vit_part1p5_..._mdmm_corr1e4.py Stage 1.5: frozen-threshold ViT
├── train_conv2d_part2_..._mdmm_corr1e4.py     Stage 2: plain Conv2D
├── train_qconv2d_part2p5_..._mdmm_corr1e4.py  Stage 2.5: QConv2D
├── generate_tfr_*.py                     TFRecord generation for each dataset variant
├── wrapper2-3_*.py                       self-sequencing launch chain (see below)
├── mdmm/
│   ├── 1ns6ns/ , 2ns5ns/                 Stage 1 MDMM threshold-search campaigns + their eval
│   └── status_and_plot.py                one-stop status/plot regen across Part 1/1.5/2.5
├── plotting/
│   ├── rnd_thr_*/                        Stage 1 (non-MDMM) threshold-search eval + GIFs
│   ├── transformer_eval_*/               Stage 1 (non-MDMM) prediction eval
│   ├── part1p5/ , part2/ , part2p5/      Stage 1.5/2/2.5 eval + plotting (mirrored structure)
│   └── comparison_stage2_stage3/         cross-stage overlay plots
├── campaign_records/                     synced summary.json provenance per campaign
└── DG import: DG/OptimizedDataGenerator_v3.py (repo root) — see below
```

Each stage's `plotting/<stage>/` directory follows the same pattern: `eval_*.py` (residuals, pulls,
sigma hists, summary "money" plot — run once per completed fingerprint), `plot_pred_angle_dists_*.py`
(reads that eval's `predictions.csv`), `plot_run_losses_*.py` and `plot_mdmm_state_*.py` (read
`training_log.csv` directly, work mid-training). `status_and_plot.py --part1 --part1p5 --part2p5`
(flags combine freely) drives all of these across every case in one shot.

## File reference

Every script in the tree, grouped by directory. "Case" below means the dataset variant a script is
specialized for (`1ns6ns`, `2ns5ns`, with/without correlated noise, with/without contained-cluster
selection) — most filenames encode this directly.

### Top level

| File | What it does |
|---|---|
| `generate_tfr_noise_corr_1ns6ns.py` | Builds TFRecords for the correlated-noise (no containment) 1ns/6ns dataset. |
| `generate_tfr_noise_corr_contained_1ns6ns.py` | Same, plus contained-cluster selection, 1ns/6ns. |
| `generate_tfr_noise_corr_contained_2ns5ns.py` | Same, 2ns/5ns case — this is the TFRecord source every Stage 1.5/2/2.5 script reads from. |
| `run_orchestrator_2ns5ns.py` | Launches the plain (non-MDMM) Stage 1 threshold search for 2ns/5ns: calls `train_vit_part1_rnd_thr_noise_corr_contained_2ns5ns.py --seed X --run_index Y` in a subprocess per run (no TF import itself, so the subprocess gets the full GPU budget), 5 target runs, deterministic seed pool (`np.random.default_rng(20260627)` — see the fingerprint-collision note above). |
| `sync_campaign_records.py` | Copies the small provenance files (threshold-run journal + median-thresholds JSON) out of a campaign's large, external-filesystem `trained_models_dir` into `campaign_records/<dest>/`, so seed/threshold provenance is git-tracked even though checkpoints/TFRecords never are. |
| `train_vit_part1_1ns6ns.py` | Stage 1 ViT, fixed initial thresholds, 1ns/6ns, no noise/containment — the earliest/simplest threshold-search variant. |
| `train_vit_part1_rnd_thr_1ns6ns.py` | Stage 1 ViT, **random** initial thresholds per attempt, 1ns/6ns, no noise/containment. |
| `train_vit_part1_rnd_thr_noise_corr_1ns6ns.py` | Same, + correlated noise. |
| `train_vit_part1_rnd_thr_noise_corr_contained_1ns6ns.py` | Same, + contained-cluster selection. |
| `train_vit_part1_rnd_thr_noise_corr_contained_2ns5ns.py` | Same, 2ns/5ns case. |
| `train_vit_part1p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | **Stage 1.5**: ViT, thresholds frozen from campaign 4's median, hard-digitized, MDMM. |
| `train_conv2d_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | **Stage 2**: plain (unquantized) Conv2D twin of Stage 2.5, same MDMM/thresholds/retry design. Defines `CreatePlainModel` — every QKeras layer swapped for its plain Keras equivalent. |
| `train_qconv2d_part2p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py` | **Stage 2.5**: QKeras-quantized Conv2D (`models.models.CreateModel`), `run_eagerly=True` (required — `QSeparableConv2D`'s quantizer calls `.numpy()` internally, which breaks under graph tracing in this QKeras/TF version combination). Currently 0/10 successful attempts. |
| `wrapper2_submit_part1p5.py` | Checks that the DG import swap (`v2p5` → `v3`) has landed in the target training script — a no-op check now, since all scripts are permanently on v3 — then launches Stage 1.5 and syncs its summary into `campaign_records/`. (Originally waited on `wrapper1_swap_v3.py`, a one-time migration script that performed that swap; wrapper1 has since been deleted since its job is permanently done and there was nothing left for it to do.) |
| `wrapper3_submit_part2p5.py` | Waits for the Stage 1.5 process to appear then exit (not just "is it running," which would misfire on cold start), then launches Stage 2.5. |

### `mdmm/` — Stage 1 MDMM threshold-search campaigns

| File | What it does |
|---|---|
| `mdmm/status_and_plot.py` | One-stop status + plot regeneration across every case and part (`--part1 --part1p5 --part2p5`, `1ns6ns\|2ns5ns\|all`, `--no-eval\|--all-runs\|--incremental`). Read-only on training processes; safe to run anytime. |
| `mdmm/1ns6ns/run_orchestrator_1ns6ns_mdmm.py` | Orchestrator for the 1ns/6ns MDMM threshold search — **not yet launched**. Uses `secrets.randbits(31)` for seeds (not the shared deterministic pool). |
| `mdmm/1ns6ns/train_loop_rnd_thr_noise_corr_contained_1ns6ns_mdmm.py` | The actual Stage 1 + MDMM training script the 1ns6ns orchestrator calls per attempt. |
| `mdmm/1ns6ns/plotting/corr1e4/eval_transformer_1ns6ns_mdmm.py` | Residuals/pulls/summary eval for the 1ns6ns MDMM campaign. |
| `mdmm/1ns6ns/plotting/corr1e4/plot_mdmm_state_1ns6ns_mdmm.py` | Per-run Lagrange multipliers, predicted correlation with truth, predicted std vs true std — the MDMM-specific diagnostic. |
| `mdmm/1ns6ns/plotting/corr1e4/plot_pred_angle_dists_1ns6ns_mdmm.py` | Predicted vs true cotA/cotB distributions — checks whether dispersion is honest input-dependent spread or gamed/uncorrelated. |
| `mdmm/1ns6ns/plotting/corr1e4/plot_run_losses_1ns6ns_mdmm.py` | Train/val loss curves, all runs overlaid. |
| `mdmm/1ns6ns/plotting/corr1e4/plot_thresholds_1ns6ns_mdmm.py` | Threshold-vs-epoch convergence plot, all runs + cross-run median. |
| `mdmm/2ns5ns/run_orchestrator_2ns5ns_mdmm.py` | Same as the 1ns6ns orchestrator, for 2ns/5ns — this is **campaign 4**, the one whose median thresholds (`[13.00, 21.90, 57.14]` mV) feed every Stage 1.5/2/2.5 script. Already completed. |
| `mdmm/2ns5ns/train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.py` | The per-attempt training script campaign 4's orchestrator calls. |
| `mdmm/2ns5ns/plotting/corr1e4/*.py` | Same five-script pattern as the 1ns6ns set above, for 2ns/5ns/campaign 4 (`eval_transformer_2ns5ns_mdmm.py`, `plot_mdmm_state_2ns5ns_mdmm.py`, `plot_pred_angle_dists_2ns5ns_mdmm.py`, `plot_run_losses_2ns5ns_mdmm.py`, `plot_thresholds_2ns5ns_mdmm.py`), plus `make_threshold_gif_2ns5ns_mdmm.py` (animated version of the threshold plot, dynamic per-frame median legend). |
| `mdmm/2ns5ns/plotting/archive_std1e4/`, `archive_mad1e4/`, `archive_scale1/` | Frozen snapshots from earlier MDMM constraint-metric iterations, back when the constraint was std-based, then MAD-based, before settling on the current Pearson-correlation constraint (`MinCorrConstraint`). The `plot_mdmm_state_*` scripts here are **reconstructions** (noted in their own docstrings) — the constraint metric changed in place via `Write` rather than `Edit` at the time, so the exact original bytes weren't preserved; these are faithful re-derivations of what produced the archived PNGs. Kept for historical reference, not meant to be re-run against current data. |

### `plotting/` — Stage-specific eval and plotting

| Directory | Case | Notes |
|---|---|---|
| `rnd_thr_1ns6ns/` | Stage 1, fixed-init, 1ns6ns | `plot_thresholds_rnd_thr_1ns6ns.py`, `plot_run_losses_rnd_thr_1ns6ns.py`, `make_threshold_gif_rnd_thr_1ns6ns.py`. |
| `rnd_thr_noise_corr_1ns6ns/` | Stage 1, random-init + correlated noise, 1ns6ns | Same pattern, no GIF built yet. |
| `rnd_thr_noise_corr_contained_1ns6ns/` | Stage 1, + containment, 1ns6ns | + `make_threshold_gif_rnd_thr_noise_corr_contained_1ns6ns.py`. |
| `rnd_thr_noise_corr_contained_2ns5ns/` | Stage 1, + containment, 2ns5ns | + `make_threshold_gif_2ns5ns.py`. |
| `rnd_thr_noise_corr_contained_compare/` | cross-case | `plot_median_thresholds_compare_1ns6ns_2ns5ns.py` — overlays the two cases' median threshold trajectories on one plot. |
| `fixed_init_thr_1ns6ns/` | Stage 1, fixed-init variant | `plot_run_losses_fixed_init_thr_1ns6ns.py`, `plot_thresholds_fixed_init_thr_1ns6ns.py`. |
| `transformer_eval_1ns6ns/`, `transformer_eval_2ns5ns/` | Stage 1 prediction eval (non-MDMM) | `eval_transformer_*.py` (residuals/pulls/summary) + `plot_pred_angle_dists_*.py`. The 1ns6ns one is hardcoded to a known-collapsed run (`3b9c78f7`, pre-MDMM) as a worked example of what collapse looks like; the 2ns5ns one takes `--fingerprint` and is neutral (no collapse assumed — its best run, `a43ed7b9`, is genuinely healthy). |
| `part1p5/` | Stage 1.5 | `eval_part1p5_2ns5ns.py`, `plot_mdmm_state_part1p5_2ns5ns.py`, `plot_pred_angle_dists_part1p5_2ns5ns.py`, `plot_run_losses_part1p5_2ns5ns.py`. |
| `part2/` | Stage 2 | Same four-script pattern, `CreatePlainModel`-based, no `run_eagerly`. |
| `part2p5/` | Stage 2.5 | Same four-script pattern, `run_eagerly=True`. All eval output so far is from failed/stuck runs. |
| `comparison_stage2_stage3/` | cross-stage | `compare_stage2_stage3.py` — Stage 1.5 vs Stage 2 residuals+uncertainty and pull overlays, adapted from das's `performance_plots.ipynb` multi-model overlay pattern. |

## The launch chain (wrapper2-3)

Each wrapper polls real, independently-observable state rather than each other, so launch order
doesn't matter. (There used to be a `wrapper4_part2p5_then_fallback_part2.py` that launched Stage 2.5
and fell back to Stage 2 only if Stage 2.5 exhausted its retries — removed now that Stage 2 already
has a good result and won't be rerun as a fallback; going forward Stage 2 is run before Stage 2.5,
not after.)

## Data generator: v3 and `digitize`

All current scripts import `DG.OptimizedDataGenerator_v3`. v3 is a strict superset of v2p5 — it adds
opt-in `digitize`/`digitize_thresholds`/`digitize_levels` params (default off) implementing a real
`tf.raw_ops.Bucketize` hard step-function digitizer, applied in `__getitem__` after `quantize`. This
is what Stages 1.5/2/2.5 use to apply the frozen thresholds; Stage 1 instead digitizes inside the
model via the trainable `SoftQuantizeLayer`.

## Stopping/retry logic (Stages 1.5/2/2.5)

Each training attempt is governed by two independent, non-overlapping checks:
- **`AbortOnStuck`** (das's original design): aborts if `val_loss` stays above `1e5` for `patience`
  consecutive epochs, or goes non-finite — a pure divergence guard.
- **Floor gate**: after `EarlyStopping` (patience=100) ends an attempt, the run is only accepted if
  `best_val_loss` clears `GOOD_VAL_LOSS_THRESHOLD = -10000.0`; otherwise it's retried with a new
  seed, same as an `AbortOnStuck` abort. Calibrated against a real historical QConv2D reference
  (`HG_Convolution_train_model_conv2D.ipynb`, das's own, converges to -15,000 to -20,000) —
  deliberately lenient for now, to be tightened once Stage 2.5 has its own real reference.

This design replaced an earlier "stuck-at-init escape-window" check that lived inside `AbortOnStuck`
itself. That check only ever asked "did the loss move," never "is the loss any good" — a single
noisy dip counted as a permanent pass even if the run immediately reconverged to the same bad
plateau (this happened for real: QConv2D fp `4e4c3f5a` "escaped" at epoch 1, refroze by epoch 5, and
got accepted at epoch 52 with `best_val_loss=97154.6` — barely different from the runs correctly
flagged as stuck). The floor gate judges the actual achieved value instead of its trend, which no
trend-based check (this one, or `EarlyStopping` itself) can express.

## Status / known gaps

- **Stage 2.5 (QConv2D)**: no valid run yet — 10/10 attempts stuck. Next planned step: inspect
  gradients/weight histograms directly to confirm or rule out the QAT-crushes-init hypothesis, and
  potentially loosen the 4-bit quantizer's `alpha` or add a warm-up period before quantizing.
- **1ns/6ns**: has Stage 1 threshold-search results (both MDMM and non-MDMM) but Stages 1.5/2/2.5
  haven't been run for this case yet. The MDMM Stage-1 campaign orchestrator
  (`mdmm/1ns6ns/run_orchestrator_1ns6ns_mdmm.py`) is built but not launched.
- **Comparison plot**: only Stage 1.5 vs Stage 2 exists so far (`plotting/comparison_stage2_stage3/`).
  Add Stage 2.5 once it has a result.

### Fingerprint collision (cosmetic, not a bug)

`rnd_thr_noise_corr_contained_1ns6ns` and `rnd_thr_noise_corr_contained_2ns5ns` (the plain,
non-MDMM Stage-1 threshold-search campaigns) share all 5 run fingerprints
(`dee839bb, 094f9cc3, a43ed7b9, 583866ca, 3b9c78f7`) — both orchestrators hardcode the same seed
pool (`np.random.default_rng(20260627)`), so `run_index=N` always draws the same seed, and hence the
same fingerprint, in both. The underlying trained models are genuinely different (different
`time_stamps`, different final thresholds, different `best_val_loss`) — only the fingerprint
*labels* collide. A fingerprint alone is never enough to identify a run between these two specific
cases; the case/directory must always be specified too. MDMM orchestrators use `secrets.randbits(31)`
instead and are not affected.

## Truth labels

Every stage's regression target is `[x-midplane, y-midplane, cotAlpha, cotBeta]` — one quadruple per
event (per simulated particle track), not per pixel. `x-midplane`/`y-midplane` are the track
position extrapolated to the sensor's mid-depth (not the raw entry point):
```python
x-midplane = x-entry + cotAlpha * (thickness/2 - z-entry)
y-midplane = y-entry + cotBeta  * (thickness/2 - z-entry)
```
a more physically representative "true" hit position, since charge deposition is smeared across the
sensor thickness along the track. The model output is 14 numbers: the 4 means above, plus 10
lower-triangular Cholesky-factor elements of the 4×4 output covariance (`M11, M21, M22, M31, M32,
M33, M41, M42, M43, M44`), giving a well-posed covariance via `Σ = L·Lᵀ`.
