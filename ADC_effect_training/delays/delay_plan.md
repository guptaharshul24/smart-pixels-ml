# Delay-aware ADC digitization — plan

Status: **design only, nothing implemented.** Open questions in the last section
must be settled before code is written. **Results below use the wrong threshold
set — see the warning under Inputs -> Thresholds.**

## Goal

The current hard digitization (`DG/OptimizedDataGenerator_v3.map_to_levels`)
assumes a comparator with **zero delay**: a pixel's mV value is bucketized against
the thresholds instantly. Real front ends have charge-dependent *time walk* — a
pixel sitting just above threshold fires late, and if it fires late enough it
misses the readout window and reads out one ADC level low.

This job injects that effect using a delay lookup table and produces a new
dataset variant to retrain on.

## Inputs

### 1. Delay LUT

`VIZARD_ADC_LUT_PixB_27c_combined_sorted.csv` — 102 rows x 240 columns, stored as
120 **column pairs**, one pair per threshold: `Charge_Qth_<X>` (the `Q_in` axis, in
e-) next to `Qth = <X>` (the delay, in seconds).

- `Q_th` grid: **25 to 3000 e- in steps of 25** (120 columns).
- `Q_in` grid is *per column* and ragged: 100 e- steps from that column's minimum up
  to 10000 e-, plus two far points at 50000 and 100000 e-. Columns are
  space-padded (`' '`) to 102 rows, so blanks must be dropped, not read as 0.
  Point count per column runs 102 (at `Q_th=25`) down to 46 (at `Q_th=3000`).
- Each column starts at `Q_in_min ~ 1.9-2.0 * Q_th` (floored at 100 e-). This is
  the plot's white region: below it the LUT is simply not tabulated.
- Delays span **1.40 to 26.65 ns** overall, but the 26 ns tail only exists at
  `Q_th = 75` / `Q_in = 100`. Delay falls monotonically with `Q_in` at fixed
  `Q_th`.
- Axes convert charge -> mV with the existing gain `CvG = 58e-6` V/e-
  (= 0.058 mV/e-), the same constant already used for the noise model in
  `generate_tfr_noise_corr_contained_2ns5ns.py:18`.

### 2. Thresholds

> **WRONG SET USED (found 2026-09-23).** Every result below was computed with
> **13.809455 / 24.07648 / 55.619907 mV**, the medians from the **non-MDMM**
> corr-noise campaign. The trained models all froze the **MDMM** campaign's set,
> **13.001228 / 21.901985 / 57.13501 mV**
> (`campaign_records/mdmm_2ns5ns/corr1e4/median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json`,
> LUT columns 225 / 375 / 975). The two files differ only by a `_mdmm` suffix and
> both describe themselves as "corr-noise, contained, 2ns/5ns".
>
> The qualitative conclusions hold — ordering, concentration in L1, sub-edge
> behaviour — but the percentages will shift (the sets differ by roughly -6 %,
> -9 %, +3 % on th1/th2/th3). **Anything compared against trained weights must be
> regenerated with the MDMM set.** Resolve thresholds from the training script,
> which names its file, not by grepping `campaign_records/`. The non-MDMM
> directory is now tagged `..._BAD-ANGLES-DO-NOT-USE`; non-MDMM runs collapse the
> angle predictions and are not a valid source for anything.

| threshold | used here (non-MDMM) | **correct (MDMM)** |
| --- | --- | --- |
| th1 | 13.809455 mV (~238 e-) | **13.001228 mV (~224 e-)** |
| th2 | 24.076480 mV (~415 e-) | **21.901985 mV (~378 e-)** |
| th3 | 55.619907 mV (~959 e-) | **57.135010 mV (~985 e-)** |

Levels `[0, 1, 2, 3]` either way.

### 3. Waveforms

Verified against `shuffled_3d/contained/train/part.0.parquet`:

- 25856 numeric columns = **101 time slices x 256 pixels**, 200 ps steps,
  spanning **0–20 ns**. Index 10 = 2 ns, index 25 = 5 ns (the two slices the
  current training uses). The 12.5 ns deadline falls at index 62.
- Values are CSA output **in mV**. For ~94 % of hit pixels the trace rises
  monotonically and saturates toward a plateau by ~20 ns, where
  plateau = `CvG * Q_in,pixel`. The remaining **6.31 %** are induced-signal
  transients that spike and decay back toward zero (3.31 % end at or below 0), so
  `Q_in` is taken from the **waveform peak**, not the plateau — see the corrections
  note under Results.
- Observed peak-pixel amplitudes reach ~444 mV (~7.7k e-), comfortably inside the
  LUT's `Q_in` coverage (which runs to 100k e-). Upper-edge clamping is a non-issue;
  the *lower* edge is the problem (see below).

## Algorithm

Per pixel, thresholds in ascending order k = 1, 2, 3:

1. `t_k` = time for the analog waveform to reach `th_k` (first crossing, from the
   101-slice trace).
2. `d_k` = `LUT(Q_th = th_k, Q_in = pixel charge)`. `Q_in` is one number per pixel
   (its collected charge), so the three lookups differ only in which `Q_th` column
   is read — giving three delays per pixel, `d_1 < d_2 < d_3`.
3. If `t_k + d_k >= 12.5 ns` -> floor to level `k-1` and stop. Else -> assign
   level `k` and continue to `k+1`.

Result: the ADC code is the highest threshold that makes the 12.5 ns window.

Note: `t_k` increases with k (higher threshold crossed later) and so does `d_k`
(less overdrive at fixed `Q_in`), so the assert time is strictly monotone in k. The sequential "floor to previous" walk therefore collapses to a
single cut — you can never have th3 make the window while th2 misses it — which
makes it cheap to vectorize. Confirmed against the real LUT numbers, not assumed.

## Results

Produced by `run_delay_study.py` (results also dumped to
`delay_study_results.json`; pass a file count as argv[1]). Noise-free, and the
**contained-cluster cut is applied** — `original_atEdge == False`, the same cut the
training pipeline makes via the DG's `select_contained=True`. Note the `contained/`
directory is *not* pre-filtered: ~53 % of its rows are atEdge. (The DG also
dropna's the recon columns; this dataset has no NaNs, so that is a no-op.)

Headline below is the **full dataset, train + test**: 100 files, 399649 raw events
-> **189956 contained clusters** (47.5 %), 48628736 pixels, of which **2584575
reach at least th1**. Pooling the splits is fine here — this characterises the
front end, no model is trained, so there is nothing to leak. The count matches the
DG's own TFR metadata exactly (31 train batches = 152037, 8 val = 37919).

The test: within the 12.5 ns auto-zero window, how many pixels drop an ADC level
once the LUT delay is applied? Both sides use the same window, so the delay is the
only difference. Asserted in the script: `d = 0` reproduces the baseline exactly.

Threshold k asserts at `t_k + d_k`. Delays are **not** summed across thresholds —
the LUT value at a given `Q_th` already supersedes the lower ones (confirmed with
the LUT authors, 2026-09-21). The serial/cumulative alternative was tested and
ruled out; its numbers are kept under "Superseded variants" below.

### `Q_in` lookup rule (decided 2026-09-22)

**Nearest tabulated row, no interpolation.** 1688 e- -> row 1700, 1610 e- -> row
1600. Below a column's first row this clamps to row 1 by construction, so the
sub-edge case and the in-range case are one rule, not two. The `Q_th` axis is
likewise nearest-column (238.1 -> 250, 415.1 -> 425, 959.0 -> 950).

### Level distribution

| | L0 | L1 | L2 | L3 |
| --- | --- | --- | --- | --- |
| baseline (no delay) | 46046455 | 308230 | 462488 | 1811563 |
| with delay | 46149283 | 250487 | 466886 | 1762080 |

### Migration, of the 74666 pixels with baseline level > 0

| changed | drop 1 | drop 2 | drop 3 | increases |
| --- | --- | --- | --- | --- |
| **7.63 %** | 7.62 % | 0.01 % | 0.00 % | 0 |

### By baseline level

Baseline levels are a genuine partition, so unlike the per-threshold rates these
are disjoint and sum correctly. The effect is overwhelmingly concentrated in L1:

| baseline level | pixels | dropped | rate |
| --- | --- | --- | --- |
| L1 | 308230 | 102515 | **33.26 %** |
| L2 | 462488 | 45085 | **9.75 %** |
| L3 | 1811563 | 49483 | **2.73 %** |
| total | 2582281 | 197083 | **7.63 %** |

### Floored pixels, exclusive by level reached

Bins are the highest threshold the waveform reaches with **no window and no
delay** — the level a perfect front end would read out. "Floored" counts anything
ending below that, whether the delay pushed it past 12.5 ns or `t_k` alone was
already past. All rates share one denominator, so they sum to the total.

| reached | pixels | floored | % of all | rate in bin |
| --- | --- | --- | --- | --- |
| th1 only | 306864 | 104585 | **4.05 %** | 34.08 % |
| th2 only | 457345 | 48745 | **1.89 %** | 10.66 % |
| th3 | 1820366 | 58286 | **2.26 %** | 3.20 % |
| **TOTAL** | **2584575** | **211616** | **8.19 %** | |

This total (8.19 %) is larger than the migration figure (7.63 %) because it also
counts the 2294 pixels that miss the window on `t_k` alone, which the
baseline-referenced view excludes.

### Cluster view

The readout is per-pixel, but the ML input is the whole 16x16 array, so the
per-cluster footprint is what matters for training:

| | |
| --- | --- |
| above-threshold pixels per cluster | 13.61 mean |
| floored pixels per cluster | 1.11 mean |
| clusters with >=1 floored pixel | 124361 (**65.47 %**) |
| clusters with >=2 | 58250 (30.66 %) |
| clusters with >=3 | 20921 (11.01 %) |
| clusters with >=5 | 1509 (0.79 %) |

So an 8 % per-pixel rate means roughly **two thirds of training examples are
perturbed**, and a third of them in more than one pixel.

A third of the lowest-level pixels are knocked to zero, while L3 barely moves.
That is the expected shape — L1 pixels sit just over threshold with small charge,
exactly where the LUT delay is largest. Worth watching in training, since those
faint cluster-edge pixels carry much of the position information.

### `t_k + delay` distribution

| threshold | p50 | p90 | p99 | max | > 12.5 ns | crossings |
| --- | --- | --- | --- | --- | --- | --- |
| th1 | 4.53 ns | 9.18 ns | 15.68 ns | 25.48 ns | 4.07 % | 2584575 |
| th2 | 4.88 ns | 9.12 ns | 14.08 ns | 22.48 ns | 2.14 % | 2277711 |
| th3 | 5.88 ns | 9.88 ns | 14.88 ns | 20.08 ns | 3.20 % | 1820366 |

### Why the per-threshold overflow rates do not sum to the migration rate

Contained part.0 (1883 events, 25543 hit pixels, 1874 changed = 7.34 %):

| | crossings | already past 12.5 ns | past 12.5 ns with delay | **newly** pushed |
| --- | --- | --- | --- | --- |
| th1 | 25572 | 29 | 996 (3.89 %) | 967 |
| th2 | 22609 | 48 | 496 (2.19 %) | 448 |
| th3 | 18070 | 87 | 552 (3.05 %) | 465 |

Three distinct reasons the raw percentages do not add to 7.34 %:

1. **Different denominators.** Each rate divides by its own threshold's crossing
   count. Put all three over hit pixels (25543) and use only the newly-pushed
   counts and they do add: 3.79 + 1.75 + 1.82 = **7.36 %** vs 7.34 % actual.
2. **Already-past pixels.** 29 + 48 + 87 = 164 crossings were past the window
   before any delay, so the baseline had already excluded them. They inflate the
   raw column and cause no migration.
3. **Nested populations.** Crossing th3 implies crossing th1 and th2, so the
   denominators are nested subsets, not disjoint bins. They reconcile exactly with
   the baseline levels: 25572-29 = 25543 = L1+L2+L3; 22609-48 = 22561 = L2+L3;
   18070-87 = 17983 = L3.

**How much is genuine double counting: very little.** By number of thresholds each
pixel fails — 2024 pixels fail exactly one, 10 fail two, **0 fail all three**. So
the 2044 raw entries are 2034 distinct pixels plus 10 multi-counted. The migration
figure itself cannot double count: it comes from a 4x4 confusion matrix, one entry
per pixel. Attributing each drop to the pixel's own top threshold gives
L1 961 + L2 448 + L3 465 = 1874 exactly.

The overlap stays small because failing two thresholds needs a pixel to cross both
*and* be slow on both, while crossing th2/th3 requires large `Q_in`, which means a
small delay. The two conditions work against each other. A steeper delay curve
would widen that band.

### Sample size

The headline now uses the **full training set** (80 files, 152037 contained events
of 320000; ~47.5 % survive the contained cut). Convergence across sample sizes:
part.0 alone 7.34 %, 12 files 7.56 %, all 80 files **7.64 %** — a single file runs
about 0.3 pp low, so the full run is worth the ~20 min. `run_delay_study.py N`
takes the file count as argv[1].

### Superseded variants

**Serial/cumulative delay model** (`t_k + sum_{i<=k} d_i`), ruled out once the LUT
authors confirmed each entry already supersedes the lower thresholds: it gave
36.55 % migration (pooled 36.72 %), with 13.85 % / 30.56 % of th2 / th3 crossings
pushed past the window. It was also inconsistent with the table itself —
`d_3 < d_1 + d_2` everywhere (2.71 vs 4.60 ns at `Q_in` = 2417 e-), so the delays
cannot be a running total. Removed from the script.

**Linear interpolation in `Q_in`** with three sub-edge policies, before the
nearest-row rule was decided: clamp 7.01 %, linear extrapolation below the edge
10.76 %, never-fires 43.15 %. The chosen nearest-row rule sits alongside clamp, as
expected — nearest-row and linear interpolation differ by at most half a row step.

Supporting numbers that informed the choice: the delay does **not** collapse onto a
universal `Q_in/Q_th` curve (at 2x overdrive the normalised delay runs from 1.22 at
`Q_th`=3000 to 5.13 at `Q_th`=75), so low-`Q_th` columns cannot be borrowed to fill
the untabulated region. Extrapolating each column's own first-row gradient gave
sub-edge delays of 4-10 ns (th1 p50 9.19, th2 5.67, th3 4.02 ns), all inside the
12.5 ns window — which is why never-fires was rejected: those pixels do fire.

### Reading these

1. **The thresholds survive in every variant.** All movement is strictly downward
   (zero increases, as it must be), all three levels stay populated, and drops are
   essentially all single-level — multi-level drops stay under 1 % everywhere. The
   effect is a blurring of the level boundaries, not a collapse.
2. **The answer spans 7 % to 46 %** depending on two modelling choices that are
   still open. Ranked by how much they matter:
   - serial vs parallel, under the clamp policy: 7.0 % -> 37.6 %
   - clamp vs nofire, under the parallel model: 7.0 % -> 43.7 %
   - under nofire the two models nearly converge (43.7 % vs 46.3 %), because the
     sub-edge population dominates and is floored either way.
3. **th1 is identical in both models**, since the cumulative sum at k=1 is just
   `d_1`. The models only diverge at th2 and th3 — which is where the serial
   model's overflow becomes severe (15 % and 31 % of crossings past the window,
   against 2-3 % for parallel).
4. **Model A is confirmed** (collaborators, 2026-09-21): the delay tabulated at a
   given `Q_th` already accounts for the lower thresholds, so `d_k` is used on its
   own and never summed. Model B is therefore out, and the answer is **7.02 %**
   (clamp) or **43.67 %** (nofire). Note the numbers say `d_k` is an *absolute*
   elapsed time to that comparator firing, not a literal running total — across the
   whole table `d_3 < d_1 + d_2` (2.71 vs 4.60 ns at `Q_in` = 2417 e-), so the
   earlier delays are superseded rather than stacked. Its reference point must be
   the threshold crossing rather than charge arrival, since `d_3 = 2.71 ns` is less
   than the analog rise time `t_3 = 3.45 ns` for that same pixel — which is what
   makes `t_k + d_k` correct rather than double-counting.
5. **The LUT's own shape independently argues the same way.** `d_k` scales with
   overdrive — large at low `Q_in`, growing with `Q_th` — which is the time-walk
   signature of an independent comparator, not a fixed conversion step. And all
   columns converge to a common 1.4-1.7 ns floor at high `Q_in`, consistent with
   three identical comparators each hitting their asymptotic propagation delay.
   Under a cumulative reading that floor would instead imply ~4.5 ns to reach th3
   no matter how large the signal. This is inference from the table, not
   documentation — it needs confirming against the actual architecture.

## Where this has to live

Not in `__getitem__`. The current flow is:

```
parquet (101 slices) --[prepare_batch_data: keep slices 10,25 + add noise]--> TFRecord (16,16,2)
TFRecord --[__getitem__: map_to_levels / Bucketize]--> digitized batch
```

`map_to_levels` runs at *training* time on TFRecords that have already thrown away
99 of the 101 time slices. Crossing times cannot be recovered there. So the delay
digitization must happen at **TFRecord creation time**, inside
`prepare_batch_data`, while the full waveform is still in hand — writing a new,
already-digitized TFRecord set (e.g. `TFR_files_2_5_noise_corr_contained_delay`)
that training then consumes with `digitize=False`.

Cost note: reading all 101 slices is ~50x the current column I/O (~400 MB/file at
4000 rows x 25856 float32). Only slices up to index 62 (12.5 ns) are needed, and
first-crossing can be accumulated in time-chunks rather than materializing the
whole array.

## Settled

- **Delays are not summed across thresholds.** Assert time is `t_k + d_k`; the LUT
  value at a given `Q_th` already supersedes the lower ones (LUT authors,
  2026-09-21).
- **Sub-edge `Q_in` uses row 1**, and **in-range `Q_in` uses the nearest row**, no
  interpolation (2026-09-22). One rule covers both. `Q_th` likewise nearest-column.
- **`Q_in` is the waveform peak**, not the 20 ns plateau — ~6 % of hit pixels are
  induced-signal transients that decay back to ~0.
- **The study stays noiseless** (2026-09-22). Note this pairs the correlated-noise
  campaign's thresholds with noise-free waveforms, which is deliberate.

## Open questions

1. **Where do the two sampling slices go?** The 12.5 ns window fixes the level
   behaviour, but the model input is still two samples. Left at 2 ns and 5 ns the
   first channel sees almost nothing once delays are applied (median `t_k + d_k` is
   4.5-5.9 ns). Re-choosing those times — somewhere past the bulk of the
   `t_k + d_k` distribution — is the natural follow-up.
2. **One code per pixel, or one per time slice?** A single 12.5 ns latch gives one
   ADC code per pixel, so the two-slice structure would collapse unless the slices
   are sampled separately from the delayed comparator outputs. This decides the
   output shape and needs settling before any TFRecord is written.
3. **Why does the LUT tabulation stop at `Q_in` ~ 1.9 * `Q_th`?** Not blocking now
   that row 1 is used for that region, but 11-27 % of crossings land there, so it
   would be worth knowing whether it is a convergence limit or a design-spec sweep
   boundary. Extending the LUT below 2x overdrive would remove the guesswork.
4. **Gain consistency.** Was the LUT (`PixB_27c`) generated with the same
   `CvG = 58e-6` V/e- front end as our mV dataset? If not, the charge<->mV
   conversion on the two sides is inconsistent.
