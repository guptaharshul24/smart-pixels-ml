# Delay-aware ADC digitization

Status: **pass-through complete** (2026-10-07). Three delayed validation sets
generated and evaluated against the frozen Stage 2 and Stage 2.5 no-noise weights.
No retraining attempted. The largest open uncertainty is column 975 of the LUT,
which carries 68 % of the delayed pixels and still misbehaves near threshold.

## Goal

The current hard digitization (`DG/OptimizedDataGenerator_v3.map_to_levels`)
assumes a comparator with **zero delay**: a pixel's mV value is bucketized against
the thresholds instantly. Real front ends have charge-dependent *time walk* — a
pixel sitting just above threshold fires late, and if it fires late enough it
misses the readout window and reads out one ADC level low.

This work injects that effect from a delay lookup table characterised from the
circuit design, and measures what it costs the trained networks. The readout
sampling is then advanced to see how much of the loss is recoverable by re-timing
alone, without retraining.

## Inputs

### 1. Delay LUT

`VIZARD_ADC_LUT_PixB_27c_global.csv`, a 2D grid: a `Qin_e` index column and one
`Qth_<n>` column per threshold, blank where untabulated. 235 Qin rows x 264 Qth
columns. It replaces the earlier 120 ragged column-pair layout, kept as
`VIZARD_ADC_LUT_PixB_27c_combined_sorted.csv`, and an intermediate version with a
coverage hole, kept as `..._global_hole_in_th3.csv`.

Parsed by `lut_loader.py`. Columns are taken from the **25 e- grid only**, giving
225 / 375 / 975 for our thresholds. The table also carries finer off-grid columns
closer to arbitrary threshold values, but those are near-threshold patches rather
than full columns, and mixing them with grid columns would make the threshold the
delay model assumes drift with `Q_in`.

Values agree with the old table to machine precision above 1100 e- (92 shared rows
per column). Below that the old bottom-edge rows were revised downward by up to
0.68 ns, which is where the original sweep was least converged.

**Coverage, old table vs new**, for the three columns in use:

| | threshold | old edge | new edge | old sub-edge band | new band |
| --- | --- | --- | --- | --- | --- |
| th1 | 13.00 mV | 23.20 mV | 14.50 mV | 13.00-23.20 | 13.00-14.50 |
| th2 | 21.90 mV | 40.60 mV | 22.62 mV | 21.90-40.60 | 21.90-22.62 |
| th3 | 57.14 mV | 104.40 mV | 56.84 mV | 57.14-104.40 | **none** |

For th3 the edge now sits below the threshold, so a pixel that crosses th3 is
always tabulated. Sub-edge clamping goes from the dominant effect, 11-30 % of
crossings, to a corner case. `Q_in` also steps 10 e- from 230 to 1000 rather than
100 e-, which matters because the gradient near the edge of column 225 is about
16.6 ns per 1000 e-.

**Known problem, column 975.** Columns 225 and 375 peak at their first row
(15.867 and 13.653 ns) and fall monotonically, as time-walk should. Column 975
peaks at `Q_in` = 1050 (4.338 ns) and *rises* from 980 to 1050, and its first row
at 1.005x overdrive gives only 2.749 ns where the other columns reach 13-16 ns at
comparable overdrive. Its dense and coarse sweeps also fail to join: the
1700 -> 1800 step is +0.406 ns against +0.023 and +0.066 for the other two.
Raised with the LUT author; the numbers below carry it.

### 2. Thresholds

Medians from the **MDMM** campaign
(`campaign_records/mdmm_2ns5ns/corr1e4/median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json`),
the set every trained Stage 1.5 / 2 / 2.5 froze, tagged
`fixed_thr_13.00_21.90_57.14` in their run dirs.

| threshold | mV | electrons | LUT column |
| --- | --- | --- | --- |
| th1 | 13.001228 | ~224 | 225 |
| th2 | 21.901985 | ~378 | 375 |
| th3 | 57.135010 | ~985 | 975 |

> Work before 2026-09-23 used 13.809455 / 24.07648 / 55.619907, the **non-MDMM**
> campaign's medians. The two files differ only by a `_mdmm` suffix and describe
> themselves identically. Non-MDMM runs collapse the angles, so that campaign is
> tagged `..._BAD-ANGLES-DO-NOT-USE`. Resolve thresholds from the training script,
> which names its file, not by grepping `campaign_records/`.

### 3. Waveforms

`shuffled_3d/contained/{train,test}/part.*.parquet`: 101 time slices x 256 pixels,
200 ps steps, 0-20 ns, CSA output in mV. The `contained/` directory is **not**
pre-filtered; ~53 % of its rows are `original_atEdge == True`, which the DG drops
via `select_contained=True`. Any script reading these parquets directly must apply
that cut itself.

`Q_in` comes from the waveform **peak**, not the 20 ns plateau: ~6 % of hit pixels
are induced-signal transients that spike and decay back to ~0, sometimes negative,
for which a plateau-based charge is meaningless. They peak just over threshold
(median 16.3 mV) and last a single 200 ps slice in 98 % of cases.

## Algorithm

Per pixel:

1. `Q_in` = peak / `CvG`, with `CvG` = 0.058 mV/e-.
2. `k_max` = highest threshold the waveform reaches at all.
3. `d` = LUT at column `k_max`, nearest tabulated row to `Q_in`. One delay per
   pixel, not three. The LUT gives a separate delay per threshold; a single value
   is what makes a rigid slide of the sampling well defined, and `d_kmax` is the
   conservative choice since it over-delays the lower thresholds.
4. Sample at shifted slice **indices**, no interpolation:
   `i = round((T + shift - d) / 0.2)` for `T` = 2 ns and 5 ns, value 0 when
   `T + shift - d <= 0`. Rounding the index also quantises `d` to 200 ps.
5. `cap` = highest k with `t_k + d_k <= 12.5 ns`, where `t_k` is the first slice
   at or above `th_k`. The window is **fixed** and does not move with the shift:
   it is a property of the front-end reset, not of where the ADC samples.
6. `level = min(Bucketize(sampled mV, thresholds), cap)`.

No interpolation anywhere. The waveforms are 200 ps native, downsampled from 10 ps
before being written to disk, and the DG has never interpolated: it selects slices
by integer index and the model only ever sees stored slice values.

## Per-pixel delays

Over the val set, 526122 pixels with `k_max >= 1`:

| | column | pixels | share | min | median | mean | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| th1 | 225 | 62130 | 0.118 | 6.411 | 10.429 | 11.163 | 15.867 |
| th2 | 375 | 104194 | 0.198 | 3.547 | 4.869 | 5.758 | 13.653 |
| th3 | 975 | 359798 | 0.684 | 1.694 | 2.677 | 2.770 | 4.338 |
| **all** | | **526122** | | **1.694** | **3.117** | 4.353 | 15.867 |

The three populations barely overlap: th1 pixels get 6.4-15.9 ns and th3 pixels
1.7-4.3 ns, so no single shift suits both.

Statistics must be taken over **assigned per-pixel delays**, not over LUT cells.
Pooling cells gives 2.086 ns, which is wrong for two reasons: the table's rows are
unevenly spaced in charge, so cell-pooling weights by sweep design rather than
physics; and a `k_max = 1` pixel sits by definition in the narrow band between th1
and th2 at the very bottom of column 225, while that column's 168 rows run to
100000 e-. Its cells median at 2.423 ns, its pixels at 10.429 ns.

## Pass-through results

Three validation sets, generated by `generate_tfr_delay_val.py --shift N`:

| set | shift | nominal slices | rationale |
| --- | --- | --- | --- |
| `..._delay_shift0` | 0 | 10, 25 | delay applied, readout unchanged |
| `..._delay_shift1p6` | 1.6 ns | 18, 33 | minimum per-pixel delay, 1.694 rounded |
| `..._delay_shift3p2` | 3.2 ns | 26, 41 | median per-pixel delay, 3.117 rounded |

Each stores **levels**, not mV, because a window cap cannot be expressed as a
voltage without fabricating one. Evaluate with `digitize=False`. Events, order,
labels and `metadata.json` are inherited from the undelayed set, and each batch
asserts that a zero-shift rebuild reproduces the source mV bit-for-bit.

### Occupancy

| | ch0 (2 ns) lit | ch1 (5 ns) lit | total | of baseline |
| --- | --- | --- | --- | --- |
| baseline | 261171 | 471473 | 732644 | 100 % |
| shift 0 | **0** | 257709 | 257709 | 35 % |
| shift 1.6 ns | 53220 | 383208 | 436428 | 60 % |
| shift 3.2 ns | 279324 | 422178 | 701502 | 96 % |

At shift 0 channel 0 is entirely dead, because the smallest per-pixel delay,
1.694 ns, already exceeds the 2 ns sample time. 1.6 ns recovers only the fastest
pixels. 3.2 ns restores occupancy to 96 % of baseline.

### Residual std, same weights throughout

Stage 2, Conv2D, `64d9b19b`:

| | x | y | cotA | cotB |
| --- | --- | --- | --- | --- |
| baseline | 12.28 | 3.15 | 0.475 | 0.175 |
| shift 0 | 28.16 | 9.03 | 3.954 | 1.060 |
| shift 1.6 ns | 18.07 | 5.62 | 2.492 | 0.808 |
| shift 3.2 ns | **15.73** | **3.91** | 2.750 | **0.647** |

Stage 2.5, QConv2D, `e61b24cc`:

| | x | y | cotA | cotB |
| --- | --- | --- | --- | --- |
| baseline | 14.38 | 3.48 | 0.651 | 0.208 |
| shift 0 | 30.27 | 8.01 | 4.332 | 1.065 |
| shift 1.6 ns | 21.63 | 5.28 | 2.863 | 0.809 |
| shift 3.2 ns | **17.40** | **4.13** | 2.913 | **0.644** |

Position largely recovers: Stage 2's x goes 28.16 -> 15.73 against a 12.28
baseline, and y to 3.91 against 3.15. Biases clear too, Stage 2.5's x mean moving
from -5.61 to +0.30.

Angles do not. `cotA` improves to 2.75 but stays about 5.8x the baseline, and
1.6 ns is marginally better than 3.2 ns for `cotA` in both models while being worse
for everything else. Angle information lives in the difference between the two
channels, which a uniform shift cannot restore.

The distribution plots carry a detail the residual widths hide: at shift 3.2 the
angle correlations are **+0.61 and +0.63** with predicted spread 0.59-0.68 of true,
so the predictions track truth but hedge toward the centre. An earlier run on the
superseded LUT had them **anti-correlated** at -0.38. Both look merely "degraded"
by residual width alone, which is why every eval now emits these plots.

## Files

| file | what it does |
| --- | --- |
| `lut_loader.py` | parses the 2D LUT, 25 e- grid columns, cross-checks against the old table |
| `lut_delay_stats.py` | per-column and pooled delay statistics for thresholds given on the command line |
| `generate_tfr_delay_val.py` | builds a delayed val set for a given `--shift` |
| `run_delay_study.py` | standalone 12.5 ns window characterisation, `argv[1]` = file count |
| `delay_study_results.json` | counts and confusion matrices from the last window study |

Eval outputs live in `plotting/part2_no_noise_delay/64d9b19b_shift{0,1p6,3p2}/`
and `plotting/part2p5_no_noise_delay/e61b24cc_shift{0,1p6,3p2}/`. The September
run on the superseded LUT is kept alongside as `*_SUPERSEDED_oldLUT/`, and the two
superseded TFR sets as `..._delay_SUPERSEDED_*` on `/work`.

## Settled

- **Delays are not summed across thresholds.** Fire time is `t_k + d_k`; the LUT
  value at a given `Q_th` already supersedes the lower ones (LUT authors,
  2026-09-21).
- **One delay per pixel**, `d_kmax`, which is what makes a rigid slide of the
  sampling well defined.
- **Nearest tabulated row for `Q_in`**, nearest 25 e- grid column for `Q_th`, no
  interpolation on either axis.
- **`Q_in` is the waveform peak**, not the 20 ns plateau.
- **No interpolation between time slices.** Sampling is index selection at the
  native 200 ps granularity, which also quantises the delay. The DG has never
  interpolated and the model only sees stored slice values.
- **The 12.5 ns window is fixed** and does not move with the readout shift, so the
  level cap is identical across all three sets and they differ in one variable.
- **Output stores levels**, evaluated with `digitize=False`, since a window cap
  cannot be expressed as a voltage without fabricating one.
- **The study stays noiseless**, deliberately pairing the correlated-noise
  campaign's thresholds with noise-free waveforms.
- **Readout shifts are 0, 1.6 and 3.2 ns**, from the per-pixel minimum and median
  delay rounded to the slice grid.

## Open questions

1. **Column 975's near-threshold rows.** Delay rises from 980 to 1050 e- instead of
   falling, its first row gives 2.749 ns where the other columns reach 13-16 ns at
   comparable overdrive, and its dense and coarse sweeps disagree by 0.406 ns at
   the 1700 -> 1800 seam. That column carries 68 % of the delayed pixels, so this
   is the largest open uncertainty in the numbers. Raised with the LUT author.
2. **Angles do not recover with a uniform shift.** `cotA` stays about 5.8x baseline
   at every shift tried, and the minimum shift is marginally better for it than the
   median while being worse for everything else. Angle information lives in the
   difference between the two channels, so a per-channel or per-threshold timing
   scheme may be needed rather than one global offset.
3. **Gain consistency.** Was the LUT (`PixB_27c`) generated with the same
   `CvG = 58e-6` V/e- front end as our mV dataset? If not, the charge to mV
   conversion on the two sides is inconsistent. The raw-charge dataset is the place
   to check, bearing in mind it is `3sr` rather than this set's `3srb`.
4. **Training on the delayed data** has not been attempted. Everything here is a
   pass-through with frozen weights, which bounds the damage but says nothing about
   how much a retrained network could recover.
