"""
Builds a cotBeta-restricted, statistics-matched subsample of the raw
pixelAV charge dataset, for a fair(er) comparison against our own
frontend-effects pipeline. Targets the ADC-effects dataset's REAL numbers
exactly: 155,000 train + 40,000 val (confirmed 2026-09-02 from
TFR_files_2_5_noise_corr_contained's actual generated TFRecords), not a
generic 195,000-then-split scheme.

Source: /work/projects/SmartPixML/datasets_16x16x20_charge/
        dataset_3sr_16x16_50x12P5_centeredIncidence_parquets/{train,test}/
        (1,441,454 + 360,799 = 1,802,253 raw-charge events, in electrons,
        NOT noise-injected -- noise/normalization happen later at
        TFRecord-generation time, same as every other dataset in this repo)

pixelAV's own train/ vs test/ split is NOT preserved -- it has no special
meaning for us. All eligible rows from both source dirs are pooled into one
set, one single global random draw of 195,000 is taken from that pool,
shuffled once, then simply SLICED: first 155,000 rows -> train, remaining
40,000 -> test. No second sampling step -- 155,000 + 40,000 = 195,000
exactly, so "remaining" and "40,000" are the same thing.

Selection: original_atEdge == False (containment) AND |cotBeta| < 2.0,
applied HERE (not deferred to TFR-gen time) so the exact target row counts
can be hit precisely. (Deferring containment to TFR-gen, tried first, makes
the post-containment count a function of which specific rows survive --
knowable only after the fact, not something you can target exactly without
already knowing containment status at selection time anyway.)
select_contained=True is still passed at TFR-gen for consistency with every
other generate_tfr_*.py script's call pattern -- a safe no-op here, since
every row already satisfies original_atEdge==False and that column is still
present in the output (no columns dropped during filtering).

Confirmed (2026-09-02) this selection is what narrows cotAlpha's realized
range (std 4.976 -> 3.458) and shifts y-midplane's mean (+6.26 -> +8.90) --
NOT the cotBeta cut itself. That narrowing is expected and unavoidable
(containment is geometrically coupled to both angles), not a sampling
artifact -- not corrected for.

pT flatness: pT has a genuine dip at pT~0 in the raw source (not flat to
begin with) that containment flips into a spike (confirmed directly against
pixelAV's own real noisy TFR set's actual selected rows, via its
metadata.json's batch_metadata -- ratio 1.20 vs. a baseline window, same
magnitude either way it's measured). A pT-stratified-sampling correction
was built and tested, then dropped: forcing flatness pre-containment is
moot since it doesn't survive containment regardless, and containment can't
be avoided (needed to match the ADC-effects dataset's own selection). Not
corrected for; pT reflects whatever this selection naturally produces.

Landing on exact multiples of 5,000 (155,000 = 31x, 40,000 = 8x) means
TFR-gen's batching produces clean, full-only batches with no tail-splitting
-- matching the ADC-effects dataset's own file structure (31/8 files)
exactly, not just approximately.

Row order: verified the source parquets are already shuffled (corr(row
position, cotAlpha/cotBeta/pt) all ~0.0001-0.008, i.e. no positional
structure). Filtering by value can't reintroduce positional structure, but
concatenating selected rows file-by-file could still group them if there's
any subtle inter-file batch effect -- so this script does one explicit
np.random.default_rng(SEED).permutation() on the selected rows before
writing, as a safety net, regardless.
"""
import os
import glob
import numpy as np
import pandas as pd

SEED = 20260902
SRC_DIR = "/work/projects/SmartPixML/datasets_16x16x20_charge/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets"
OUT_DIR = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
COTB_CUT = 2.0
TRAIN_TARGET = 155_000
VAL_TARGET = 40_000
TOTAL_TARGET = TRAIN_TARGET + VAL_TARGET
ROWS_PER_FILE = 5000


def eligible_row_map(src_files):
    """Scan every file, reading only the small filter columns, and return a
    list of (file_path, local_row_indices) for rows passing containment +
    |cotBeta| < COTB_CUT."""
    out = []
    total = 0
    for f in src_files:
        df = pd.read_parquet(f, columns=["cotBeta", "original_atEdge"])
        mask = (~df["original_atEdge"]) & (df["cotBeta"].abs() < COTB_CUT)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx):
            out.append((f, idx))
            total += len(idx)
    return out, total


def write_split(df_slice, out_dir, rows_per_file):
    os.makedirs(out_dir, exist_ok=True)
    n = len(df_slice)
    assert n % rows_per_file == 0
    n_files = n // rows_per_file
    for i in range(n_files):
        chunk = df_slice.iloc[i * rows_per_file:(i + 1) * rows_per_file]
        chunk.to_parquet(os.path.join(out_dir, f"part.{i}.parquet"))
    print(f"wrote {n_files} files ({rows_per_file} rows/file, {n:,} rows total) -> {out_dir}")


def main():
    rng = np.random.default_rng(SEED)

    src_files = (sorted(glob.glob(os.path.join(SRC_DIR, "train", "part.*.parquet"))) +
                 sorted(glob.glob(os.path.join(SRC_DIR, "test", "part.*.parquet"))))
    row_map, total_eligible = eligible_row_map(src_files)
    print(f"eligible rows (contained & |cotBeta|<{COTB_CUT}): {total_eligible:,} "
          f"across {len(row_map)} source files (target {TOTAL_TARGET:,})")
    if total_eligible < TOTAL_TARGET:
        raise SystemExit(f"only {total_eligible:,} eligible rows, need {TOTAL_TARGET:,}")

    # global (file_idx, local_row) arrays, draw TOTAL_TARGET without replacement
    flat_file_idx = np.concatenate([np.full(len(idx), fi) for fi, (_, idx) in enumerate(row_map)])
    flat_local_row = np.concatenate([idx for _, idx in row_map])
    chosen = rng.choice(len(flat_file_idx), size=TOTAL_TARGET, replace=False)

    # group chosen rows by source file so each file is read once
    per_file = {}
    for c in chosen:
        fi = flat_file_idx[c]
        per_file.setdefault(fi, []).append(flat_local_row[c])

    frames = []
    for fi, rows in per_file.items():
        fpath, _ = row_map[fi]
        df = pd.read_parquet(fpath)
        frames.append(df.iloc[sorted(rows)])
    selected = pd.concat(frames, ignore_index=True)

    # explicit final shuffle -- see module docstring
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    assert len(selected) == TOTAL_TARGET

    # sanity gate before writing anything
    cb = selected["cotBeta"]
    assert cb.abs().max() < COTB_CUT, f"cotBeta cut violated: max={cb.abs().max()}"
    print(f"cotBeta range=[{cb.min():+.3f},{cb.max():+.3f}]")

    train_slice = selected.iloc[:TRAIN_TARGET]
    val_slice = selected.iloc[TRAIN_TARGET:TRAIN_TARGET + VAL_TARGET]
    assert len(train_slice) == TRAIN_TARGET and len(val_slice) == VAL_TARGET

    write_split(train_slice, os.path.join(OUT_DIR, "train"), ROWS_PER_FILE)
    write_split(val_slice, os.path.join(OUT_DIR, "test"), ROWS_PER_FILE)


if __name__ == "__main__":
    main()
