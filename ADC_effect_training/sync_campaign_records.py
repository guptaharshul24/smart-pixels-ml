"""
Copies the small provenance files (threshold-run journal + median-thresholds
JSON) out of a campaign's (large, external-filesystem) trained_models_dir
into this repo's ADC_effect_training/campaign_records/<dest_subdir>/, so seed/
threshold provenance is git-tracked even though the actual training output
(checkpoints, TFRecords) never is.

Zero heavy deps (no TF/keras) -- safe to import directly into any
orchestrator, including TF-free ones, without adding CUDA-context overhead.

Idempotent: safe to call repeatedly (each call just overwrites with the
current file contents) and safe to call before the median JSON exists yet
(median_thresholds_*.json just won't be copied until it's written).
"""
import os
import glob
import shutil

_REPO_ADC_DIR = os.path.dirname(os.path.abspath(__file__))


def sync_campaign_records(trained_models_dir, dest_subdir,
                           jsonl_pattern="threshold_runs_*.jsonl",
                           median_pattern="median_thresholds_*.json"):
    """Copy threshold_runs_*.jsonl + median_thresholds_*.json (if present)
    from trained_models_dir into ADC_effect_training/campaign_records/<dest_subdir>/.
    Returns the list of destination paths actually copied."""
    dest = os.path.join(_REPO_ADC_DIR, "campaign_records", dest_subdir)
    os.makedirs(dest, exist_ok=True)

    copied = []
    for pattern in (jsonl_pattern, median_pattern):
        for src in glob.glob(os.path.join(trained_models_dir, pattern)):
            dst = os.path.join(dest, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trained_models_dir")
    parser.add_argument("dest_subdir", help="subdir under ADC_effect_training/campaign_records/")
    args = parser.parse_args()
    copied = sync_campaign_records(args.trained_models_dir, args.dest_subdir)
    if copied:
        for p in copied:
            print(f"synced: {p}")
    else:
        print("nothing to sync (no matching files found)")
