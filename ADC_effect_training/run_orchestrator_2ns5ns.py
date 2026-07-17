"""
Orchestrator for 2ns/5ns threshold search.
No TF imports — the subprocess gets the full GPU memory budget (no CUDA context leak).
Calls train_vit_part1_rnd_thr_noise_corr_contained_2ns5ns.py --seed X --run_index Y per run.
"""

import os
import sys
import json
import time
import logging
import subprocess
import numpy as np

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_loop_rnd_thr_noise_corr_contained_2ns5ns.out"),
        logging.StreamHandler(),
    ],
)

# ── constants ─────────────────────────────────────────────────────────────────
TRAINED_MODELS_DIR = (
    "/home/harshul-cern/work/projects/SmartPixML/"
    "dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/"
    "shuffled_3d/trained_models_2_5_noise_corr_contained"
)
JSONL_PATH = os.path.join(
    TRAINED_MODELS_DIR,
    "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns.jsonl",
)
MEDIAN_PATH = os.path.join(
    TRAINED_MODELS_DIR,
    "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns.json",
)
TRAINING_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "train_vit_part1_rnd_thr_noise_corr_contained_2ns5ns.py",
)
PYTHON = sys.executable

# Same seed pool as the training script (deterministic, generator seeded with 20260627)
SEEDS = [int(s) for s in np.random.default_rng(20260627).integers(0, 2**31 - 1, size=40)]

TARGET_RUNS = 5
MAX_OOM_RETRIES = 3
RETRY_WAIT_S = 30


def load_collected():
    records = []
    if os.path.exists(JSONL_PATH):
        for line in open(JSONL_PATH):
            if line.strip():
                records.append(json.loads(line))
    return records


def write_median(records):
    non_stuck = [r for r in records if not r.get("stuck", False)]
    if not non_stuck:
        return
    thresholds = [r["final_thresholds"] for r in non_stuck]
    n_thr = len(thresholds[0])
    median_thr = [float(np.median([t[i] for t in thresholds])) for i in range(n_thr)]
    summary = {
        "n_runs": len(non_stuck),
        "median_thresholds": median_thr,
        "levels": [0.0, 1.0, 2.0, 3.0],
        "time_stamps": [10, 25],
        "note": "median over non-stuck runs; per-run details in threshold_runs_rnd_thr_noise_corr_contained_2ns5ns.jsonl",
    }
    with open(MEDIAN_PATH, "w") as f:
        json.dump(summary, f, indent=1)
    logging.info(f"Median written: {len(non_stuck)} runs, thresholds={median_thr}")


def main():
    logging.info("=== Orchestrator started (no TF — full GPU budget for subprocesses) ===")

    done_seeds = {r["seed"] for r in load_collected()}
    completed_runs = sum(1 for r in load_collected() if not r.get("stuck", False))
    if done_seeds:
        logging.info(
            f"Resuming: {len(done_seeds)} seed(s) already in JSONL, "
            f"{completed_runs} completed (non-stuck)."
        )

    for run_index, run_seed in enumerate(SEEDS):
        if completed_runs >= TARGET_RUNS:
            logging.info(f"Reached TARGET_RUNS={TARGET_RUNS}. Done.")
            break
        if run_seed in done_seeds:
            logging.info(f"Skipping seed={run_seed} (already in JSONL).")
            continue

        oom_retries = 0
        while True:
            logging.info(
                f"Launching subprocess seed={run_seed}, run_index={run_index} "
                f"(attempt {oom_retries + 1}/{MAX_OOM_RETRIES + 1})"
            )
            result = subprocess.run(
                [PYTHON, TRAINING_SCRIPT,
                 "--seed", str(run_seed),
                 "--run_index", str(run_index)],
                env=os.environ.copy(),
            )
            if result.returncode == 0:
                new_rec = next(
                    (r for r in load_collected() if r["seed"] == run_seed), None
                )
                if new_rec is None:
                    logging.error(
                        f"Subprocess exited 0 but no JSONL record for seed={run_seed}; skipping."
                    )
                    break
                stuck = new_rec.get("stuck", False)
                if not stuck:
                    completed_runs += 1
                logging.info(
                    f"--- Run done: {completed_runs}/{TARGET_RUNS} "
                    f"(run_index={run_index}, seed={run_seed}, stuck={stuck}) ---"
                )
                break
            else:
                oom_retries += 1
                if oom_retries > MAX_OOM_RETRIES:
                    logging.error(
                        f"Subprocess failed {MAX_OOM_RETRIES} times for seed={run_seed}; skipping."
                    )
                    break
                logging.warning(
                    f"Subprocess failed (exit code {result.returncode}) "
                    f"seed={run_seed} retry {oom_retries}/{MAX_OOM_RETRIES}; waiting {RETRY_WAIT_S}s."
                )
                time.sleep(RETRY_WAIT_S)

    write_median(load_collected())
    logging.info("=== Orchestrator finished ===")


if __name__ == "__main__":
    main()
