"""
Orchestrator for the 2ns/5ns MDMM threshold search.
No TF imports -- the subprocess gets the full GPU memory budget (no CUDA context leak).

Bookkeeping (all state lives in the JSONL, nothing in code):
- Seeds are drawn fresh from OS entropy per run (no fixed pool). Reproducible
  because the training subprocess journals a "started" event (seed, fingerprint,
  init thresholds) before training, and fingerprint/thresholds derive
  deterministically from the seed.
- Resume: "started" events without a matching "completed" event are resumed first
  (the training script's mid-run checkpoint resume picks up from the last epoch).
- Failures: every non-zero subprocess exit is journaled as a "failed" event with
  the return code and attempt number; seeds that exhaust their retries are
  journaled as "abandoned" and never retried again. The JSONL is a full audit
  trail of the campaign.
"""

import os
import sys
import json
import time
import secrets
import logging
import subprocess
import numpy as np

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.out"),
        logging.StreamHandler(),
    ],
)

# ── constants ─────────────────────────────────────────────────────────────────
TRAINED_MODELS_DIR = (
    "/home/harshul-cern/work/projects/SmartPixML/"
    "dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/"
    "shuffled_3d/trained_models_2_5_noise_corr_contained_2ns5ns_mdmm"
)
JSONL_PATH = os.path.join(
    TRAINED_MODELS_DIR,
    "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl",
)
MEDIAN_PATH = os.path.join(
    TRAINED_MODELS_DIR,
    "median_thresholds_rnd_thr_noise_corr_contained_2ns5ns_mdmm.json",
)
TRAINING_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.py",
)
PYTHON = sys.executable

TIME_STAMPS = [10, 25]
TARGET_RUNS = 5
MAX_RETRIES_PER_SEED = 3
RETRY_WAIT_S = 30


def load_events():
    events = []
    if os.path.exists(JSONL_PATH):
        for line in open(JSONL_PATH):
            if line.strip():
                events.append(json.loads(line))
    return events


def append_event(rec):
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def completed_records(events=None):
    events = load_events() if events is None else events
    return [e for e in events if e.get("status", "completed") == "completed"]


def campaign_state():
    """Returns (n_completed_non_stuck, pending [(seed, run_index)...], next_run_index)."""
    events = load_events()
    completed_seeds = {e["seed"] for e in completed_records(events)}
    abandoned_seeds = {e["seed"] for e in events if e.get("status") == "abandoned"}
    n_completed = sum(1 for e in completed_records(events) if not e.get("stuck", False))

    pending, seen = [], set()
    for e in events:
        if (e.get("status") == "started"
                and e["seed"] not in completed_seeds
                and e["seed"] not in abandoned_seeds
                and e["seed"] not in seen):
            pending.append((e["seed"], e["run_index"]))
            seen.add(e["seed"])

    indices = [e["run_index"] for e in events if "run_index" in e]
    next_index = (max(indices) + 1) if indices else 0
    return n_completed, pending, next_index


def write_median():
    non_stuck = [r for r in completed_records() if not r.get("stuck", False)]
    if not non_stuck:
        return
    thresholds = np.array([r["final_thresholds"] for r in non_stuck])
    summary = {
        "n_runs": len(non_stuck),
        "median_thresholds": np.median(thresholds, axis=0).tolist(),
        "levels": [0.0, 1.0, 2.0, 3.0],
        "time_stamps": TIME_STAMPS,
        "note": "median over non-stuck runs; per-run details in threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl",
    }
    with open(MEDIAN_PATH, "w") as f:
        json.dump(summary, f, indent=1)
    logging.info(f"Median written: {summary['n_runs']} runs, thresholds={summary['median_thresholds']}")


def run_one(seed, run_index):
    """Launch the training subprocess for one seed with retries. Returns True if completed."""
    for attempt in range(1, MAX_RETRIES_PER_SEED + 2):
        logging.info(f"Launching subprocess seed={seed}, run_index={run_index} "
                     f"(attempt {attempt}/{MAX_RETRIES_PER_SEED + 1})")
        result = subprocess.run(
            [PYTHON, TRAINING_SCRIPT, "--seed", str(seed), "--run_index", str(run_index)],
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            rec = next((r for r in completed_records() if r["seed"] == seed), None)
            if rec is None:
                logging.error(f"Subprocess exited 0 but no completed record for seed={seed}.")
                return False
            logging.info(f"--- Run done (run_index={run_index}, seed={seed}, "
                         f"stuck={rec.get('stuck', False)}, best_val={rec.get('best_val_loss')}) ---")
            return True

        append_event({
            "status": "failed",
            "seed": seed,
            "run_index": run_index,
            "attempt": attempt,
            "returncode": result.returncode,
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        })
        if attempt > MAX_RETRIES_PER_SEED:
            logging.error(f"Subprocess failed {attempt} times for seed={seed}; giving up on it.")
            return False
        logging.warning(f"Subprocess failed (exit code {result.returncode}) for seed={seed}; "
                        f"waiting {RETRY_WAIT_S}s before retry.")
        time.sleep(RETRY_WAIT_S)


def main():
    logging.info("=== MDMM orchestrator started (no TF, random seeds, journal-driven resume) ===")

    n_completed, pending, _ = campaign_state()
    if n_completed or pending:
        logging.info(f"Resuming campaign: {n_completed} completed (non-stuck) run(s), "
                     f"{len(pending)} interrupted run(s) to resume: {pending}")

    while True:
        n_completed, pending, next_index = campaign_state()
        if n_completed >= TARGET_RUNS:
            logging.info(f"Reached TARGET_RUNS={TARGET_RUNS}. Done.")
            break

        if pending:
            seed, run_index = pending[0]
            logging.info(f"Resuming interrupted run: seed={seed}, run_index={run_index}")
        else:
            used_seeds = {e["seed"] for e in load_events() if "seed" in e}
            seed = secrets.randbits(31)
            while seed in used_seeds:  # ~1e-8 odds, but a collision would confuse the journal
                seed = secrets.randbits(31)
            run_index = next_index
            logging.info(f"Drew new random seed {seed} for run_index={run_index}")

        if not run_one(seed, run_index):
            # journal so campaign_state never returns this seed as pending again
            # (otherwise a persistently failing seed would loop forever)
            append_event({
                "status": "abandoned",
                "seed": seed,
                "run_index": run_index,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            })
            logging.error(f"Seed {seed} abandoned after repeated failures.")

    write_median()
    logging.info("=== Orchestrator finished ===")


if __name__ == "__main__":
    main()
