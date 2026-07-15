#!/usr/bin/env python3
"""
Wrapper 2 of the Part 2/3 launch chain.

Waits for Wrapper 1 to finish swapping campaign 4's training script to v3
(checked directly against the file's actual content, not a "Wrapper 1 is
done" signal -- correct even if this wrapper starts polling before Wrapper 1
finishes, since it just keeps checking the real file). This is deliberately
NOT gated on the median JSON's n_runs count: that file can be stale (e.g. it
may have been written before a bad run got excluded and replaced), while the
import swap only happens after Wrapper 1 confirms the orchestrator process
has genuinely exited -- by which point the median file is guaranteed final.

Once confirmed, launches Part 2 (ViT, frozen thresholds) as a subprocess and
waits for it to complete, then copies its summary.json into campaign_records/
for provenance (seed/fingerprint/thresholds actually used -- Part 2 also
draws a random seed per retry attempt, same "need a durable record" concern
as the Part 1 campaigns).

No TF import here (stays a lightweight watcher) -- the training subprocess
gets the full GPU memory budget, same rule as the campaign orchestrators.

Designed to be launched standalone alongside Wrapper 1 and Wrapper 3 -- each
polls real, independently-observable state, so launch order doesn't matter.

Usage: nohup python wrapper2_submit_part2.py > wrapper2_submit_part2.out 2>&1 &
"""
import os
import sys
import glob
import time
import shutil
import logging
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
POLL_INTERVAL_S = 60

SWAP_TARGET = os.path.join(HERE, "mdmm", "2ns5ns", "train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.py")
NEW_IMPORT = "from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator"

PART2_SCRIPT = os.path.join(HERE, "train_vit_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py")
PART2_OUTPUT_DIR = (
    "/home/harshul-cern/work/projects/SmartPixML/"
    "dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/"
    "shuffled_3d/trained_models_1_6_noise_corr_contained_2ns5ns_mdmm/part2_vit"
)
RECORDS_DEST = os.path.join(HERE, "campaign_records", "mdmm_2ns5ns", "part2_vit")

PYTHON = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python"
PIXI_LIB = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "wrapper2_submit_part2.log")),
        logging.StreamHandler(),
    ],
)


def wrapper1_swap_done():
    if not os.path.exists(SWAP_TARGET):
        return False
    return NEW_IMPORT in open(SWAP_TARGET).read()


def sync_part2_summary():
    matches = glob.glob(os.path.join(PART2_OUTPUT_DIR, "**", "summary.json"), recursive=True)
    if not matches:
        logging.warning("No summary.json found under Part 2 output dir; nothing to sync.")
        return
    os.makedirs(RECORDS_DEST, exist_ok=True)
    for src in matches:
        dst = os.path.join(RECORDS_DEST, os.path.basename(os.path.dirname(src)) + "_summary.json")
        shutil.copy2(src, dst)
        logging.info(f"synced: {dst}")


def main():
    logging.info("Wrapper 2 started: waiting for Wrapper 1's v3 import swap...")
    while not wrapper1_swap_done():
        time.sleep(POLL_INTERVAL_S)
    logging.info("Wrapper 1's swap confirmed. Launching Part 2 (ViT) training.")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run([PYTHON, PART2_SCRIPT], cwd=HERE, env=env)
    if result.returncode != 0:
        logging.error(f"Part 2 training exited with code {result.returncode}.")
    else:
        logging.info("Part 2 training completed successfully.")

    sync_part2_summary()
    logging.info("Wrapper 2 done.")


if __name__ == "__main__":
    main()
