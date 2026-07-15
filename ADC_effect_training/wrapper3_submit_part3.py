#!/usr/bin/env python3
"""
Wrapper 3 of the Part 2/3 launch chain.

Waits for the Part 2 (ViT) training process to APPEAR (start), then waits
for it to DISAPPEAR (exit, GPU memory freed) -- not just a single "is it
running" check, which would misfire if this wrapper starts polling before
Part 2 has even launched (no process found would look identical to "already
finished"). Runs Part 3 strictly after Part 2 exits, never in parallel, to
avoid GPU OOM -- Part 2's ViT needs the same full-slice memory profile as
campaign 4, so overlapping it with even a lightweight QConv2D run risks
contention.

Once confirmed, launches Part 3 (QConv2D, frozen thresholds) as a subprocess
and waits for it to complete, then copies its summary.json into
campaign_records/ for provenance, same as Wrapper 2 does for Part 2.

No TF import here (stays a lightweight watcher) -- the training subprocess
gets the full GPU memory budget.

Designed to be launched standalone alongside Wrapper 1 and Wrapper 2 -- each
polls real, independently-observable state, so launch order doesn't matter.

Usage: nohup python wrapper3_submit_part3.py > wrapper3_submit_part3.out 2>&1 &
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

PART2_SCRIPT_BASENAME = "train_vit_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py"

PART3_SCRIPT = os.path.join(HERE, "train_qconv2d_part3_noise_corr_contained_2ns5ns_mdmm_corr1e4.py")
PART3_OUTPUT_DIR = (
    "/home/harshul-cern/work/projects/SmartPixML/"
    "dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/"
    "shuffled_3d/trained_models_1_6_noise_corr_contained_2ns5ns_mdmm/part3_qconv2d"
)
RECORDS_DEST = os.path.join(HERE, "campaign_records", "mdmm_2ns5ns", "part3_qconv2d")

PYTHON = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python"
PIXI_LIB = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "wrapper3_submit_part3.log")),
        logging.StreamHandler(),
    ],
)


def part2_running():
    result = subprocess.run(["pgrep", "-f", PART2_SCRIPT_BASENAME], capture_output=True, text=True)
    return bool(result.stdout.strip())


def sync_part3_summary():
    matches = glob.glob(os.path.join(PART3_OUTPUT_DIR, "**", "summary.json"), recursive=True)
    if not matches:
        logging.warning("No summary.json found under Part 3 output dir; nothing to sync.")
        return
    os.makedirs(RECORDS_DEST, exist_ok=True)
    for src in matches:
        dst = os.path.join(RECORDS_DEST, os.path.basename(os.path.dirname(src)) + "_summary.json")
        shutil.copy2(src, dst)
        logging.info(f"synced: {dst}")


def main():
    logging.info("Wrapper 3 started: waiting for Part 2 process to start...")
    while not part2_running():
        time.sleep(POLL_INTERVAL_S)
    logging.info("Part 2 process detected. Waiting for it to exit...")
    while part2_running():
        time.sleep(POLL_INTERVAL_S)
    logging.info("Part 2 process exited. Launching Part 3 (QConv2D) training.")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run([PYTHON, PART3_SCRIPT], cwd=HERE, env=env)
    if result.returncode != 0:
        logging.error(f"Part 3 training exited with code {result.returncode}.")
    else:
        logging.info("Part 3 training completed successfully.")

    sync_part3_summary()
    logging.info("Wrapper 3 done.")


if __name__ == "__main__":
    main()
