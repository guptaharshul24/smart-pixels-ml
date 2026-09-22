#!/usr/bin/env python3
"""
Wrapper: launches Stage 2 (non-quantized Conv2D, pixelAV-matched, no-noise
TFRs + frozen Stage-1 thresholds) and, only if it exits successfully,
launches Stage 2.5 (QConv2D) right after.

Unlike ADC_effect_training/wrapper3_submit_part2p5.py (which polls
independently-launched processes via pgrep and always launches the next
stage regardless of outcome), this wrapper launches Stage 2 itself via a
blocking subprocess.run and gates Stage 2.5 on its returncode. Both training
scripts already raise RuntimeError (nonzero exit) if they exhaust
MAX_RETRIES without a run clearing the floor gate, so returncode == 0 is an
authoritative "this stage produced a usable summary.json" signal -- no
separate polling/race-prone process-detection logic needed since this
wrapper controls both launches directly.

Runs each stage to completion (blocking) before considering the next, so
there is no chance of two orchestrators/GPU jobs racing each other -- the
exact class of bug ("mistake we did earlier") that corrupted a run's
checkpoint/log files earlier in this campaign via a duplicate orchestrator
process left alive by a silently-failed kill.

Usage: nohup python wrapper_submit_part2_part2p5.py > wrapper_submit_part2_part2p5.out 2>&1 &
"""
import os
import subprocess
import logging

HERE = os.path.dirname(os.path.abspath(__file__))

PART2_SCRIPT = os.path.join(HERE, "train_conv2d_part2_pixelav_matched_mdmm.py")
PART2P5_SCRIPT = os.path.join(HERE, "train_qconv2d_part2p5_pixelav_matched_mdmm.py")

PYTHON = "/work/users/harshul-cern/smartpixels/pixi-env/.pixi/envs/default/bin/python"
PIXI_LIB = "/work/users/harshul-cern/smartpixels/pixi-env/.pixi/envs/default/lib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "wrapper_submit_part2_part2p5.log")),
        logging.StreamHandler(),
    ],
)


def run_stage(script_path, label):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    logging.info(f"Launching {label}: {script_path}")
    result = subprocess.run([PYTHON, script_path], cwd=HERE, env=env)

    if result.returncode == 0:
        logging.info(f"{label} completed successfully (returncode 0).")
        return True
    else:
        logging.error(f"{label} FAILED (returncode {result.returncode}). Aborting chain.")
        return False


def main():
    logging.info("Wrapper started: Stage 2 (Conv2D) -> Stage 2.5 (QConv2D) chain.")

    if not run_stage(PART2_SCRIPT, "Stage 2 (Conv2D)"):
        logging.error("Stage 2 did not succeed; Stage 2.5 will NOT be launched.")
        return

    if not run_stage(PART2P5_SCRIPT, "Stage 2.5 (QConv2D)"):
        logging.error("Stage 2.5 did not succeed.")
        return

    logging.info("Wrapper done: both Stage 2 and Stage 2.5 completed successfully.")


if __name__ == "__main__":
    main()
