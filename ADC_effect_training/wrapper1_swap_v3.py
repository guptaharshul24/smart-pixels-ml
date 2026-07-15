#!/usr/bin/env python3
"""
Wrapper 1 of the Part 2/3 launch chain.

Waits for campaign 4's orchestrator process (mdmm/2ns5ns/run_orchestrator_2ns5ns_mdmm.py)
to actually exit -- polls the real OS process, not a proxy file like the median
JSON, which can be stale if a run was excluded/replaced after it was last
written. Once genuinely exited (no more possibility of a retry touching this
file), swaps the DG.OptimizedDataGenerator_v2p5 import to v3 in campaign 4's
training script.

Designed to be launched standalone and left running: safe to start anytime,
including while the orchestrator is still active or already finished.

Usage: nohup python wrapper1_swap_v3.py > wrapper1_swap_v3.out 2>&1 &
"""
import os
import time
import logging
import subprocess

ORCH_PATTERN = "run_orchestrator_2ns5ns_mdmm.py"
POLL_INTERVAL_S = 60

TARGET_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mdmm", "2ns5ns", "train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.py",
)

OLD_IMPORT = "from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator"
NEW_IMPORT = "from DG.OptimizedDataGenerator_v3 import OptimizedDataGenerator"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrapper1_swap_v3.log")),
        logging.StreamHandler(),
    ],
)


def orchestrator_running():
    result = subprocess.run(["pgrep", "-f", ORCH_PATTERN], capture_output=True, text=True)
    return bool(result.stdout.strip())


def swap_import(filepath):
    text = open(filepath).read()
    if NEW_IMPORT in text:
        logging.info(f"{filepath}: already on v3, nothing to do")
        return
    if OLD_IMPORT not in text:
        raise RuntimeError(
            f"{filepath}: expected import string not found -- file may have "
            f"changed since this wrapper was written; refusing to silently no-op"
        )
    open(filepath, "w").write(text.replace(OLD_IMPORT, NEW_IMPORT))
    logging.info(f"{filepath}: swapped v2p5 import to v3")


def main():
    logging.info("Wrapper 1 started: waiting for campaign 4 orchestrator to exit...")
    while orchestrator_running():
        time.sleep(POLL_INTERVAL_S)
    logging.info("Campaign 4 orchestrator has exited. Swapping import.")
    swap_import(TARGET_FILE)
    logging.info("Wrapper 1 done.")


if __name__ == "__main__":
    main()
