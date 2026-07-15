#!/usr/bin/env python3
"""
Runs Part 2.5 (QConv2D) directly as a subprocess; if it exhausts all
MAX_RETRIES attempts without clearing the val_loss floor (exits non-zero,
raises RuntimeError), automatically launches Part 2 (plain Conv2D) as a
fallback/diagnostic per das's suggestion -- quantization may be the cause of
QConv2D's repeated stuck-at-init failures, and training the plain twin
isolates that (same architecture shape, same MDMM/dataset/thresholds, no
QKeras quantizers).

Unlike wrapper2/wrapper3 (which poll for an externally-launched process to
appear then disappear -- built that way because launch order between wrappers
wasn't guaranteed), this wrapper launches Part 2.5 itself directly via
subprocess.run and reads its real exit code. No race condition, no need to
infer "not started yet" vs "already finished" from process-list absence.

Usage: nohup python wrapper4_part2p5_then_fallback_part2.py > wrapper4_part2p5_then_fallback_part2.out 2>&1 &
"""
import os
import subprocess
import logging

HERE = os.path.dirname(os.path.abspath(__file__))

PART2P5_SCRIPT = os.path.join(HERE, "train_qconv2d_part2p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py")
PART2_SCRIPT = os.path.join(HERE, "train_conv2d_part2_noise_corr_contained_2ns5ns_mdmm_corr1e4.py")

PYTHON = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python"
PIXI_LIB = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "wrapper4_part2p5_then_fallback_part2.log")),
        logging.StreamHandler(),
    ],
)


def main():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    logging.info("Launching Part 2.5 (QConv2D)...")
    result = subprocess.run([PYTHON, PART2P5_SCRIPT], cwd=HERE, env=env)

    if result.returncode == 0:
        logging.info("Part 2.5 succeeded (cleared the val_loss floor). No fallback needed.")
        return

    logging.info(f"Part 2.5 exited with code {result.returncode} (failed to clear the floor "
                 f"after all retries). Launching Part 2 (plain Conv2D) fallback per das's "
                 f"quantization-hypothesis diagnostic.")
    result2 = subprocess.run([PYTHON, PART2_SCRIPT], cwd=HERE, env=env)
    if result2.returncode == 0:
        logging.info("Part 2 (plain Conv2D) succeeded.")
    else:
        logging.error(f"Part 2 (plain Conv2D) also failed, exit code {result2.returncode}.")


if __name__ == "__main__":
    main()
