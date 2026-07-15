#!/bin/bash
# Waits for the current train_loop.py (PID 1822) to finish, then launches
# train_loop_rnd_thr.py the same way (detached, survives logout).
TARGET_PID=1822

while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 300
done

cd /home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training || exit 1
export LD_LIBRARY_PATH=/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib:$LD_LIBRARY_PATH
exec /home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python train_loop_rnd_thr.py > train_loop_rnd_thr.out 2>&1
