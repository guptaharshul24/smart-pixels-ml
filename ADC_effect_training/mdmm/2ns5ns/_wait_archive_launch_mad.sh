#!/bin/bash
# Watcher: wait for the std-1e4 run 0 subprocess (PID $1) to finish, then
# archive the std-constraint campaign as *_std1e4 and launch the MAD-constraint
# campaign via the orchestrator. Detach with: setsid nohup ... & disown
set -u
PID=$1
BASE=/home/harshul-cern/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d
HERE=/home/harshul-cern/harshul/smart-pixels-ml/ADC_effect_training/mdmm/2ns5ns
LOG=$HERE/_wait_archive_launch_mad.log

echo "$(date) waiting for PID $PID (std-1e4 run 0) to exit" >> $LOG
while kill -0 $PID 2>/dev/null; do sleep 60; done
echo "$(date) PID $PID exited; waiting 60s for JSONL flush" >> $LOG
sleep 60

# --- archive std-1e4 campaign ---
OLD=$BASE/trained_models_1_6_noise_corr_contained_2ns5ns_mdmm
NEW=$BASE/trained_models_1_6_noise_corr_contained_2ns5ns_mdmm_std1e4
if [ -d "$OLD" ]; then
    mv "$OLD" "$NEW"
    mv "$NEW/threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl" \
       "$NEW/threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm_std1e4.jsonl"
    sed -i 's|trained_models_1_6_noise_corr_contained_2ns5ns_mdmm/|trained_models_1_6_noise_corr_contained_2ns5ns_mdmm_std1e4/|g' \
       "$NEW/threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm_std1e4.jsonl"
    echo "$(date) archived campaign dir + jsonl" >> $LOG
else
    echo "$(date) WARNING: $OLD not found, skipping archive" >> $LOG
fi
for f in train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.out runLOG_rnd_thr_noise_corr_contained_2ns5ns_mdmm.txt; do
    [ -f "$HERE/$f" ] && mv "$HERE/$f" "$HERE/${f%.*}_std1e4.${f##*.}"
done
for p in run_thresholds run_losses mdmm_state; do
    [ -f "$HERE/plotting/${p}_2ns5ns_mdmm.png" ] && \
        mv "$HERE/plotting/${p}_2ns5ns_mdmm.png" "$HERE/plotting/${p}_2ns5ns_mdmm_std1e4.png"
done
echo "$(date) logs + plots archived" >> $LOG

# --- launch MAD campaign ---
cd $HERE
export LD_LIBRARY_PATH=/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}
echo "$(date) launching MAD orchestrator" >> $LOG
exec /home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python run_orchestrator_2ns5ns_mdmm.py \
    >> train_loop_rnd_thr_noise_corr_contained_2ns5ns_mdmm.out 2>&1 < /dev/null
