#!/usr/bin/env python3
"""
One-stop status + plotting for the MDMM correlation-constraint (corr1e4) campaigns.

Usage:
    /home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python status_and_plot.py [1ns6ns|2ns5ns|all] [--no-eval|--all-runs|--incremental] [--part1] [--part1p5] [--part2p5]

Default (no arg) = all. Read-only on the training processes; only writes new
PNGs into plotting/corr1e4/ (Part 1) or plotting/part1p5/ or plotting/part2p5/ (Part 1.5/2.5).
Safe to run anytime, including mid-training.

--no-eval: campaign-level plots only (thresholds/losses/mdmm-state), skip
  per-fingerprint eval entirely (fastest).
--all-runs: force-evaluate every run with a checkpoint, every time (slowest,
  no caching).
--incremental: evaluate every run with a checkpoint (across Part 1, Part 1.5,
  and Part 2.5 alike), but skip ones that are already completed and already
  have a predictions.csv. Cheap plots (loss/threshold/mdmm-state) always
  regenerate regardless of this flag -- only the expensive per-fingerprint
  eval is cached.
(no flag): only evaluate the single best-completed/latest-in-progress run,
  per part.

--part1 / --part1p5 / --part2p5: restrict to one or more parts (combine freely,
  e.g. --part1p5 --part2p5 to skip Part 1 entirely). No part flag given = all
  three (default, backward-compatible).

For each case, prints:
  - whether the orchestrator/subprocess are alive
  - the campaign journal (started/completed/failed/abandoned per run)
  - for each run with data: current epoch, best_val_loss so far, latest
    lambda/correlation/std per parameter (from mdmm_state_log.csv)
Then regenerates plot_thresholds / plot_run_losses / plot_mdmm_state under
plotting/corr1e4/<case>/ for whichever cases have any campaign data.

--all-runs force-evaluates every fingerprint with a checkpoint, every time
(no caching) -- use this if eval_transformer itself changed or you want a
guaranteed-fresh regen. --incremental evaluates the same set of fingerprints
but skips any that are already "completed" in the journal AND already have a
predictions.csv (checkpoint is final, won't change). Still-running
fingerprints and completed-but-never-evaluated fingerprints are always
(re-)evaluated under --incremental too -- it never checks the stuck flag, so
a newly-stuck-but-unevaluated run still gets its diagnostic plots made once.

Only covers the CURRENT (corr1e4) constraint generation -- archived campaigns
(archive_scale1/, archive_std1e4/, archive_mad1e4/) are static and don't need
re-plotting; use their own scripts directly if ever needed.
"""
import os
import sys
import csv
import json
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from sync_campaign_records import sync_campaign_records

PYTHON = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/bin/python"
PIXI_LIB = "/home/harshul-cern/work/pixi/global/.pixi/envs/default/lib"
MDMM_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = ("/home/harshul-cern/work/projects/SmartPixML/"
                 "dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d")

CASES = {
    "1ns6ns": {
        "trained_models_dir": os.path.join(DATASET_BASE, "trained_models_1_6_noise_corr_contained_mdmm"),
        "jsonl_name": "threshold_runs_rnd_thr_noise_corr_contained_mdmm.jsonl",
        "proc_pattern": "1ns6ns_mdmm",
        "records_dest": "mdmm_1ns6ns/corr1e4",
    },
    "2ns5ns": {
        "trained_models_dir": os.path.join(DATASET_BASE, "trained_models_1_6_noise_corr_contained_2ns5ns_mdmm"),
        "jsonl_name": "threshold_runs_rnd_thr_noise_corr_contained_2ns5ns_mdmm.jsonl",
        "proc_pattern": "2ns5ns_mdmm",
        "records_dest": "mdmm_2ns5ns/corr1e4",
    },
}

ADC_ROOT = os.path.dirname(MDMM_ROOT)

# Part 1.5 (ViT, frozen thresholds) / Part 2.5 (QConv2D, frozen thresholds) -- both
# trained on campaign 4's median, both single-run-with-retry (no multi-run
# journal/median the way Part 1 campaigns have). Only 2ns5ns exists so far.
PART1P5_2P5_CASES = {
    "2ns5ns": {
        "part1p5_script": "train_vit_part1p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py",
        "part2p5_script": "train_qconv2d_part2p5_noise_corr_contained_2ns5ns_mdmm_corr1e4.py",
        "part1p5_output_dir": os.path.join(
            DATASET_BASE, "trained_models_1_6_noise_corr_contained_2ns5ns_mdmm", "part1p5_vit"),
        "part2p5_output_dir": os.path.join(
            DATASET_BASE, "trained_models_1_6_noise_corr_contained_2ns5ns_mdmm", "part2p5_qconv2d"),
        "part1p5_ckpt_prefix": "Transformer_model",
        "part2p5_ckpt_prefix": "QConv2D_model",
        "part1p5_plot_dir": os.path.join(ADC_ROOT, "plotting", "part1p5"),
        "part2p5_plot_dir": os.path.join(ADC_ROOT, "plotting", "part2p5"),
    },
}


def load_events(jsonl_path):
    if not os.path.exists(jsonl_path):
        return []
    return [json.loads(l) for l in open(jsonl_path) if l.strip()]


def find_processes(pattern):
    try:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout
    except FileNotFoundError:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        out = "\n".join(l for l in out.splitlines() if pattern in l and "grep" not in l)
    lines = [l for l in out.splitlines() if l.strip()]
    # drop the shell-snapshot wrapper line the launching shell leaves behind
    return [l for l in lines if "shell-snapshots" not in l]


def latest_row(csv_path):
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    return rows[-1] if rows else None


def print_status(case_name, cfg):
    print(f"\n{'='*70}\n{case_name}\n{'='*70}")

    procs = find_processes(cfg["proc_pattern"])
    if procs:
        print(f"RUNNING ({len(procs)} process(es)):")
        for p in procs:
            print(f"  {p[:220]}")
    else:
        print("No processes running.")

    jsonl_path = os.path.join(cfg["trained_models_dir"], cfg["jsonl_name"])
    events = load_events(jsonl_path)
    if not events:
        print("No campaign journal found yet.")
        return

    completed = [e for e in events if e.get("status", "completed") == "completed"]
    n_ok = sum(1 for e in completed if not e.get("stuck", False))
    print(f"Journal: {n_ok} completed non-stuck run(s) (target 5)")

    # per-seed latest status line, in journal order, de-duplicated by seed
    seen = set()
    for e in events:
        seed = e.get("seed")
        if seed is None or seed in seen:
            continue
        seen.add(seed)
        status = e.get("status", "completed")
        fp = e.get("fingerprint", "?")
        line = f"  seed={seed} fp={fp} status={status}"
        if status == "completed":
            line += f" stuck={e.get('stuck')} best_val={e.get('best_val_loss'):.0f} epochs={e.get('epochs_run')}"
        print(line)

        # for started/completed runs, look up live epoch/loss/corr detail
        ckpt_dirs = glob.glob(os.path.join(
            cfg["trained_models_dir"], "2t_*", f"Transformer_model-{fp}-checkpoints"))
        if not ckpt_dirs:
            continue
        d = ckpt_dirs[0]
        tlog = latest_row_all(os.path.join(d, "training_log.csv"))
        mlog = latest_row(os.path.join(d, "mdmm_state_log.csv"))
        if tlog:
            n_epochs, best_val, last_val = tlog
            print(f"    epoch {n_epochs}, best_val={best_val:.0f}, last_val={last_val:.0f}")
        if mlog:
            corrs = {k.replace("pred_corr_", ""): round(float(v), 3)
                     for k, v in mlog.items() if k.startswith("pred_corr")}
            lmbdas = {k.replace("lmbda_corr_", ""): round(float(v), 3)
                      for k, v in mlog.items() if k.startswith("lmbda_corr")}
            print(f"    corr={corrs}")
            print(f"    lambda={lmbdas}")


def latest_row_all(csv_path):
    """Returns (n_epochs, best_val_loss, last_val_loss) or None."""
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    vals = [float(r["val_loss"]) for r in rows]
    return len(rows), min(vals), vals[-1]


def eval_one(case_name, cfg, plot_dir, env, fingerprint=None, eval_script_glob="eval_transformer_*.py",
             proc_pattern=None):
    """Run eval_<...> (+ pred_angle_dists on top of it) for one run.
    fingerprint=None lets the eval script auto-select (best completed, else
    latest in-progress). Returns the fingerprint actually evaluated, or None."""
    proc_pattern = proc_pattern or cfg.get("proc_pattern")
    eval_script = glob.glob(os.path.join(plot_dir, eval_script_glob))
    if not eval_script:
        print(f"  [{case_name}] no eval script matching '{eval_script_glob}', skipping money/pull/pred-angle plots")
        return None
    eval_script = eval_script[0]
    # CPU if the GPU is busy training (a live run holds the MIG slice); the
    # eval only needs inference on 8 val batches so CPU is tolerable.
    eval_env = dict(env)
    if proc_pattern and find_processes(proc_pattern):
        eval_env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [PYTHON, os.path.basename(eval_script)]
    if fingerprint:
        cmd += ["--fingerprint", fingerprint]
    result = subprocess.run(cmd, cwd=plot_dir, env=eval_env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{case_name}] {os.path.basename(eval_script)} ({fingerprint or 'auto'}): FAILED\n{result.stderr.strip()[-500:]}")
        return None
    # "Evaluating run <fingerprint> (...)" isn't necessarily the first stdout
    # line -- utils.check_GPU() prints "No GPU(s)"/"N Physical GPUs..." first,
    # so search all lines rather than assuming position.
    eval_line = next((l for l in result.stdout.splitlines() if l.startswith("Evaluating run ")), "")
    print(f"  [{case_name}] {os.path.basename(eval_script)}: {eval_line or 'ok (no eval-run line found)'}")
    fp = eval_line.split()[2] if eval_line else fingerprint
    if not fp:
        print(f"  [{case_name}] could not parse fingerprint from eval output, skipping pred-angle plot")
        return None

    pred_angle_script = glob.glob(os.path.join(plot_dir, "plot_pred_angle_dists_*.py"))
    if not pred_angle_script:
        print(f"  [{case_name}] no plot_pred_angle_dists script, skipping")
        return fp
    result = subprocess.run([PYTHON, os.path.basename(pred_angle_script[0]), "--fingerprint", fp],
                             cwd=plot_dir, env=env,
                             capture_output=True, text=True)
    tag = os.path.basename(pred_angle_script[0])
    if result.returncode == 0:
        print(f"  [{case_name}] {tag}: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'ok'}")
    else:
        print(f"  [{case_name}] {tag}: FAILED\n{result.stderr.strip()[-500:]}")
    return fp


def all_fingerprints_with_checkpoints(cfg):
    """Every unique fingerprint in the journal that actually has a checkpoint
    dir (i.e. training got far enough to matter -- excludes pure 'started'
    events from crashed/abandoned attempts with zero epochs)."""
    events = load_events(os.path.join(cfg["trained_models_dir"], cfg["jsonl_name"]))
    seen, fps = set(), []
    for e in events:
        fp = e.get("fingerprint")
        if not fp or fp in seen:
            continue
        if glob.glob(os.path.join(cfg["trained_models_dir"], "2t_*", f"Transformer_model-{fp}-checkpoints")):
            seen.add(fp)
            fps.append(fp)
    return fps


def fingerprint_completed(cfg, fp):
    """True if this fingerprint has a 'completed' event in the journal --
    i.e. training is done and its checkpoint/best-weights are final and
    won't change on a future eval pass."""
    events = load_events(os.path.join(cfg["trained_models_dir"], cfg["jsonl_name"]))
    return any(e.get("fingerprint") == fp and e.get("status", "completed") == "completed"
               for e in events)


def regenerate_plots(case_name, cfg, run_eval=True, all_runs=False, incremental=False):
    plot_dir = os.path.join(MDMM_ROOT, case_name, "plotting", "corr1e4")
    if not os.path.isdir(plot_dir):
        print(f"  [{case_name}] no plotting/corr1e4/ dir, skipping plots")
        return
    if not os.path.exists(cfg["trained_models_dir"]) or not load_events(
            os.path.join(cfg["trained_models_dir"], cfg["jsonl_name"])):
        print(f"  [{case_name}] no campaign data yet, skipping plots")
        return

    synced = sync_campaign_records(cfg["trained_models_dir"], cfg["records_dest"])
    if synced:
        print(f"  [{case_name}] synced {len(synced)} campaign record file(s) into campaign_records/")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    # campaign-level plots: thresholds / losses / mdmm state (cheap, CSV-only)
    scripts = sorted(glob.glob(os.path.join(plot_dir, "plot_mdmm_state_*.py")) +
                     glob.glob(os.path.join(plot_dir, "plot_run_losses_*.py")) +
                     glob.glob(os.path.join(plot_dir, "plot_thresholds_*.py")))
    for script in scripts:
        result = subprocess.run([PYTHON, os.path.basename(script)],
                                 cwd=plot_dir, env=env,
                                 capture_output=True, text=True)
        tag = os.path.basename(script)
        if result.returncode == 0:
            print(f"  [{case_name}] {tag}: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'ok'}")
        else:
            print(f"  [{case_name}] {tag}: FAILED\n{result.stderr.strip()[-500:]}")

    if not run_eval:
        return

    # fingerprint-level eval: money/pull/residual/sigma plots + predictions.csv,
    # then the pred-angle-distribution plot on top of those predictions. Heavier
    # (loads a checkpoint, runs inference on the full val set) -- skip with
    # run_eval=False if only the cheap campaign plots are wanted.
    if all_runs or incremental:
        fps = all_fingerprints_with_checkpoints(cfg)
        if not fps:
            print(f"  [{case_name}] no runs with checkpoints yet, skipping eval")
            return
        if incremental:
            to_eval = []
            for fp in fps:
                already_done = (fingerprint_completed(cfg, fp) and
                                 os.path.exists(os.path.join(plot_dir, fp, "predictions.csv")))
                if not already_done:
                    to_eval.append(fp)
            skipped = len(fps) - len(to_eval)
            if skipped:
                print(f"  [{case_name}] skipping {skipped} already-evaluated completed run(s)")
            if not to_eval:
                print(f"  [{case_name}] nothing new to evaluate")
                return
        else:
            to_eval = fps
        print(f"  [{case_name}] evaluating {len(to_eval)} run(s): {to_eval}")
        for fp in to_eval:
            eval_one(case_name, cfg, plot_dir, env, fingerprint=fp)
    else:
        eval_one(case_name, cfg, plot_dir, env, fingerprint=None)


# ============================================================================
# Part 1.5 (ViT, frozen thresholds) / Part 2.5 (QConv2D, frozen thresholds)
# ============================================================================

def part1p5_3_ckpt_dirs(output_dir, ckpt_prefix):
    return sorted(
        glob.glob(os.path.join(output_dir, "**", f"{ckpt_prefix}-*-checkpoints"), recursive=True),
        key=os.path.getctime)


def all_fingerprints_with_checkpoints_part1p5_3(output_dir, ckpt_prefix):
    return [os.path.basename(d).split("-")[1] for d in part1p5_3_ckpt_dirs(output_dir, ckpt_prefix)]


def fingerprint_completed_part1p5_3(output_dir, ckpt_prefix, fp):
    dirs = glob.glob(os.path.join(output_dir, "**", f"{ckpt_prefix}-{fp}-checkpoints"), recursive=True)
    return bool(dirs) and os.path.exists(os.path.join(dirs[0], "summary.json"))


def print_part1p5_2p5_status(case_name, cfg, do_part1p5=True, do_part2p5=True):
    print(f"\n{'-'*70}\n{case_name} -- Part 1.5/2.5\n{'-'*70}")
    rows = [("Part 1.5 (ViT)", "part1p5_script", "part1p5_ckpt_prefix", "part1p5_output_dir", do_part1p5),
            ("Part 2.5 (QConv2D)", "part2p5_script", "part2p5_ckpt_prefix", "part2p5_output_dir", do_part2p5)]
    for label, script_key, prefix_key, out_key, enabled in rows:
        if not enabled:
            continue
        procs = find_processes(cfg[script_key])
        print(f"{label}: {'RUNNING' if procs else 'not running'}")
        for p in procs:
            print(f"  {p[:220]}")

        ckpt_dirs = part1p5_3_ckpt_dirs(cfg[out_key], cfg[prefix_key])
        if not ckpt_dirs:
            print("  no runs yet")
            continue
        for d in ckpt_dirs:
            fp = os.path.basename(d).split("-")[1]
            tlog = latest_row_all(os.path.join(d, "training_log.csv"))
            status = "completed" if os.path.exists(os.path.join(d, "summary.json")) else "in-progress"
            if tlog:
                n_epochs, best_val, last_val = tlog
                print(f"  {fp} ({status}): epoch {n_epochs}, best_val={best_val:.0f}, last_val={last_val:.0f}")
            else:
                print(f"  {fp} ({status}): no training_log.csv yet")


def regenerate_part1p5_2p5_plots(case_name, cfg, run_eval=True, all_runs=False, incremental=False,
                             do_part1p5=True, do_part2p5=True):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = PIXI_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    rows = [("part1p5", "part1p5_script", "part1p5_ckpt_prefix", "part1p5_output_dir", "part1p5_plot_dir",
              "eval_part1p5_*.py", do_part1p5),
            ("part2p5", "part2p5_script", "part2p5_ckpt_prefix", "part2p5_output_dir", "part2p5_plot_dir",
              "eval_part2p5_*.py", do_part2p5)]
    for part, script_key, prefix_key, out_key, plot_key, eval_glob, enabled in rows:
        if not enabled:
            continue
        plot_dir = cfg[plot_key]
        if not os.path.isdir(plot_dir):
            continue
        if not part1p5_3_ckpt_dirs(cfg[out_key], cfg[prefix_key]):
            print(f"  [{case_name}/{part}] no runs yet, skipping plots")
            continue

        # cheap campaign-level plots: losses + mdmm state (CSV-only)
        scripts = sorted(glob.glob(os.path.join(plot_dir, "plot_run_losses_*.py")) +
                         glob.glob(os.path.join(plot_dir, "plot_mdmm_state_*.py")))
        for script in scripts:
            result = subprocess.run([PYTHON, os.path.basename(script)],
                                     cwd=plot_dir, env=env, capture_output=True, text=True)
            tag = os.path.basename(script)
            if result.returncode == 0:
                out_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "ok"
                print(f"  [{case_name}/{part}] {tag}: {out_line}")
            else:
                print(f"  [{case_name}/{part}] {tag}: FAILED\n{result.stderr.strip()[-500:]}")

        if not run_eval:
            continue

        proc_pattern = cfg[script_key]
        if all_runs or incremental:
            fps = all_fingerprints_with_checkpoints_part1p5_3(cfg[out_key], cfg[prefix_key])
            if not fps:
                continue
            if incremental:
                to_eval = []
                for fp in fps:
                    already_done = (fingerprint_completed_part1p5_3(cfg[out_key], cfg[prefix_key], fp) and
                                     os.path.exists(os.path.join(plot_dir, fp, "predictions.csv")))
                    if not already_done:
                        to_eval.append(fp)
                skipped = len(fps) - len(to_eval)
                if skipped:
                    print(f"  [{case_name}/{part}] skipping {skipped} already-evaluated run(s)")
                if not to_eval:
                    print(f"  [{case_name}/{part}] nothing new to evaluate")
                    continue
            else:
                to_eval = fps
            print(f"  [{case_name}/{part}] evaluating {len(to_eval)} run(s): {to_eval}")
            for fp in to_eval:
                eval_one(case_name, cfg, plot_dir, env, fingerprint=fp,
                         eval_script_glob=eval_glob, proc_pattern=proc_pattern)
        else:
            eval_one(case_name, cfg, plot_dir, env, fingerprint=None,
                     eval_script_glob=eval_glob, proc_pattern=proc_pattern)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_eval = "--no-eval" not in sys.argv[1:]
    all_runs = "--all-runs" in sys.argv[1:]
    incremental = "--incremental" in sys.argv[1:]
    if all_runs and incremental:
        print("--all-runs and --incremental together mean the same runs; using --incremental (skips already-evaluated).")
        all_runs = False
    if (all_runs or incremental) and not run_eval:
        print("--no-eval means nothing to evaluate; ignoring --all-runs/--incremental.")
        all_runs = incremental = False

    # --part1/--part1p5/--part2p5: select which part(s) to cover. None given = all
    # three (backward-compatible default).
    part_flags = {"--part1", "--part1p5", "--part2p5"} & set(sys.argv[1:])
    do_part1 = not part_flags or "--part1" in part_flags
    do_part1p5 = not part_flags or "--part1p5" in part_flags
    do_part2p5 = not part_flags or "--part2p5" in part_flags

    which = args[0] if args else "all"
    cases = CASES.keys() if which == "all" else [which]

    for case_name in cases:
        if case_name not in CASES:
            print(f"Unknown case '{case_name}'; choices: {list(CASES)} or 'all'")
            continue
        if do_part1:
            print_status(case_name, CASES[case_name])
        if case_name in PART1P5_2P5_CASES and (do_part1p5 or do_part2p5):
            print_part1p5_2p5_status(case_name, PART1P5_2P5_CASES[case_name], do_part1p5=do_part1p5, do_part2p5=do_part2p5)

    if not run_eval:
        mode = " (campaign-level only, --no-eval)"
    elif all_runs:
        mode = " (ALL runs, --all-runs, force re-eval)"
    elif incremental:
        mode = " (incremental, --incremental, skip already-evaluated)"
    else:
        mode = ""
    print(f"\n{'='*70}\nRegenerating plots{mode}\n{'='*70}")
    for case_name in cases:
        if case_name not in CASES:
            continue
        if do_part1:
            regenerate_plots(case_name, CASES[case_name], run_eval=run_eval, all_runs=all_runs, incremental=incremental)
        if case_name in PART1P5_2P5_CASES and (do_part1p5 or do_part2p5):
            regenerate_part1p5_2p5_plots(case_name, PART1P5_2P5_CASES[case_name],
                                     run_eval=run_eval, all_runs=all_runs, incremental=incremental,
                                     do_part1p5=do_part1p5, do_part2p5=do_part2p5)


if __name__ == "__main__":
    main()
