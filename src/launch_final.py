"""Parallel scheduler for the final official-fold run (scheduling only; method and results unchanged).

Runs `python -m src.run_final <task> --fold <k>` for every (task, fold) whose predictions file is
missing, as separate processes, at most MAX_WORKERS at a time. Each worker is pinned to a fixed
thread count (experts use n_jobs=4; BLAS/OpenMP capped at 4 threads), so results do not depend on
how many workers run concurrently. Resumable: finished folds are skipped. Wall-clock time per job is
appended to results/final/timing.jsonl.

Usage: python -m src.launch_final [--workers 5] [--dry-run]
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from .data import ROOT, TASKS

MAX_WORKERS = 5
THREADS = "4"
ORDER = ["matbench_glass", "matbench_expt_is_metal", "matbench_expt_gap", "matbench_steels"]  # largest first


def jobs():
    out = []
    for t in ORDER:
        for k in range(5):
            if not (ROOT / "results" / "final" / t / f"fold{k}_predictions.npz").exists():
                out.append((t, k))
    return out


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else MAX_WORKERS
    todo = jobs()
    print(f"{len(todo)} jobs; workers={workers}; threads/worker={THREADS}", flush=True)
    if "--dry-run" in args:
        print(todo); return
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
               NUMEXPR_NUM_THREADS=THREADS, PYTHONIOENCODING="utf-8")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results" / "final").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < workers:
            t, k = todo.pop(0)
            log = open(ROOT / "logs" / f"final_{t}_fold{k}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "src.run_final", t, "--fold", str(k)], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (t, k, time.time(), log)
            print(f"start {t} fold {k} (pid {p.pid})", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                t, k, t0, log = running.pop(p)
                log.close()
                rec = {"task": t, "fold": k, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1),
                       "workers": workers, "threads_per_worker": int(THREADS)}
                with open(ROOT / "results" / "final" / "timing.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"done {t} fold {k}: rc={p.returncode} {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
