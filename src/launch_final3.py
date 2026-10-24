"""Parallel scheduler for the Amendment-3 run on this PC (scheduling only; method and results unchanged).

Two kinds of jobs, run as separate processes, at most --workers at a time, each pinned to 4 threads
(experts use n_jobs=4; BLAS/OpenMP capped at 4), so results do not depend on how many workers run concurrently:
  fit    : `python -m src.run_final3 fit <task> --fold k --cand <bs>`  (expert cross-fits of one candidate)
  select : `python -m src.run_final3 select <task> --fold k [--drop-cand F_min]`  (candidate scoring, selection,
           final predictions; fits whatever cache is still missing)
Resumable: finished folds are skipped. Wall-clock time per job is appended to results/final_a3/timing.jsonl.

Usage:
  python -m src.launch_final3 fit <task> [<task> ...] --cands F_red[,F_min] [--workers 5]
  python -m src.launch_final3 select <task> [<task> ...] [--drop-cand F_min] [--workers 5] [--dry-run]
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from .data import ROOT
from .frozen_config3 import CAND_BLOCKS

MAX_WORKERS = 5      # 5 x 4 = 20 threads at most on this PC
THREADS = "4"
RES = ROOT / "results" / "final_a3"


def main():
    a = sys.argv[1:]
    mode = a.pop(0)
    workers = int(a.pop(a.index("--workers") + 1)) if "--workers" in a else MAX_WORKERS
    if "--workers" in a:
        a.remove("--workers")
    cands = a.pop(a.index("--cands") + 1).split(",") if "--cands" in a else []
    if "--cands" in a:
        a.remove("--cands")
    drops = []
    while "--drop-cand" in a:
        i = a.index("--drop-cand"); drops.append(a[i + 1]); del a[i:i + 2]
    dry = "--dry-run" in a
    tasks = [t for t in a if not t.startswith("--")]
    todo = []
    for t in tasks:
        for k in range(5):
            if (RES / t / f"fold{k}_predictions.npz").exists():
                continue
            if mode == "fit":
                for bs in cands:
                    assert bs in CAND_BLOCKS[t], (t, bs)
                    todo.append((t, k, ["fit", t, "--fold", str(k), "--cand", bs], f"fit_{t}_fold{k}_{bs}"))
            else:
                cmd = ["select", t, "--fold", str(k)]
                for d in drops:
                    cmd += ["--drop-cand", d]
                todo.append((t, k, cmd, f"select_{t}_fold{k}"))
    print(f"{len(todo)} jobs; workers={workers}; threads/worker={THREADS}", flush=True)
    if dry:
        print([j[3] for j in todo]); return
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
               NUMEXPR_NUM_THREADS=THREADS, PYTHONIOENCODING="utf-8", A3_N_WORKERS=str(workers))
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < workers:
            t, k, cmd, name = todo.pop(0)
            log = open(RES / "logs" / f"{name}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "src.run_final3", *cmd], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (t, k, name, time.time(), log)
            print(f"start {name} (pid {p.pid})", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                t, k, name, t0, log = running.pop(p)
                log.close()
                rec = {"job": name, "task": t, "fold": k, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1),
                       "workers": workers, "threads_per_worker": int(THREADS), "host": "pc"}
                with open(RES / "timing.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"done {name}: rc={p.returncode} {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
