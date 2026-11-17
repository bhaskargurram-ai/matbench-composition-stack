"""Parallel scheduler for the Amendment-4 run on this PC (scheduling only; method and results unchanged).

One job per (task, fold): `python -m src.run_final4 select <task> --fold k` (all expert fits of both candidate block
sets, candidate scoring, selection, final predictions). At most --workers jobs at a time, each pinned to 4 threads
(experts use n_jobs=4; BLAS/OpenMP capped at 4), so results do not depend on how many workers run concurrently.
Resumable: finished folds are skipped and finished experts are cached. Wall-clock time per job is appended to
results/final_a4/timing.jsonl. On Windows the process asks the OS not to sleep while it runs.

Usage: python -m src.launch_final4 [<task> ...] [--workers 5] [--dry-run]
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from .data import ROOT, TASKS

MAX_WORKERS = 5      # 5 x 4 = 20 threads at most on this PC
THREADS = "4"
RES = ROOT / "results" / "final_a4"
ORDER = ["matbench_expt_gap", "matbench_glass", "matbench_expt_is_metal", "matbench_steels"]   # longest first


def main():
    a = sys.argv[1:]
    workers = int(a.pop(a.index("--workers") + 1)) if "--workers" in a else MAX_WORKERS
    if "--workers" in a:
        a.remove("--workers")
    dry = "--dry-run" in a
    tasks = [t for t in a if not t.startswith("--")] or ORDER
    assert set(tasks) <= set(TASKS)
    todo = [(t, k, ["select", t, "--fold", str(k)], f"select_{t}_fold{k}") for t in tasks for k in range(5)
            if not (RES / t / f"fold{k}_predictions.npz").exists()]
    print(f"{len(todo)} jobs; workers={workers}; threads/worker={THREADS}", flush=True)
    if dry:
        print([j[3] for j in todo]); return
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
               NUMEXPR_NUM_THREADS=THREADS, PYTHONIOENCODING="utf-8", A3_N_WORKERS=str(workers))
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while todo or running:
        while todo and len(running) < workers:
            t, k, cmd, name = todo.pop(0)
            log = open(RES / "logs" / f"{name}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "src.run_final4", *cmd], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (t, k, name, time.time(), log)
            print(f"start {name} (pid {p.pid})", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                t, k, name, t0, log = running.pop(p)
                log.close()
                rec = {"job": name, "task": t, "fold": k, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1),
                       "workers": workers, "threads_per_worker": int(THREADS), "host": "pc",
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                with open(RES / "timing.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"done {name}: rc={p.returncode} {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
