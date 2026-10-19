"""Parallel scheduler for the post-hoc ablation (scheduling only). 5 workers x 4 threads.
Usage: python -m src.launch_ablation"""
import json, os, subprocess, sys, time
from .data import ROOT

JOBS = [("nohurdle", "matbench_expt_gap", k) for k in range(5)] + \
       [("nodft", t, k) for t in ("matbench_expt_gap", "matbench_expt_is_metal", "matbench_glass", "matbench_steels")
        for k in range(5)]


def main():
    todo = [j for j in JOBS if not (ROOT / "results" / f"ablation_{j[0]}" / j[1] / f"fold{j[2]}_predictions.npz").exists()]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
    running = {}
    print(len(todo), "jobs", flush=True)
    while todo or running:
        while todo and len(running) < 5:
            v, t, k = todo.pop(0)
            log = open(ROOT / "logs" / f"ablation_{v}_{t}_fold{k}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "src.run_ablation", v, t, "--fold", str(k)], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (v, t, k, time.time(), log)
            print("start", v, t, k, flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                v, t, k, t0, log = running.pop(p); log.close()
                rec = {"variant": v, "task": t, "fold": k, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1)}
                (ROOT / "results" / f"ablation_{v}").mkdir(parents=True, exist_ok=True)
                with open(ROOT / "results" / f"ablation_{v}" / "timing.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("done", rec, flush=True)


if __name__ == "__main__":
    main()
