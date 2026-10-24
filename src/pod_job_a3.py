"""Amendment-3 expert-fit runner for a RunPod CPU pod (scheduling only; the fits are src.run_final3 `fit` jobs).

The pod receives a copy of src/, the feature caches (data/cache/*.pkl, *.json) and the Amendment-2 expert caches
(results/final_a2/<task>/fold<k>_experts/). Each job is `python -m src.run_final3 fit <task> --fold k --cand <bs>`
(cache reuse by identical inputs) or, for the determinism check, `... fit <task> --fold k --cand F_full --no-reuse
--tag determinism` (every expert re-fitted from scratch). Jobs run as separate processes, --workers at a time,
each pinned to 4 threads (the experts' n_jobs), longest jobs first. Timing goes to results/final_a3/pod_timing.jsonl.

Usage (on the pod, cwd = project root):
  python -m src.pod_job_a3 --workers 8 matbench_expt_gap:0:F_red matbench_expt_gap:1:F_red ... det:matbench_steels:0
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "final_a3"
THREADS = "4"


def main():
    a = sys.argv[1:]
    workers = int(a.pop(a.index("--workers") + 1)) if "--workers" in a else 8
    if "--workers" in a:
        a.remove("--workers")
    todo = []
    for spec in a:
        parts = spec.split(":")
        if parts[0] == "det":
            _, task, k = parts
            todo.append((spec, ["fit", task, "--fold", k, "--cand", "F_full", "--no-reuse", "--tag", "determinism"]))
        else:
            task, k, bs = parts
            todo.append((spec, ["fit", task, "--fold", k, "--cand", bs]))
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS,
               NUMEXPR_NUM_THREADS=THREADS, NUMEXPR_MAX_THREADS=THREADS, LOKY_MAX_CPU_COUNT=THREADS,
               PYTHONIOENCODING="utf-8", A3_N_WORKERS=str(workers))
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.lower().startswith("model name")][0]
    except Exception:
        cpu = "?"
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count()
    print(f"{len(todo)} jobs; workers={workers}; threads/worker={THREADS}; vCPU={ncpu}; cpu={cpu}", flush=True)
    running, rc_all = {}, 0
    while todo or running:
        while todo and len(running) < workers:
            spec, cmd = todo.pop(0)
            log = open(RES / "logs" / f"pod_{spec.replace(':', '_')}.log", "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "src.run_final3", *cmd], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (spec, time.time(), log)
            print(f"[{time.strftime('%H:%M:%S')}] start {spec} (pid {p.pid})", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                spec, t0, log = running.pop(p)
                log.close()
                rec = {"job": spec, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1), "workers": workers,
                       "threads_per_worker": int(THREADS), "host": "runpod", "cpu_model": cpu, "vcpu": ncpu}
                with open(RES / "pod_timing.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                rc_all |= int(p.returncode != 0)
                print(f"[{time.strftime('%H:%M:%S')}] done {spec}: rc={p.returncode} {rec['wall_s']}s", flush=True)
    sys.exit(rc_all)


if __name__ == "__main__":
    main()
