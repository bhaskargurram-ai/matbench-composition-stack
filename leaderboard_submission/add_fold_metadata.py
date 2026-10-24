"""Add per-fold compute metadata to the submission's results.json.gz (requested in matbench PR #372).

For every task and fold the recorded ``parameters`` gain: wall_time_s (results/final_a2/timing.jsonl, the launcher's
wall time around the fold), n_workers / threads_per_worker (the launcher's parallel layout), cpu_threads_used (every
expert runs with 4 threads), cpu_model, gpu and expert_fit_seconds (results/final_a2/<task>/fold<k>_meta.json
["expert_seconds"], the per-expert cross-fit time). user_metadata.algorithm becomes "DFT-Stack" (the board name);
the variant key stays. Predictions, scores and the datestamp are not touched: the script checks that every fold's
recorded predictions and every task's scores are identical before and after, and that the untouched file
round-trips byte-for-byte through the same serialiser. The file is re-saved gzip-compressed (mtime=0) in place.
Run from the repository root:
    python leaderboard_submission/add_fold_metadata.py
"""
from __future__ import annotations
import gzip
import json
import pathlib
import subprocess
import sys

from matbench.bench import MatbenchBenchmark
from matbench.util import hash_dictionary

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "leaderboard_submission" / "matbench_v0.1_DFTStack" / "results.json.gz"
A2 = ROOT / "results" / "final_a2"
GPU = "none"
CPU_THREADS_USED = 4          # n_jobs / thread_count of every expert (dftstack/experts2.py)


def cpu_model():
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
        return " ".join(out.split())
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine()


def load(path):
    """matbench's from_file; the 0.6 wheel cannot read a gzip, so fall back to the decompressed JSON."""
    try:
        return MatbenchBenchmark.from_file(str(path))
    except Exception:
        return MatbenchBenchmark.from_dict(json.loads(gzip.open(path).read().decode("utf-8")))


def main():
    raw = gzip.open(SUB).read().decode("utf-8")
    d0 = json.loads(raw)
    assert json.dumps(d0) == raw, "serialiser does not round-trip the existing file"
    mb = load(SUB)
    assert mb.is_valid and mb.is_recorded
    before = {t.dataset_name: ({k: dict(v["data"]) for k, v in t.results.items()}, dict(t.scores)) for t in mb.tasks}

    timing = {}
    for line in open(A2 / "timing.jsonl"):
        if line.strip():
            r = json.loads(line)
            if r.get("returncode") == 0:
                timing[(r["task"], r["fold"])] = r          # last successful record per (task, fold)
    cpu = cpu_model()
    for task in mb.tasks:
        name = task.dataset_name
        for k in range(5):
            meta = json.load(open(A2 / name / f"fold{k}_meta.json"))
            tr = timing[(name, k)]
            p = dict(task.results[f"fold_{k}"]["parameters"])
            p.setdefault("pipeline", "DFT-Stack (Amendment-2 frozen config)")
            p.setdefault("seed", 0)
            p.update({
                "wall_time_s": float(tr["wall_s"]),
                "n_workers": int(tr["workers"]),
                "threads_per_worker": int(tr["threads_per_worker"]),
                "cpu_threads_used": CPU_THREADS_USED,
                "cpu_model": cpu,
                "gpu": GPU,
                "expert_fit_seconds": {e: round(float(s), 1) for e, s in meta["expert_seconds"].items()},
            })
            task.results[f"fold_{k}"]["parameters"] = p
    um = dict(mb.user_metadata)
    um["algorithm"] = "DFT-Stack"
    mb.add_metadata(um)

    d = mb.as_dict()
    d["datestamp"] = d0["datestamp"]            # keep the scoring-time stamp; re-hash the final dict
    d.pop("hash")
    d["hash"] = hash_dictionary(d)
    out = json.dumps(d)
    with gzip.GzipFile(SUB, "wb", mtime=0) as f:
        f.write(out.encode("utf-8"))

    # verify: predictions and scores unchanged, file loads, metadata present
    mb2 = load(SUB)
    assert mb2.is_valid and mb2.is_recorded
    for t in mb2.tasks:
        data0, scores0 = before[t.dataset_name]
        for fk, v in t.results.items():
            assert dict(v["data"]) == data0[fk], (t.dataset_name, fk)
            assert json.dumps(d0["tasks"][t.dataset_name]["results"][fk]["data"]) == \
                json.dumps(d["tasks"][t.dataset_name]["results"][fk]["data"]), (t.dataset_name, fk)
            for key in ("wall_time_s", "n_workers", "threads_per_worker", "cpu_threads_used", "cpu_model", "gpu",
                        "expert_fit_seconds", "pipeline", "seed"):
                assert key in v["parameters"], (t.dataset_name, fk, key)
        assert dict(t.scores) == scores0, t.dataset_name
    print("user_metadata:", mb2.user_metadata)
    for t in mb2.tasks:
        key = "mae" if t.metadata.task_type == "regression" else "rocauc"
        print(f"{t.dataset_name}: {key} mean {t.scores[key]['mean']:.4f} (unchanged)")
    steels = next(t for t in mb2.tasks if t.dataset_name == "matbench_steels")
    print("example parameters:", json.dumps(steels.results["fold_0"]["parameters"], indent=1))
    print("wrote", SUB)


if __name__ == "__main__":
    main()
