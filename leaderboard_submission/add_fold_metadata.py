"""Add per-fold compute metadata to the submission's results.json.gz (requested in matbench PR #372).

Amendment-3 version. For every task and fold the recorded ``parameters`` gain:
  wall_time_s          selection job wall time on the PC (results/final_a3/<task>/fold<k>_meta.json["compute"]
                       ["selection_job"]) plus the wall time of the separate expert-fit jobs of that fold (pod / PC,
                       ``fit_info`` runs of every candidate cache dir, excluding the run logged by the selection job
                       itself, which is already inside its wall time);
  n_workers, threads_per_worker, cpu_threads_used, cpu_model, gpu   of the selection job (every expert runs with
                       4 threads); the hosts of the expert-fit jobs are listed under ``fit_hosts`` (per candidate block
                       set: host, cpu_model, n_workers, threads_per_worker, wall_s, experts fitted there);
  expert_fit_seconds   per expert of the selected candidate (fit time of the computation actually used, measured
                       where it was computed; Amendment-2 caches reused by identical inputs keep their original time);
  selected_candidate, candidate_inner_cv_scores, candidates_dropped_for_budget, pipeline, seed.
user_metadata.algorithm is "DFT-Stack". Predictions, scores and the datestamp are not touched: the script checks
that every fold's recorded predictions and every task's scores are identical before and after. The file is
re-saved gzip-compressed (mtime=0) in place.  Run from the repository root:
    python leaderboard_submission/add_fold_metadata.py
"""
from __future__ import annotations
import gzip
import json
import pathlib

from matbench.bench import MatbenchBenchmark
from matbench.util import hash_dictionary

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "leaderboard_submission" / "matbench_v0.1_DFTStack" / "results.json.gz"
A3 = ROOT / "results" / "final_a3"
GPU = "none"
CPU_THREADS_USED = 4          # n_jobs / thread_count of every expert (dftstack/experts2.py)
PIPELINE = "DFT-Stack (Amendment-3: fold-internal candidate selection, frozen Amendment-2 pipeline)"


def load(path):
    """matbench's from_file; the 0.6 wheel cannot read a gzip, so fall back to the decompressed JSON."""
    try:
        return MatbenchBenchmark.from_file(str(path))
    except Exception:
        return MatbenchBenchmark.from_dict(json.loads(gzip.open(path).read().decode("utf-8")))


def fold_params(meta):
    c = meta["compute"]
    sel = c["selection_job"]
    wall = float(sel["wall_s"])
    hosts = {}
    for bs, info in c["expert_fits"].items():
        runs = info["fit_info"]["runs"][:-1]          # the last run is the selection job's own (already counted)
        for r in runs:
            wall += float(r["wall_s"])
        hosts[bs] = [{"host": "cloud CPU pod (RunPod cpu5c)" if "EPYC" in r["cpu_model"] else "workstation", "cpu_model": r["cpu_model"], "n_workers": r["n_workers"],
                      "threads_per_worker": r["threads_per_worker"], "wall_s": round(float(r["wall_s"]), 1),
                      "experts_fitted": r["fitted_here"]} for r in runs]
    ds = meta["design_selection"]
    return {
        "pipeline": PIPELINE, "seed": 0,
        "wall_time_s": round(wall, 1),
        "n_workers": int(sel["n_workers"]), "threads_per_worker": int(sel["threads_per_worker"]),
        "cpu_threads_used": CPU_THREADS_USED, "cpu_model": sel["cpu_model"], "gpu": GPU,
        "fit_hosts": hosts,
        "expert_fit_seconds": {e: round(float(s), 1) for e, s in meta["expert_seconds"].items()},
        "selected_candidate": ds["selected"],
        "candidate_inner_cv_scores": {r["candidate"]: round(float(r["stack_oof_score"]), 6) for r in ds["candidates"]},
        "candidates_dropped_for_budget": ds.get("dropped_for_budget", []),
    }


def main():
    raw = gzip.open(SUB).read().decode("utf-8")
    d0 = json.loads(raw)
    mb = load(SUB)
    assert mb.is_valid and mb.is_recorded
    before = {t.dataset_name: ({k: dict(v["data"]) for k, v in t.results.items()}, dict(t.scores)) for t in mb.tasks}
    for task in mb.tasks:
        name = task.dataset_name
        for k in range(5):
            meta = json.load(open(A3 / name / f"fold{k}_meta.json"))
            p = dict(task.results[f"fold_{k}"]["parameters"])
            p.pop("variant", None)
            p.update(fold_params(meta))
            task.results[f"fold_{k}"]["parameters"] = p
    um = dict(mb.user_metadata)
    um["algorithm"] = "DFT-Stack"
    um.pop("variant", None); um.pop("submitted", None)
    mb.add_metadata(um)

    d = mb.as_dict()
    d["datestamp"] = d0["datestamp"]            # keep the scoring-time stamp; re-hash the final dict
    d.pop("hash")
    d["hash"] = hash_dictionary(d)
    out = json.dumps(d)
    with gzip.GzipFile(SUB, "wb", mtime=0) as f:
        f.write(out.encode("utf-8"))

    mb2 = load(SUB)
    assert mb2.is_valid and mb2.is_recorded
    for t in mb2.tasks:
        data0, scores0 = before[t.dataset_name]
        for fk, v in t.results.items():
            assert dict(v["data"]) == data0[fk], (t.dataset_name, fk)
            assert json.dumps(d0["tasks"][t.dataset_name]["results"][fk]["data"]) == \
                json.dumps(d["tasks"][t.dataset_name]["results"][fk]["data"]), (t.dataset_name, fk)
            for key in ("wall_time_s", "n_workers", "threads_per_worker", "cpu_threads_used", "cpu_model", "gpu",
                        "expert_fit_seconds", "pipeline", "seed", "selected_candidate", "fit_hosts"):
                assert key in v["parameters"], (t.dataset_name, fk, key)
        assert dict(t.scores) == scores0, t.dataset_name
    print("user_metadata:", mb2.user_metadata)
    for t in mb2.tasks:
        key = "mae" if t.metadata.task_type == "regression" else "rocauc"
        print(f"{t.dataset_name}: {key} mean {t.scores[key]['mean']:.4f} (unchanged); selected per fold:",
              [t.results[f'fold_{k}']['parameters']['selected_candidate'] for k in range(5)])
    steels = next(t for t in mb2.tasks if t.dataset_name == "matbench_steels")
    print("example parameters:", json.dumps(steels.results["fold_0"]["parameters"], indent=1)[:1500])
    print("wrote", SUB)


if __name__ == "__main__":
    main()
