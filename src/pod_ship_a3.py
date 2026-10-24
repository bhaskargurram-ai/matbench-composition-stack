"""Build the folder shipped to a RunPod CPU pod for the Amendment-3 expert fits (scheduling only).

Copies src/ (the frozen modules), the cached feature blocks and official folds of the given tasks
(data/cache/<task>_frame.pkl, _folds.json, _features*.pkl, _fb_<block>.pkl for the blocks in BLOCKS2), the
Amendment-2 expert caches (results/final_a2/<task>/fold<k>_experts/, reused by identical inputs) and a pinned
requirements file. No .env, no git data, no test targets beyond the task frame that the pipeline already uses
(the pod jobs read training targets only; fold membership is the official one).
Usage: python -m src.pod_ship_a3 <dest> <task> [<task> ...]
"""
from __future__ import annotations
import pathlib, shutil, sys
from .data import ROOT, CACHE
from .frozen_config2 import BLOCKS2

PINS = ["numpy==2.5.4", "scipy==1.18.1", "pandas==2.3.3", "scikit-learn==1.9.1", "lightgbm==4.7.0", "xgboost==3.4.1",
        "catboost==1.2.10", "optuna==5.0.0", "joblib==1.6.0"]   # same versions as the submission's requirements.txt


def main(dest, tasks):
    dest = pathlib.Path(dest)
    (dest / "src").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "src").glob("*.py"):
        shutil.copy2(f, dest / "src" / f.name)
    (dest / "data" / "cache").mkdir(parents=True, exist_ok=True)
    n = 0
    for t in tasks:
        names = [f"{t}_frame.pkl", f"{t}_folds.json", f"{t}_features.pkl", f"{t}_features_extra.pkl", f"{t}_features_dft.pkl"]
        names += [f"{t}_fb_{b}.pkl" for b in BLOCKS2 if b not in ("base", "extra", "dft")]
        for nme in names:
            shutil.copy2(CACHE / nme, dest / "data" / "cache" / nme); n += 1
        for d in (ROOT / "results" / "final_a2" / t).glob("fold*_experts"):
            shutil.copytree(d, dest / "results" / "final_a2" / t / d.name, dirs_exist_ok=True)
    (dest / "requirements_pod.txt").write_text("\n".join(PINS) + "\n")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"shipped folder {dest}: {n} cache files, {size:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
