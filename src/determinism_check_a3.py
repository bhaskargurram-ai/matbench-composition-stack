"""Reporting only (ANALYSIS_PLAN.md Section 13.4): compare the from-scratch re-fit of fold 0 (block set F_full, full
pool, done on a Linux cloud CPU pod) with the cached Amendment-2 expert outputs of the same fold (Windows
workstation). For every expert: max |difference| of OOF and test predictions; for the tuned LightGBM expert whether
the re-tuned parameters are identical. Writes results/final_a3/determinism/report.json and prints it. No test
target is read.
Usage: python -m src.determinism_check_a3
"""
from __future__ import annotations
import json
import numpy as np
from .data import ROOT

DET = ROOT / "results" / "final_a3" / "determinism"
A2 = ROOT / "results" / "final_a2"
TREES = ("lgbm_F2_tuned", "xgb_F2", "cat_F1", "et_F1", "et_F2", "et_frac", "hurdle_lgbm_F2_exp", "hurdle_lgbm_F2_median",
         "hurdle_et_F1_exp")


def main():
    rep = {"tasks": {}}
    for d in sorted(DET.glob("matbench_*")):
        task = d.name
        new, old = d / "fold0_experts_F_full", A2 / task / "fold0_experts"
        per, info = {}, json.load(open(new / "fit_info.json"))["runs"][-1]
        for f in sorted(new.glob("*.npz")):
            name = f.stem
            if not (old / f.name).exists():
                per[name] = {"note": "not part of the Amendment-2 run (et_frac was dropped for expt_gap); no reference"}
                continue
            zn, zo = np.load(f), np.load(old / f.name)
            per[name] = {"max_abs_diff_oof": float(np.max(np.abs(zn["oof"] - zo["oof"]))),
                         "max_abs_diff_test": float(np.max(np.abs(zn["test"] - zo["test"]))),
                         "seconds_pod": float(zn["seconds"]), "seconds_pc": float(zo["seconds"])}
        pn, po = json.load(open(new / "lgbm_F2_tuned_params.json")), json.load(open(old / "lgbm_F2_tuned_params.json"))
        same = pn["params"] == po["params"]
        def mx(names, key):
            v = [per[n][key] for n in names if n in per and key in per[n]]
            return float(max(v)) if v else None
        trees = [n for n in per if n in TREES]
        other = [n for n in per if n not in TREES]
        rep["tasks"][task] = {
            "pod": {k: info[k] for k in ("host", "cpu_model", "platform", "python")},
            "lgbm_params_identical": bool(same), "lgbm_best_value_pod": pn["value"], "lgbm_best_value_pc": po["value"],
            "max_abs_diff_test_all_experts": mx(list(per), "max_abs_diff_test"),
            "max_abs_diff_oof_all_experts": mx(list(per), "max_abs_diff_oof"),
            "max_abs_diff_test_trees": mx(trees, "max_abs_diff_test"),
            "max_abs_diff_test_other": mx(other, "max_abs_diff_test"),
            "per_expert": per,
        }
    DET.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(DET / "report.json", "w"), indent=1)
    for t, r in rep["tasks"].items():
        print(t, {k: v for k, v in r.items() if k != "per_expert"})
        for n, v in r["per_expert"].items():
            print("   ", n, v)


if __name__ == "__main__":
    main()
