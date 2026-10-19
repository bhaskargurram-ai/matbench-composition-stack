"""Score the post-hoc ablation variants ONCE (ANALYSIS_PLAN.md Section 12). Compares each variant's final
arm (stack_local) with the Amendment-2 final arm and with the leader, using the same paired bootstrap on
the board metric as src.evaluate2 (B = 10,000, within-fold resampling, rng seed 12345).
Writes results/ablation_summary.json (all numbers, incl. per-fold values) and then calls
src.make_ablation_assets to write the manuscript macros/table from that JSON (no hand-typed numbers).
Reporting-only extension (2026-10-19, before any ablation scoring): per-fold board metrics, mean +- s.d.,
probability ROC-AUC for classification; the statistics themselves are unchanged."""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from .data import ROOT, TASKS
from . import evaluate2 as ev2

VARIANTS = {"nodft": TASKS, "nohurdle": ["matbench_expt_gap"]}


def main():
    rng = np.random.default_rng(12345)
    out = {}
    for v, tasks in VARIANTS.items():
        for t in tasks:
            av = ev2.load(ROOT / "results" / f"ablation_{v}", t)
            a2 = ev2.load(ev2.A2, t)
            ys = [x[1] for x in a2]
            assert all(np.array_equal(x[1], y) for x, y in zip(av, ys))
            pv = [x[2]["stack_local"] for x in av]; tv = [x[3].get("thresholds", {}).get("stack_local") for x in av]
            p2 = [x[2]["stack_local"] for x in a2]; t2 = [x[3].get("thresholds", {}).get("stack_local") for x in a2]
            lp = ev2.leader_preds(t, [x[0] for x in a2])
            fv = [float(ev2.board_metric(t, y, q, th)) for y, q, th in zip(ys, pv, tv)]
            f2 = [float(ev2.board_metric(t, y, q, th)) for y, q, th in zip(ys, p2, t2)]
            fl = [float(ev2.board_metric(t, y, q)) for y, q in zip(ys, lp)]
            d, lo, hi, p = ev2.paired_boot(t, ys, pv, p2, tv, t2, rng)
            dl, lol, hil, pl = ev2.paired_boot(t, ys, pv, lp, tv, [None] * 5, rng)
            rec = {"variant": v, "task": t, "metric": "mae" if t in ev2.REG else "balanced_accuracy",
                   "variant_metric": float(np.mean(fv)), "variant_std": float(np.std(fv, ddof=1)), "variant_folds": fv,
                   "a2_metric": float(np.mean(f2)), "a2_std": float(np.std(f2, ddof=1)), "a2_folds": f2,
                   "leader": ev2.LEADER_NAME[ev2.LEADER[t]], "leader_metric": float(np.mean(fl)),
                   "leader_std": float(np.std(fl, ddof=1)), "leader_folds": fl,
                   "delta_vs_a2": d, "ci_vs_a2": [lo, hi], "p_vs_a2": p,
                   "per_fold_delta_vs_a2": [a - b for a, b in zip(fv, f2)],
                   "delta_vs_leader": dl, "ci_vs_leader": [lol, hil], "p_vs_leader": pl,
                   "corr_on_folds": sum(1 for x in av if x[3]["sel_stack_local"].get("alpha", 0) > 0),
                   "best_single": [x[3]["best_single"] for x in av],
                   "weights": [x[3]["stack_weights"] for x in av]}
            if t not in ev2.REG:
                rec["prob_rocauc_variant"] = float(np.mean([roc_auc_score(y.astype(int), q) for y, q in zip(ys, pv)]))
                rec["prob_rocauc_a2"] = float(np.mean([roc_auc_score(y.astype(int), q) for y, q in zip(ys, p2)]))
            out[f"{v}:{t}"] = rec
    json.dump(out, open(ROOT / "results" / "ablation_summary.json", "w"), indent=1)
    for k, r in out.items():
        print(k, f"variant={r['variant_metric']:.4f}+-{r['variant_std']:.4f} A2={r['a2_metric']:.4f}+-{r['a2_std']:.4f} "
                 f"delta={r['delta_vs_a2']:+.4f} CI={[round(c, 4) for c in r['ci_vs_a2']]} p={r['p_vs_a2']:.4f} | "
                 f"vs leader {r['delta_vs_leader']:+.4f} CI={[round(c, 4) for c in r['ci_vs_leader']]} p={r['p_vs_leader']:.4f}")
    from . import make_ablation_assets
    make_ablation_assets.main()


if __name__ == "__main__":
    main()
