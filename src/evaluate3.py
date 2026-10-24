"""Amendment-3 scoring (ANALYSIS_PLAN.md Section 13): a reporting-only copy of src.evaluate2 pointed at
results/final_a3. Loads official test targets ONLY here, after the Amendment-3 freeze and the complete final_a3 run
(all 20 fold prediction files committed first). Run ONCE.

Same metrics and statistics as Amendment 2: per task, the final arm (`stack_local`) vs the leaderboard leader on
the board's metric semantics (MAE; balanced accuracy at the inner-CV threshold, leader from its recorded labels or
probabilities at 0.5), paired bootstrap over test points within folds (B = 10,000, rng seed 12345), Holm over the
4 tasks; stack vs best single; local correction vs stack; probability ROC-AUC. Added for Amendment 3: A3 vs the
Amendment-2 final arm and A3 vs the Amendment-1 final arm (same paired bootstrap), and the per-fold selected
candidates. `--submission` writes the matbench files (labels and probabilities variants).
Usage: python -m src.evaluate3 [--submission]
"""
from __future__ import annotations
import gzip, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_task_frames, ROOT, TASKS
from .design import TASK_TYPE
from .evaluate import fold_metrics, holm
from . import evaluate as ev1
from .evaluate2 import leader_preds, load, board_metric, paired_boot, LEADER, LEADER_NAME, REG

A3 = ROOT / "results" / "final_a3"
A2 = ROOT / "results" / "final_a2"
A1 = ROOT / "results" / "final"


def main(write_submission=False):
    rng = np.random.default_rng(12345)
    rows, stats = [], {}
    for task in TASKS:
        tt = TASK_TYPE[task]
        a3 = load(A3, task); a2 = load(A2, task); a1 = load(A1, task)
        ids = [x[0] for x in a3]; ys = [x[1] for x in a3]
        lp = leader_preds(task, ids)
        thr_of = lambda runs, k: runs[k][3].get("thresholds", {}).get("stack_local")
        for k, (_, y, P, meta) in enumerate(a3):
            thr = meta.get("thresholds", {})
            for a in P:
                rows.append({"task": task, "fold": k, "arm": a, **fold_metrics(tt, y, P[a], thr.get(a)),
                             "board_metric": board_metric(task, y, P[a], thr.get(a))})
            rows.append({"task": task, "fold": k, "arm": "LEADER:" + LEADER_NAME[LEADER[task]],
                         "board_metric": board_metric(task, y, lp[k])})
            rows.append({"task": task, "fold": k, "arm": "A2:stack_local",
                         "board_metric": board_metric(task, y, a2[k][2]["stack_local"], thr_of(a2, k))})
            rows.append({"task": task, "fold": k, "arm": "A1:stack_local",
                         "board_metric": board_metric(task, y, a1[k][2]["stack_local"], thr_of(a1, k))})
        fin = [x[2]["stack_local"] for x in a3]
        fthr = [thr_of(a3, k) for k in range(5)]
        st = {}
        obs, lo, hi, p = paired_boot(task, ys, fin, lp, fthr, [None] * 5, rng)
        ours = np.mean([board_metric(task, y, q, t) for y, q, t in zip(ys, fin, fthr)])
        lead = np.mean([board_metric(task, y, q) for y, q in zip(ys, lp)])
        st["primary_vs_leader"] = {"leader": LEADER_NAME[LEADER[task]], "metric": "mae" if task in REG else "balanced_accuracy",
                                   "ours": float(ours), "leader_value": float(lead), "delta": obs, "ci95": [lo, hi], "p_boot": p,
                                   "ours_folds": [float(board_metric(task, y, q, t)) for y, q, t in zip(ys, fin, fthr)],
                                   "leader_folds": [float(board_metric(task, y, q)) for y, q in zip(ys, lp)]}
        for lab, prev in (("a3_vs_a2", a2), ("a3_vs_a1", a1)):
            pp = [x[2]["stack_local"] for x in prev]; pt = [thr_of(prev, k) for k in range(5)]
            o2, l2, h2, p2 = paired_boot(task, ys, fin, pp, fthr, pt, rng)
            st[lab] = {"delta": o2, "ci95": [l2, h2], "p_boot": p2,
                       "prev_mean": float(np.mean([board_metric(task, y, q, t) for y, q, t in zip(ys, pp, pt)])),
                       "prev_folds": [float(board_metric(task, y, q, t)) for y, q, t in zip(ys, pp, pt)]}
        data = [(y, P, m) for _, y, P, m in a3]
        st["stack_vs_best_single"] = ev1.compare(tt, data, "stack", "best_single", rng)
        st["local_vs_stack"] = ev1.compare(tt, data, "stack_local", "stack", rng)
        st["selections"] = [{"fold": k, "design_selection": m["design_selection"]["selected"],
                             "candidate_scores": {r["candidate"]: r["stack_oof_score"] for r in m["design_selection"]["candidates"]},
                             "dropped_for_budget": m["design_selection"].get("dropped_for_budget", []),
                             "best_single": m["best_single"], "stack_weights": m["stack_weights"],
                             "sel_stack_local": m["sel_stack_local"], "gate_kept": m["gate_stack_local"]["kept"],
                             "tuned": {n: v.get("params") for n, v in m.get("tuned", {}).items()},
                             "expert_seconds": m.get("expert_seconds")} for k, (_, _, _, m) in enumerate(a3)]
        if task not in REG:
            auc_ours = [roc_auc_score(y.astype(int), q) for y, q in zip(ys, fin)]
            st["prob_rocauc_ours"] = float(np.mean(auc_ours))
            if lp[0].dtype != bool:
                st["prob_rocauc_leader"] = float(np.mean([roc_auc_score(y.astype(int), q) for y, q in zip(ys, lp)]))
        stats[task] = st
    ps = [stats[t]["primary_vs_leader"]["p_boot"] for t in TASKS]
    for t, p in zip(TASKS, holm(ps)):
        stats[t]["primary_vs_leader"]["p_holm"] = p
    df = pd.DataFrame(rows)
    df.to_csv(A3 / "metrics_per_fold.csv", index=False)
    summ = {t: {a: {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std(ddof=1))}
                    for m in sub.columns if m not in ("task", "fold", "arm") and sub[m].notna().any()}
                for a, sub in df[df.task == t].groupby("arm")} for t in TASKS}
    json.dump(summ, open(A3 / "summary.json", "w"), indent=1)
    json.dump(stats, open(A3 / "stats.json", "w"), indent=1)
    for t in TASKS:
        s = stats[t]["primary_vs_leader"]; d2 = stats[t]["a3_vs_a2"]
        print(t, f"A3={s['ours']:.4f} A2={d2['prev_mean']:.4f} leader({s['leader']})={s['leader_value']:.4f} "
                 f"dLeader={s['delta']:+.4f} CI={s['ci95']} p={s['p_boot']:.4f} pHolm={s['p_holm']:.4f}; "
                 f"dA2={d2['delta']:+.4f} CI={d2['ci95']} p={d2['p_boot']:.4f}; "
                 f"selected={[x['design_selection'] for x in stats[t]['selections']]}")
    if write_submission:
        make_submission()


def make_submission():
    from matbench.bench import MatbenchBenchmark
    for variant in ("labels", "probabilities"):
        mb = MatbenchBenchmark(autoload=False, subset=TASKS)
        for task in mb.tasks:
            tt = TASK_TYPE[task.dataset_name]
            task.load()
            for k, fold in enumerate(task.folds):
                z = np.load(A3 / task.dataset_name / f"fold{k}_predictions.npz")
                meta = json.load(open(A3 / task.dataset_name / f"fold{k}_meta.json"))
                p = z["stack_local"]
                if tt == "classification" and variant == "labels":
                    p = [bool(v) for v in (p >= meta["thresholds"]["stack_local"])]
                else:
                    p = [float(v) for v in p]
                task.record(fold, p, params={"pipeline": "Amendment-3 final", "variant": variant, "seed": 0,
                                             "selected_candidate": meta["design_selection"]["selected"]})
        out = A3 / "matbench_submission" / variant
        out.mkdir(parents=True, exist_ok=True)
        mb.add_metadata({"algorithm": "DFT-Stack (Amendment 3: fold-internal candidate selection)", "variant": variant,
                         "submitted": False})
        mb.to_file(str(out / "results.json.gz"))
        print("wrote", out / "results.json.gz", "valid:", mb.is_valid)


if __name__ == "__main__":
    main(write_submission="--submission" in sys.argv)
