"""Amendment-2 scoring (ANALYSIS_PLAN.md Sections 9-10). Loads official test targets ONLY here, and only
after the Amendment-2 freeze and the complete final_a2 run.

Primary (per task): final Amendment-2 pipeline (`stack_local`: stack + inner-CV-selected optional local
correction) vs the leaderboard leader, on the board's own metric semantics:
  - regression: MAE;
  - classification: balanced accuracy, ours at the inner-CV threshold; leader from its recorded labels
    (Darwin) or its probabilities thresholded at 0.5 (MODNet v0.1.12), i.e. what the board computed.
  Paired bootstrap over test points within folds (B = 10,000), using the leader's public per-sample
  predictions (matbench repository `benchmarks/*/results.json.gz`); Holm over the 4 tasks.
Retained: stack vs best single; final A2 vs Amendment-1 primary arm; local correction vs stack.
Also reported: probability ROC-AUC (ours; leader's where probabilities exist).
Usage: python -m src.evaluate2 [--submission]
"""
from __future__ import annotations
import gzip, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score, roc_auc_score

from .data import load_task_frames, ROOT, TASKS
from .design import TASK_TYPE
from .evaluate import fold_metrics, holm, ece
from . import evaluate as ev1

A2 = ROOT / "results" / "final_a2"
A1 = ROOT / "results" / "final"
LEAD = ROOT / "data" / "cache" / "external" / "leaders"
LEADER = {"matbench_steels": "TPOT", "matbench_expt_gap": "darwin", "matbench_expt_is_metal": "darwin",
          "matbench_glass": "modnet_v0.1.12"}
LEADER_NAME = {"TPOT": "TPOT-Mat", "darwin": "Darwin", "modnet_v0.1.12": "MODNet (v0.1.12)"}
B = 10000
REG = {"matbench_steels", "matbench_expt_gap"}


def leader_preds(task, ids_by_fold):
    r = json.load(gzip.open(LEAD / f"matbench_v0.1_{LEADER[task]}.results.json.gz"))
    out = []
    for k, ids in enumerate(ids_by_fold):
        d = r["tasks"][task]["results"][f"fold_{k}"]["data"]
        out.append(np.array([d[i] for i in ids]))
    return out


def load(root, task):
    df, folds = load_task_frames(task)
    res = []
    for k in range(5):
        z = np.load(root / task / f"fold{k}_predictions.npz")
        meta = json.load(open(root / task / f"fold{k}_meta.json"))
        ids = [str(i) for i in z["ids"]]
        assert ids == folds[k]["test"]
        y = df.loc[ids, "target"].to_numpy(dtype=float)
        res.append((ids, y, {a.replace("__", ":"): z[a] for a in z.files if a != "ids"}, meta))
    return res


def board_metric(task, y, p, thr=None):
    """Board semantics: MAE; or balanced accuracy from labels (bool), from probabilities at thr (ours) or 0.5."""
    if task in REG:
        return mean_absolute_error(y, p)
    yi = y.astype(int)
    if p.dtype == bool:
        return balanced_accuracy_score(yi, p)
    return balanced_accuracy_score(yi, p >= (0.5 if thr is None else thr))


def paired_boot(task, ys, pa, pb, ta, tb, rng):
    f = lambda y, p, t: board_metric(task, y, p, t)
    obs = np.mean([f(y, a, t1) - f(y, b, t2) for y, a, b, t1, t2 in zip(ys, pa, pb, ta, tb)])
    d = np.empty(B)
    for i in range(B):
        vals = []
        for y, a, b, t1, t2 in zip(ys, pa, pb, ta, tb):
            r = rng.integers(0, len(y), len(y))
            vals.append(f(y[r], a[r], t1) - f(y[r], b[r], t2))
        d[i] = np.mean(vals)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(1.0, max(p, 1.0 / B)))


def main(write_submission=False):
    rng = np.random.default_rng(12345)
    rows, stats = [], {}
    for task in TASKS:
        tt = TASK_TYPE[task]
        a2 = load(A2, task); a1 = load(A1, task)
        ids = [x[0] for x in a2]; ys = [x[1] for x in a2]
        lp = leader_preds(task, ids)
        for k, (_, y, P, meta) in enumerate(a2):
            thr = meta.get("thresholds", {})
            for a in P:
                rows.append({"task": task, "fold": k, "arm": a, **fold_metrics(tt, y, P[a], thr.get(a)),
                             "board_metric": board_metric(task, y, P[a], thr.get(a))})
            rows.append({"task": task, "fold": k, "arm": "LEADER:" + LEADER_NAME[LEADER[task]],
                         "board_metric": board_metric(task, y, lp[k])})
            rows.append({"task": task, "fold": k, "arm": "A1:stack_local",
                         "board_metric": board_metric(task, y, a1[k][2]["stack_local"], a1[k][3].get("thresholds", {}).get("stack_local"))})
        fin = [x[2]["stack_local"] for x in a2]
        fthr = [x[3].get("thresholds", {}).get("stack_local") for x in a2]
        st = {}
        obs, lo, hi, p = paired_boot(task, ys, fin, lp, fthr, [None] * 5, rng)
        ours = np.mean([board_metric(task, y, q, t) for y, q, t in zip(ys, fin, fthr)])
        lead = np.mean([board_metric(task, y, q) for y, q in zip(ys, lp)])
        st["primary_vs_leader"] = {"leader": LEADER_NAME[LEADER[task]], "metric": "mae" if task in REG else "balanced_accuracy",
                                   "ours": float(ours), "leader_value": float(lead), "delta": obs, "ci95": [lo, hi], "p_boot": p,
                                   "ours_folds": [float(board_metric(task, y, q, t)) for y, q, t in zip(ys, fin, fthr)],
                                   "leader_folds": [float(board_metric(task, y, q)) for y, q in zip(ys, lp)]}
        a1p = [x[2]["stack_local"] for x in a1]; a1t = [x[3].get("thresholds", {}).get("stack_local") for x in a1]
        o2, l2, h2, p2 = paired_boot(task, ys, fin, a1p, fthr, a1t, rng)
        st["a2_vs_a1"] = {"delta": o2, "ci95": [l2, h2], "p_boot": p2}
        # retained Amendment-1-style comparisons on the native metric (MAE / probability ROC-AUC)
        data = [(y, P, m) for _, y, P, m in a2]
        st["stack_vs_best_single"] = ev1.compare(tt, data, "stack", "best_single", rng)
        st["local_vs_stack"] = ev1.compare(tt, data, "stack_local", "stack", rng)
        st["selections"] = [{"fold": k, "best_single": m["best_single"], "stack_weights": m["stack_weights"],
                             "sel_stack_local": m["sel_stack_local"], "tuned": {n: v.get("params") for n, v in m.get("tuned", {}).items()},
                             "expert_seconds": m.get("expert_seconds")} for k, (_, _, _, m) in enumerate(a2)]
        if task not in REG:
            # probability ROC-AUC, where the leader recorded probabilities
            auc_ours = [roc_auc_score(y.astype(int), q) for y, q in zip(ys, fin)]
            st["prob_rocauc_ours"] = float(np.mean(auc_ours))
            if lp[0].dtype != bool:
                st["prob_rocauc_leader"] = float(np.mean([roc_auc_score(y.astype(int), q) for y, q in zip(ys, lp)]))
        stats[task] = st
    ps = [stats[t]["primary_vs_leader"]["p_boot"] for t in TASKS]
    for t, p in zip(TASKS, holm(ps)):
        stats[t]["primary_vs_leader"]["p_holm"] = p
    df = pd.DataFrame(rows)
    df.to_csv(A2 / "metrics_per_fold.csv", index=False)
    summ = {t: {a: {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std(ddof=1))}
                    for m in sub.columns if m not in ("task", "fold", "arm") and sub[m].notna().any()}
                for a, sub in df[df.task == t].groupby("arm")} for t in TASKS}
    json.dump(summ, open(A2 / "summary.json", "w"), indent=1)
    json.dump(stats, open(A2 / "stats.json", "w"), indent=1)
    for t in TASKS:
        s = stats[t]["primary_vs_leader"]
        print(t, f"ours={s['ours']:.4f} leader({s['leader']})={s['leader_value']:.4f} delta={s['delta']:+.4f} "
                 f"CI={s['ci95']} p={s['p_boot']:.4f} pHolm={s['p_holm']:.4f}")
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
                z = np.load(A2 / task.dataset_name / f"fold{k}_predictions.npz")
                meta = json.load(open(A2 / task.dataset_name / f"fold{k}_meta.json"))
                p = z["stack_local"]
                if tt == "classification" and variant == "labels":
                    p = [bool(v) for v in (p >= meta["thresholds"]["stack_local"])]
                else:
                    p = [float(v) for v in p]
                task.record(fold, p, params={"pipeline": "Amendment-2 final", "variant": variant})
        out = A2 / "matbench_submission" / variant
        out.mkdir(parents=True, exist_ok=True)
        mb.add_metadata({"algorithm": "XStack-A2 (cross-fitted stack, DFT transfer features)", "variant": variant,
                         "submitted": False})
        mb.to_file(str(out / "results.json.gz"))
        print("wrote", out / "results.json.gz", "valid:", mb.is_valid)


if __name__ == "__main__":
    main(write_submission="--submission" in sys.argv)
