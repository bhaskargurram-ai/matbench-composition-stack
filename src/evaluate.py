"""Scoring and pre-registered statistics (ANALYSIS_PLAN.md). This is the ONLY module that loads
official test-fold targets, and it only reads predictions already written by src/run_final.py.

Usage: python -m src.evaluate            -> results/final/metrics_per_fold.csv, summary.json, stats.json
       python -m src.evaluate --submission -> also writes the (unsubmitted) Matbench result files
"""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (mean_absolute_error, mean_squared_error, roc_auc_score,
                             balanced_accuracy_score, brier_score_loss, log_loss)

from .data import load_task_frames, ROOT, TASKS
from .design import TASK_TYPE

FINAL = ROOT / "results" / "final"
PRIMARY = ("stack_local", "stack")          # primary endpoint: full method vs global stack
SECONDARY = [("stack_local", "best_single"), ("stack", "best_single"),
             ("single_local", "best_single"), ("stack_local", "mean_ensemble"),
             ("stack_fm", "stack"), ("stack_fm", "stack_local"),
             ("stack_local_magpie", "stack"), ("stack_local_frac", "stack")]
B = 10000
RNG_SEED = 12345


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def load_task(task):
    df, folds = load_task_frames(task)
    out = []
    for k in range(5):
        z = np.load(FINAL / task / f"fold{k}_predictions.npz", allow_pickle=False)
        meta = json.load(open(FINAL / task / f"fold{k}_meta.json"))
        ids = [str(i) for i in z["ids"]]
        assert ids == folds[k]["test"]
        y = df.loc[ids, "target"].to_numpy(dtype=float)
        preds = {a.replace("__", ":"): z[a] for a in z.files if a != "ids"}
        out.append((y, preds, meta))
    return out


def fold_metrics(tt, y, p, thr=None):
    if tt == "regression":
        return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p)))}
    yi = y.astype(int)
    m = {"rocauc": roc_auc_score(yi, p), "balanced_accuracy_0.5": balanced_accuracy_score(yi, p >= 0.5),
         "brier": brier_score_loss(yi, p), "logloss": log_loss(yi, np.clip(p, 1e-6, 1 - 1e-6)), "ece": ece(yi, p)}
    if thr is not None:
        m["balanced_accuracy_innerthr"] = balanced_accuracy_score(yi, p >= thr)
    return m


def primary_metric(tt):
    return "mae" if tt == "regression" else "rocauc"


def paired_bootstrap(tt, data, a, b, metric, rng):
    """Resample test points within each fold; statistic = mean over folds of metric(a) - metric(b)."""
    f = (lambda y, p: mean_absolute_error(y, p)) if metric == "mae" else (lambda y, p: roc_auc_score(y.astype(int), p))
    obs = np.mean([f(y, P[a]) - f(y, P[b]) for y, P, _ in data])
    deltas = np.empty(B)
    idx_cache = [np.arange(len(y)) for y, _, _ in data]
    for i in range(B):
        ds = []
        for (y, P, _), ix in zip(data, idx_cache):
            r = rng.integers(0, len(ix), len(ix))
            if metric != "mae" and len(np.unique(y[r])) < 2:
                r = ix
            ds.append(f(y[r], P[a][r]) - f(y[r], P[b][r]))
        deltas[i] = np.mean(ds)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(1.0, max(p, 1.0 / B)))


def compare(tt, data, a, b, rng):
    m = primary_metric(tt)
    fa = np.array([fold_metrics(tt, y, P[a])[m] for y, P, _ in data])
    fb = np.array([fold_metrics(tt, y, P[b])[m] for y, P, _ in data])
    d = fa - fb
    improved = int((d < 0).sum()) if tt == "regression" else int((d > 0).sum())
    t = stats.ttest_rel(fa, fb) if np.any(d != 0) else None
    # Nadeau-Bengio corrected resampled t-test (k-fold: variance inflated by 1/k + n_test/n_train)
    k = len(d)
    n_te = np.mean([len(y) for y, _, _ in data]); n_tr = n_te * (k - 1)
    sd = d.std(ddof=1)
    if sd > 0:
        t_nb = d.mean() / np.sqrt((1.0 / k + n_te / n_tr) * sd ** 2)
        p_nb = float(2 * stats.t.sf(abs(t_nb), df=k - 1))
    else:
        p_nb = 1.0
    obs, lo, hi, p_boot = paired_bootstrap(tt, data, a, b, m, rng)
    res = {"arm": a, "comparator": b, "metric": m, "mean_a": float(fa.mean()), "mean_b": float(fb.mean()),
           "delta": float(d.mean()), "rel_change_pct": float(100 * d.mean() / fb.mean()),
           "per_fold_delta": d.tolist(), "folds_improved": improved,
           "paired_t_p": float(t.pvalue) if t is not None else 1.0, "nadeau_bengio_p": p_nb,
           "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
           "boot_delta": obs, "boot_ci95": [lo, hi], "boot_p": p_boot}
    if tt == "regression":
        ea = np.concatenate([np.abs(y - P[a]) for y, P, _ in data])
        eb = np.concatenate([np.abs(y - P[b]) for y, P, _ in data])
        res["wilcoxon_p"] = float(stats.wilcoxon(ea, eb).pvalue) if np.any(ea != eb) else 1.0
    return res


def support_strata(task, arms, n_bins=4):
    """Secondary (pre-registered): test errors stratified by support, i.e. the distance from each test
    composition to its nearest training composition (standardised Magpie, Euclidean). Bin edges are
    the quartiles of the TRAINING rows' leave-one-out nearest-neighbour distances in each fold."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from .data import featurise
    df, folds = load_task_frames(task)
    X = featurise(task, df)
    mag = [c for c in X.columns if c.startswith("magpie_")]
    tt = TASK_TYPE[task]
    recs = []
    for k, f in enumerate(folds):
        sc = StandardScaler().fit(X.loc[f["train"], mag])
        Ztr, Zte = sc.transform(X.loc[f["train"], mag]), sc.transform(X.loc[f["test"], mag])
        nn = NearestNeighbors(n_neighbors=2).fit(Ztr)
        dtr = nn.kneighbors(Ztr)[0][:, 1]
        dte = nn.kneighbors(Zte, n_neighbors=1)[0][:, 0]
        edges = np.quantile(dtr, np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.digitize(dte, edges)
        z = np.load(FINAL / task / f"fold{k}_predictions.npz")
        y = df.loc[f["test"], "target"].to_numpy(dtype=float)
        for a in arms:
            p = z[a.replace(":", "__")]
            for bi in range(n_bins):
                m = b == bi
                if m.sum() < 2:
                    continue
                if tt == "regression":
                    v = float(np.mean(np.abs(y[m] - p[m])))
                else:
                    v = float(np.mean((y[m] - p[m]) ** 2))  # Brier within stratum
                recs.append({"task": task, "fold": k, "arm": a, "bin": bi, "n": int(m.sum()), "value": v})
    return recs


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main(write_submission=False):
    rng = np.random.default_rng(RNG_SEED)
    rows, summary, statsd = [], {}, {}
    for task in TASKS:
        tt = TASK_TYPE[task]
        data = load_task(task)
        arms = list(data[0][1].keys())
        for k, (y, P, meta) in enumerate(data):
            thr = meta.get("thresholds", {})
            for a in arms:
                r = {"task": task, "fold": k, "arm": a, **fold_metrics(tt, y, P[a], thr.get(a))}
                rows.append(r)
        dfm = pd.DataFrame([r for r in rows if r["task"] == task])
        summary[task] = {a: {m: {"mean": float(sub[m].mean()), "std": float(sub[m].std(ddof=1))}
                             for m in sub.columns if m not in ("arm", "task", "fold") and sub[m].notna().any()}
                         for a, sub in dfm.groupby("arm")}
        statsd[task] = {"primary": compare(tt, data, *PRIMARY, rng),
                        "secondary": [compare(tt, data, a, b, rng) for a, b in SECONDARY
                                      if a in data[0][1] and b in data[0][1]],
                        "selections": [{"fold": k, "best_single": m["best_single"], "stack_weights": m["stack_weights"],
                                        "sel_stack_local": m["sel_stack_local"], "sel_single_local": m["sel_single_local"],
                                        "sel_stack_fm": m.get("sel_stack_fm"), "perm_control_naive": m.get("perm_control_naive"),
                                        "perm_control_fm": m.get("perm_control_fm"),
                                        "design_selection": m.get("design_selection")}
                                       for k, (_, _, m) in enumerate(data)]}
    ps = [statsd[t]["primary"]["boot_p"] for t in TASKS]
    for t, p in zip(TASKS, holm(ps)):
        statsd[t]["primary"]["boot_p_holm"] = p
    strata = []
    for task in TASKS:
        arms = [a for a in ("stack", "stack_local", "stack_fm", "best_single") if a in load_task(task)[0][1]]
        strata += support_strata(task, arms)
    pd.DataFrame(strata).to_csv(FINAL / "support_strata.csv", index=False)
    pd.DataFrame(rows).to_csv(FINAL / "metrics_per_fold.csv", index=False)
    json.dump(summary, open(FINAL / "summary.json", "w"), indent=1, default=str)
    json.dump(statsd, open(FINAL / "stats.json", "w"), indent=1)
    print(json.dumps({t: {k: statsd[t]["primary"][k] for k in ("mean_a", "mean_b", "delta", "rel_change_pct",
          "folds_improved", "boot_ci95", "boot_p", "boot_p_holm")} for t in TASKS}, indent=1))
    if write_submission:
        make_submission()


def make_submission():
    """Record the primary arm in MatbenchBenchmark objects and write result files. NOT submitted.
    Classification: (a) hard labels at the inner-CV threshold (comparable with the label-based
    entries on the board) and (b) probabilities."""
    from matbench.bench import MatbenchBenchmark
    for variant in ("labels", "probabilities"):
        mb = MatbenchBenchmark(autoload=False, subset=TASKS)
        for task in mb.tasks:
            tt = TASK_TYPE[task.dataset_name]
            task.load()
            for k, fold in enumerate(task.folds):
                z = np.load(FINAL / task.dataset_name / f"fold{k}_predictions.npz")
                meta = json.load(open(FINAL / task.dataset_name / f"fold{k}_meta.json"))
                p = z["stack_local"]
                if tt == "classification" and variant == "labels":
                    p = [bool(v) for v in (p >= meta["thresholds"]["stack_local"])]
                else:
                    p = [float(v) for v in p]
                task.record(fold, p, params={"arm": "stack_local", "variant": variant})
        out = FINAL / "matbench_submission" / variant
        out.mkdir(parents=True, exist_ok=True)
        mb.add_metadata({"algorithm": "XStack-LOC (cross-fitted stack + novelty-damped local OOF correction)",
                         "variant": variant, "submitted": False})
        mb.to_file(str(out / "results.json.gz"))
        print("wrote", out / "results.json.gz", "valid:", mb.is_valid)


if __name__ == "__main__":
    main(write_submission="--submission" in sys.argv)
