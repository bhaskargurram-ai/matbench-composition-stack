"""Amendment-2 development on fold 0's TRAINING rows only (pseudo-outer folds; seed 2026).
Usage: python -m src.design2 <task> <tag> [n_pseudo] [cfg-json]"""
import json, sys, time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from .data import load_task_frames, ROOT
from .features_all import features
from .pipeline2 import run_split2
from .pipeline import score
from .design import TASK_TYPE
from .frozen_config2 import CFG2, BLOCKS2

def main(task, tag, n_pseudo=2, cfg=None):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    X_all = features(task, df, BLOCKS2); X_all["composition"] = df["composition"].values
    tr = folds[0]["train"]
    X, y = X_all.loc[tr], df.loc[tr, "target"].to_numpy(float)
    del X_all
    cfg = {**CFG2[task], **(cfg or {})}
    out = ROOT / "results" / "design_a2" / tag / task; out.mkdir(parents=True, exist_ok=True)
    kf = KFold(5, shuffle=True, random_state=2026) if tt == "regression" else StratifiedKFold(5, shuffle=True, random_state=2026)
    logf = open(out / "log.txt", "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    log(f"== design_a2 {tag} {task} cfg={cfg}")
    rows = []
    for i, (a, b) in enumerate(kf.split(X, y.astype(int) if tt != "regression" else None)):
        if i >= n_pseudo: break
        t0 = time.time()
        preds, meta = run_split2(tt, X.iloc[a], y[a], X.iloc[b], out / f"pseudo{i}", cfg=cfg, log=log)
        sc = {k: float(score(tt, y[b], v)) for k, v in preds.items()}
        if tt != "regression":
            from sklearn.metrics import balanced_accuracy_score
            for k2 in ("stack", "stack_local", "best_single"):
                sc[k2 + "_BA"] = float(balanced_accuracy_score(y[b].astype(int), preds[k2] >= meta["thresholds"][k2]))
        rows.append(sc)
        log(f"  pseudo {i} ({time.time()-t0:.0f}s): " + ", ".join(f"{k}={v:.4f}" for k, v in sc.items()))
        log(f"    weights={ {k: round(v, 3) for k, v in meta['stack_weights'].items()} } corr={meta['sel_stack_local']['alpha']}")
    log("  MEAN: " + ", ".join(f"{k}={np.mean([r[k] for r in rows]):.4f}" for k in rows[0]))

if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 2, json.loads(a[3]) if len(a) > 3 else None)
