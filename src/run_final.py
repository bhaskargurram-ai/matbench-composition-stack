"""FINAL EVALUATION on the official Matbench v0.1 folds (run only after the design freeze).

For every task and official fold, the frozen pipeline (src/pipeline.py, src/experts.py,
cfg from src/frozen_config.py) is fitted on the official training rows and predicts the official
test rows. Test TARGETS are never loaded here; scoring happens in src/evaluate.py.
Resumable: a fold whose predictions file exists is skipped. Per-expert caches allow
resuming inside a fold.

Usage: python -m src.run_final [task ...]
"""
from __future__ import annotations
import json, sys, time
import numpy as np

from .data import load_task_frames, featurise, featurise_extra, ROOT, TASKS
from .pipeline import run_split
from .design import TASK_TYPE
from .frozen_config import CFG, SEED, N_INNER


def run_task(task, only_fold=None):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    X_all = featurise(task, df)
    if (CFG.get(task) or {}).get("extra_features"):
        X_all = X_all.join(featurise_extra(task, df))
    out_root = ROOT / "results" / "final" / task
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / (f"log_fold{only_fold}.txt" if only_fold is not None else "log.txt"), "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    for k, f in enumerate(folds):
        if only_fold is not None and k != only_fold:
            continue
        pf = out_root / f"fold{k}_predictions.npz"
        if pf.exists():
            log(f"[{task}] fold {k}: already done, skipping")
            continue
        t0 = time.time()
        tr, te = f["train"], f["test"]
        assert not (set(tr) & set(te))
        X_tr, y_tr = X_all.loc[tr], df.loc[tr, "target"].to_numpy(dtype=float)
        X_te = X_all.loc[te]  # features only; no test targets
        log(f"[{task}] fold {k}: n_train={len(tr)} n_test={len(te)} cfg={CFG.get(task)}")
        preds, meta = run_split(tt, X_tr, y_tr, X_te, out_root / f"fold{k}_experts", cfg=CFG.get(task),
                                n_inner=N_INNER, seed=SEED, log=log)
        np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
        json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
        log(f"[{task}] fold {k}: done in {time.time() - t0:.0f}s; weights={meta['stack_weights']}; "
            f"sel={ {x: meta['sel_stack_local'][x] for x in ('space', 'k', 'hmult', 'kappa', 'alpha')} }")


if __name__ == "__main__":
    args = sys.argv[1:]
    fold = None
    if "--fold" in args:
        i = args.index("--fold"); fold = int(args[i + 1]); args = args[:i] + args[i + 2:]
    for t in (args or TASKS):
        run_task(t, fold)
