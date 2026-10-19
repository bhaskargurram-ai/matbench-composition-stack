"""AMENDMENT-2 FINAL EVALUATION on the official Matbench v0.1 folds (ANALYSIS_PLAN.md Sections 9-10).
Per task and official fold k, using the official training rows T_k only: build the Amendment-2 feature
matrix (label-free blocks + disclosed DFT transfer block), run src.pipeline2.run_split2 with the frozen
configuration (src.frozen_config2) and save test predictions. Test TARGETS are never loaded here.
Usage: python -m src.run_final2 [task ...] [--fold k]"""
from __future__ import annotations
import json, sys, time
import numpy as np
from .data import load_task_frames, ROOT, TASKS
from .features_all import features
from .pipeline2 import run_split2
from .design import TASK_TYPE
from .frozen_config2 import CFG2, BLOCKS2, SEED, N_INNER

OUT = ROOT / "results" / "final_a2"


def run_task(task, only_fold=None):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    X_all = features(task, df, BLOCKS2)
    X_all["composition"] = df["composition"].values
    out_root = OUT / task
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / (f"log_fold{only_fold}.txt" if only_fold is not None else "log.txt"), "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    for k, f in enumerate(folds):
        if only_fold is not None and k != only_fold:
            continue
        pf = out_root / f"fold{k}_predictions.npz"
        if pf.exists():
            log(f"[{task}] fold {k}: already done, skipping"); continue
        t0 = time.time()
        tr, te = f["train"], f["test"]
        assert not (set(tr) & set(te))
        y_tr = df.loc[tr, "target"].to_numpy(dtype=float)
        log(f"[{task}] fold {k}: n_train={len(tr)} n_test={len(te)} ncols={X_all.shape[1]} cfg={CFG2[task]}")
        preds, meta = run_split2(tt, X_all.loc[tr], y_tr, X_all.loc[te], out_root / f"fold{k}_experts", cfg=CFG2[task],
                                 n_inner=N_INNER, seed=SEED, log=log)
        np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
        json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
        log(f"[{task}] fold {k}: done in {time.time() - t0:.0f}s; weights={ {a: round(b, 3) for a, b in meta['stack_weights'].items()} }")


if __name__ == "__main__":
    args = sys.argv[1:]
    fold = None
    if "--fold" in args:
        i = args.index("--fold"); fold = int(args[i + 1]); args = args[:i] + args[i + 2:]
    for t in (args or TASKS):
        run_task(t, fold)
