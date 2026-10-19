"""FINAL EVALUATION on the official Matbench v0.1 folds (ANALYSIS_PLAN.md incl. Amendment 1).

For every task and official fold k, using the official training rows T_k ONLY:
  1. select_design: cross-fit the experts of every candidate configuration (frozen_config.CANDIDATES)
     on T_k and keep the candidate whose cross-fitted global-stack OOF score is best;
  2. run the full pipeline (src/pipeline.py) with the selected configuration (+ fold-matched arm) and
     predict the official test rows.
Test TARGETS are never loaded here; scoring happens in src/evaluate.py.
Resumable: a fold whose predictions file exists is skipped; per-expert caches allow resuming inside a
fold. Expert caches depend only on (task, fold, feature set, expert config, seed, J); they live in
fold<k>_experts (extra features on) and fold<k>_experts_base (extra features off).

Usage: python -m src.run_final [task ...] [--fold k]
"""
from __future__ import annotations
import json, sys, time
import numpy as np

from .data import load_task_frames, featurise, featurise_extra, ROOT, TASKS
from .pipeline import run_split, crossfit_experts, crossfitted_stack, inner_splits, score, better
from .design import TASK_TYPE
from .frozen_config import CANDIDATES, ALWAYS, SEED, N_INNER


def cache_dir(out_root, k, cand):
    return out_root / (f"fold{k}_experts" if cand.get("extra_features") else f"fold{k}_experts_base")


def select_design(tt, X_base, X_ext, y_tr, out_root, k, log):
    """Inner-CV design selection inside T_k: score = cross-fitted OOF score of the global stack."""
    splits = inner_splits(tt, y_tr, N_INNER, SEED)
    rows, best = [], None
    for cand in CANDIDATES[tt]:
        X = X_ext if cand.get("extra_features") else X_base
        P_oof, _, names, _ = crossfit_experts(tt, X["tr"], y_tr, X["te"], splits, cache_dir(out_root, k, cand),
                                              cfg=cand, seed=SEED, log=log, fold_matched=False)
        s_oof = crossfitted_stack(P_oof, y_tr, splits, tt, cand.get("stack_loss", "l2"))
        sc = float(score(tt, y_tr, s_oof))
        rows.append({"candidate": cand["name"], "stack_oof_score": sc, "experts": names})
        log(f"    design candidate {cand['name']}: stack OOF score {sc:.5f}")
        if best is None or better(tt, sc, best[1]):
            best = (cand, sc)
    return best[0], rows


def run_task(task, only_fold=None):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    Xb_all = featurise(task, df)
    Xe_all = Xb_all.join(featurise_extra(task, df))
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
        y_tr = df.loc[tr, "target"].to_numpy(dtype=float)       # training targets only
        Xb = {"tr": Xb_all.loc[tr], "te": Xb_all.loc[te]}        # test FEATURES only
        Xe = {"tr": Xe_all.loc[tr], "te": Xe_all.loc[te]}
        log(f"[{task}] fold {k}: n_train={len(tr)} n_test={len(te)}; per-fold design selection")
        cand, rows = select_design(tt, Xb, Xe, y_tr, out_root, k, log)
        cfg = {**{x: v for x, v in cand.items() if x != "name"}, **ALWAYS}
        log(f"[{task}] fold {k}: selected design {cand['name']} -> cfg={cfg}")
        X = Xe if cand.get("extra_features") else Xb
        preds, meta = run_split(tt, X["tr"], y_tr, X["te"], cache_dir(out_root, k, cand), cfg=cfg,
                                n_inner=N_INNER, seed=SEED, log=log)
        meta["design_selection"] = {"selected": cand["name"], "cfg": cfg, "candidates": rows}
        np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
        json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
        log(f"[{task}] fold {k}: done in {time.time() - t0:.0f}s; design={cand['name']}; weights={meta['stack_weights']}; "
            f"sel={ {x: meta['sel_stack_local'][x] for x in ('space', 'k', 'hmult', 'kappa', 'alpha')} }")


if __name__ == "__main__":
    args = sys.argv[1:]
    fold = None
    if "--fold" in args:
        i = args.index("--fold"); fold = int(args[i + 1]); args = args[:i] + args[i + 2:]
    for t in (args or TASKS):
        run_task(t, fold)
