"""Post-hoc ablation of the frozen Amendment-2 pipeline (ANALYSIS_PLAN.md Section 12; not used for any
selection). Variants:
  nodft     : all tasks, DFT block removed (features built without the 'dft' block)
  nohurdle  : expt_gap only, hurdle experts removed (DFT block kept)
Everything else is the frozen Amendment-2 code/config (src.pipeline2.run_split2, src.frozen_config2).
Expert caches from results/final_a2 are copied only for experts whose inputs/config are identical.
Usage: python -m src.run_ablation <variant> <task> --fold k
"""
from __future__ import annotations
import json, shutil, sys, time
import numpy as np
from .data import load_task_frames, ROOT
from .features_all import features
from .pipeline2 import run_split2
from .design import TASK_TYPE
from .frozen_config2 import CFG2, BLOCKS2, SEED, N_INNER

A2 = ROOT / "results" / "final_a2"
# experts whose inputs do not involve the DFT block (magpie_/frac_ columns only)
NO_DFT_SAFE = {"krr_frac", "krr_magpie", "svc_frac", "svc_magpie", "knn_frac", "et_frac"}


def run(variant, task, k):
    tt = TASK_TYPE[task]
    df, folds = load_task_frames(task)
    cfg = dict(CFG2[task])
    if variant == "nodft":
        blocks = tuple(b for b in BLOCKS2 if b != "dft")
        reuse = NO_DFT_SAFE
    elif variant == "nohurdle":
        assert task == "matbench_expt_gap"
        blocks = BLOCKS2
        cfg = dict(cfg, hurdle=False)
        reuse = None  # every remaining expert (and the tuned parameters) is identical to Amendment 2
    else:
        raise ValueError(variant)
    out_root = ROOT / "results" / f"ablation_{variant}" / task
    out_root.mkdir(parents=True, exist_ok=True)
    pf = out_root / f"fold{k}_predictions.npz"
    if pf.exists():
        print("already done"); return
    cache = out_root / f"fold{k}_experts"
    cache.mkdir(parents=True, exist_ok=True)
    src = A2 / task / f"fold{k}_experts"
    for f in src.glob("*"):
        name = f.name.replace("_params.json", "").replace(".npz", "")
        if (reuse is None and not name.startswith("hurdle")) or (reuse is not None and name in reuse):
            if not (cache / f.name).exists():
                shutil.copy2(f, cache / f.name)
    X_all = features(task, df, blocks)
    assert not any(c.startswith("dft_") for c in X_all.columns) or variant != "nodft"
    X_all["composition"] = df["composition"].values
    f = folds[k]; tr, te = f["train"], f["test"]
    y_tr = df.loc[tr, "target"].to_numpy(dtype=float)
    logf = open(out_root / f"log_fold{k}.txt", "a", encoding="utf-8")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    t0 = time.time()
    log(f"[ablation {variant}] {task} fold {k}: ncols={X_all.shape[1]} cfg={cfg}")
    preds, meta = run_split2(tt, X_all.loc[tr], y_tr, X_all.loc[te], cache, cfg=cfg, n_inner=N_INNER, seed=SEED, log=log)
    np.savez(pf, ids=np.array(te), **{a.replace(":", "__"): v for a, v in preds.items()})
    json.dump(meta, open(out_root / f"fold{k}_meta.json", "w"), indent=1, default=float)
    log(f"[ablation {variant}] {task} fold {k}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], int(a[a.index("--fold") + 1]))
