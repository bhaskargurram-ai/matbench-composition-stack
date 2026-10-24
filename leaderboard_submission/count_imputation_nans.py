"""Count the cells that the two label-free median imputations fill, per task, before imputation.

Two feature builders fill missing values with column medians over ALL rows of a task (training and test rows
together, no labels): CBFV generate_features (CBFV/composition.py, `X[cols].fillna(X[cols].median())`) for the
cbfv_oliynyk block, and src/data.py featurise_extra (`X.fillna(X.median())`) for the 21 "extra" descriptors.
This script rebuilds both blocks the way the pipeline does and records how many cells / rows were missing.
Output: results/final_a2/imputation_nan_counts.json (read by make_submission_docs.py). Run from the repository root:
    python leaderboard_submission/count_imputation_nans.py
"""
from __future__ import annotations
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from src.data import TASKS, load_task_frames  # noqa: E402


def cbfv_pre_imputation(df):
    """Run CBFV exactly as features_more.block does and capture the NaN count at its fillna call."""
    import CBFV.composition as cc
    seen = {}
    orig = pd.DataFrame.fillna

    def spy(self, value=None, *a, **k):
        if isinstance(value, pd.Series) and "n" not in seen:
            seen["n"] = (int(self.isna().sum().sum()), int(self.isna().any(axis=1).sum()), list(self.shape))
        return orig(self, value, *a, **k)

    pd.DataFrame.fillna = spy
    try:
        tmp = pd.DataFrame({"formula": df["composition"].values, "target": 0.0})
        _, _, _, skipped = cc.generate_features(tmp, elem_prop="oliynyk", drop_duplicates=False,
                                                extend_features=False, sum_feat=True)
    finally:
        pd.DataFrame.fillna = orig
    assert len(skipped) == 0
    return seen["n"]


def extra_pre_imputation(df):
    """The featurise_extra block of src/data.py, stopped before its fillna."""
    from pymatgen.core import Composition
    from matminer.featurizers.composition import (Stoichiometry, ValenceOrbital, IonProperty, BandCenter,
                                                  AtomicOrbitals)
    comps = [Composition(c) for c in df["composition"]]
    fz = [Stoichiometry(), ValenceOrbital(impute_nan=True), IonProperty(fast=True, impute_nan=True),
          BandCenter(impute_nan=True), AtomicOrbitals()]
    blocks = []
    for f in fz:
        vals = []
        for c in comps:
            try:
                vals.append(f.featurize(c))
            except Exception:
                vals.append([np.nan] * len(f.feature_labels()))
        blocks.append(np.array([[v if isinstance(v, (int, float, np.floating)) else np.nan for v in r] for r in vals],
                               dtype=float))
    X = pd.DataFrame(np.hstack(blocks), index=df.index)
    X = X.loc[:, X.notna().any()]
    return int(X.isna().sum().sum()), int(X.isna().any(axis=1).sum()), list(X.shape)


def main():
    out = {}
    for t in TASKS:
        df, _ = load_task_frames(t)
        c_cells, c_rows, c_shape = cbfv_pre_imputation(df)
        e_cells, e_rows, e_shape = extra_pre_imputation(df)
        out[t] = {"n_rows": len(df),
                  "cbfv_oliynyk": {"nan_cells": c_cells, "rows_with_nan": c_rows, "shape": c_shape},
                  "extra": {"nan_cells": e_cells, "rows_with_nan": e_rows, "shape": e_shape}}
        print(t, out[t], flush=True)
    p = ROOT / "results" / "final_a2" / "imputation_nan_counts.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
