"""Unit checks of the train-only imputation (src/experts4.py). Synthetic data only; no Matbench row is read.

1. The medians an expert uses come from its fit rows only: changing the values of rows OUTSIDE the fit rows
   (the would-be test / inner-validation rows) does not change the fitted medians or the predictions for a fixed
   query row.
2. With no missing cell, Expert4 gives exactly the Expert2 predictions (so steels, which has none, is unaffected by
   the wrapper).
Usage: python -m src.test_experts4
"""
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from .experts2 import Expert2
from .experts4 import Expert4


def frame(rng, n):
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=["magpie_a", "frac_b", "extra_c", "extra_d", "cbfv_oliynyk_e", "ep_f"])
    return X


def main():
    rng = np.random.default_rng(0)
    fac = lambda p: ExtraTreesRegressor(n_estimators=20, random_state=0)
    X = frame(rng, 120); y = X["magpie_a"].to_numpy() + rng.normal(size=120)
    Xn = X.copy()
    Xn.iloc[::7, 2] = np.nan; Xn.iloc[::5, 4] = np.nan; Xn["extra_d"] = np.nan      # extra_d: NaN in every row
    tr, te = np.arange(80), np.arange(80, 120)
    e = Expert4("t", "F2", fac, "regression").fit(Xn.iloc[tr], y[tr])
    want = Xn.iloc[tr][["extra_c", "extra_d", "cbfv_oliynyk_e"]].median().fillna(0.0)
    assert e.med.equals(want) and e.med["extra_d"] == 0.0
    p1 = e.predict(Xn.iloc[te])
    X2 = Xn.copy()
    X2.iloc[100:, :] = X2.iloc[100:, :] * 50 + 1000          # change rows the fit never sees
    e2 = Expert4("t", "F2", fac, "regression").fit(X2.iloc[tr], y[tr])
    assert e2.med.equals(e.med)
    assert np.array_equal(e2.predict(Xn.iloc[80:100]), p1[:20])
    # the old all-row medians differ from the train-row medians on this data (the test would be vacuous otherwise)
    assert not np.isclose(Xn["extra_c"].median(), e.med["extra_c"])
    a = Expert2("t", "F2", fac, "regression").fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    b = Expert4("t", "F2", fac, "regression").fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    assert np.array_equal(a, b)
    c = Expert4("t", "F2", fac, "regression").fit(X.iloc[tr], y[tr]).clone()
    assert isinstance(c, Expert4) and c.med is None
    print("experts4 checks OK")


if __name__ == "__main__":
    main()
