"""Development helper (training rows only): 5-fold CV of a default LightGBM on fold-0 TRAINING rows
for different feature-block combinations. Never touches official test rows."""
import sys, json, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from .data import load_task_frames
from .features_all import features
from .design import TASK_TYPE

def run(task, blocks):
    df, folds = load_task_frames(task)
    tr = folds[0]["train"]
    X = features(task, df, blocks).loc[tr]; y = df.loc[tr, "target"].to_numpy(float)
    tt = TASK_TYPE[task]
    kf = KFold(5, shuffle=True, random_state=7) if tt == "regression" else StratifiedKFold(5, shuffle=True, random_state=7)
    oof = np.zeros(len(y))
    for a, b in kf.split(X, y.astype(int) if tt != "regression" else None):
        p = dict(n_estimators=1500 if len(y) > 1000 else 800, learning_rate=0.02, num_leaves=31, min_child_samples=5 if len(y) < 1000 else 10,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.4, random_state=0, n_jobs=4, verbose=-1)
        if tt == "regression":
            m = lgb.LGBMRegressor(objective="l1", **p).fit(X.iloc[a], y[a]); oof[b] = m.predict(X.iloc[b])
        else:
            m = lgb.LGBMClassifier(**p).fit(X.iloc[a], y[a].astype(int)); oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    s = mean_absolute_error(y, oof) if tt == "regression" else roc_auc_score(y.astype(int), oof)
    return s, X.shape[1]

if __name__ == "__main__":
    task = sys.argv[1]
    for spec in sys.argv[2:]:
        blocks = tuple(spec.split("+"))
        t0 = time.time(); s, n = run(task, blocks)
        print(f"{task} {spec}: score={s:.4f} ncols={n} ({time.time()-t0:.0f}s)", flush=True)
