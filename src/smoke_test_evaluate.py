"""Smoke test of src/evaluate.py WITHOUT touching official test-fold targets.

Builds a fake 'task' from fold 0's TRAINING rows only (split into 5 pseudo-folds), writes fake
prediction files in a temporary results directory, and runs the evaluation code on them.
Usage: python -m src.smoke_test_evaluate
"""
import json, tempfile, pathlib
import numpy as np
import src.evaluate as ev
from src import data as D
from src.design import TASK_TYPE
from sklearn.model_selection import KFold


_REAL = D.load_task_frames


def fake_frames_factory():
    cache = {}
    def fake(task):
        if task in cache:
            return cache[task]
        df, folds = _REAL(task)
        tr = folds[0]["train"]
        df = df.loc[tr]
        kf = KFold(5, shuffle=True, random_state=0)
        ff = [{"train": [tr[i] for i in a], "test": [tr[i] for i in b]} for a, b in kf.split(tr)]
        cache[task] = (df, ff)
        return cache[task]
    return fake


def main():
    tmp = pathlib.Path(tempfile.mkdtemp())
    fake = fake_frames_factory()
    ev.load_task_frames = fake
    D.load_task_frames = fake
    ev.FINAL = tmp
    ev.B = 200
    rng = np.random.default_rng(0)
    arms = ["stack", "stack_local", "stack_fm", "best_single", "single_local", "mean_ensemble", "single:et_all"]
    for task in D.TASKS:
        df, folds = fake(task)
        (tmp / task).mkdir(parents=True)
        tt = TASK_TYPE[task]
        for k, f in enumerate(folds):
            y = df.loc[f["test"], "target"].to_numpy(dtype=float)
            P = {}
            for a in arms:
                if tt == "regression":
                    P[a.replace(":", "__")] = y + rng.normal(0, y.std() * 0.3, len(y))
                else:
                    P[a.replace(":", "__")] = np.clip(0.7 * y + 0.3 * rng.random(len(y)), 0, 1)
            np.savez(tmp / task / f"fold{k}_predictions.npz", ids=np.array(f["test"]), **P)
            meta = {"best_single": "et_all", "stack_weights": {"et_all": 1.0}, "sel_stack_local": {}, "sel_single_local": {},
                    "thresholds": {a: 0.5 for a in arms}}
            json.dump(meta, open(tmp / task / f"fold{k}_meta.json", "w"))
    ev.main(write_submission=False)
    print("smoke test OK; outputs in", tmp, sorted(p.name for p in tmp.iterdir()))


if __name__ == "__main__":
    main()
