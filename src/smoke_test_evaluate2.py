"""Smoke test of src/evaluate2.py that never reads real targets: load_task_frames is replaced by a version
whose targets are random numbers (official ids and folds kept, so leader files can be aligned), and
fake prediction files are written to temporary A1/A2 directories."""
import json, tempfile, pathlib
import numpy as np
import src.evaluate2 as ev2
from src import data as D
from src.design import TASK_TYPE

_REAL = D.load_task_frames


def fake(task):
    df, folds = _REAL(task)
    df = df.copy()
    rng = np.random.default_rng(1)
    df["target"] = rng.normal(1, 1, len(df)) if TASK_TYPE[task] == "regression" else (rng.random(len(df)) > 0.5)
    return df, folds


def main():
    ev2.load_task_frames = fake
    ev2.B = 100
    import src.evaluate as ev1
    ev1.B = 100
    tmp = pathlib.Path(tempfile.mkdtemp())
    ev2.A2, ev2.A1 = tmp / "a2", tmp / "a1"
    rng = np.random.default_rng(0)
    for root in (ev2.A2, ev2.A1):
        for t in D.TASKS:
            df, folds = fake(t)
            (root / t).mkdir(parents=True)
            for k, f in enumerate(folds):
                n = len(f["test"])
                P = {a: rng.random(n) for a in ("stack", "stack_local", "best_single", "single_local", "mean_ensemble", "single__et_F1")}
                np.savez(root / t / f"fold{k}_predictions.npz", ids=np.array(f["test"]), **P)
                json.dump({"best_single": "et_F1", "stack_weights": {"et_F1": 1.0}, "sel_stack_local": {"alpha": 0},
                           "thresholds": {a: 0.5 for a in P}, "tuned": {}, "expert_seconds": {}},
                          open(root / t / f"fold{k}_meta.json", "w"))
    ev2.main(write_submission=False)
    print("smoke test OK", tmp)


if __name__ == "__main__":
    main()
