"""Build the Matbench leaderboard folder for Amendment 4 (composition-only): leaderboard_submission/matbench_v0.1_XStack.

Reporting only; run AFTER `python -m src.evaluate4 --submission`. It
  1. copies the frozen research modules verbatim into ./xstack (no DFT module is shipped);
  2. takes results/final_a4/matbench_submission/labels/results.json.gz, adds the per-fold compute and selection
     parameters from results/final_a4/<task>/fold<k>_meta.json, sets user_metadata.algorithm = "XStack", checks that
     every recorded prediction and every score is unchanged, and writes results.json.gz (gzip, mtime=0);
  3. writes info.json, requirements.txt, README.md and ../PR_BODY.md. All numbers are read from files.
Run from the repository root:  python leaderboard_submission/build_a4_submission.py
"""
from __future__ import annotations
import gzip
import json
import pathlib
import shutil
from importlib.metadata import version

import pandas as pd
from matbench.bench import MatbenchBenchmark
from matbench.util import hash_dictionary

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "leaderboard_submission" / "matbench_v0.1_XStack"
A4, A3, A2, A1 = (ROOT / "results" / d for d in ("final_a4", "final_a3", "final_a2", "final"))
TASKS = ["matbench_steels", "matbench_expt_gap", "matbench_expt_is_metal", "matbench_glass"]
REPO = "https://github.com/bhaskargurram-ai/matbench-composition-stack"
ALGO = "XStack"
PIPELINE = "XStack (Amendment 4: composition-only, train-only imputation, fold-internal candidate selection)"
MODULES = ["data", "design", "experts", "experts2", "experts4", "features4", "features_more", "features_all",
           "frozen_config2", "frozen_config3", "frozen_config4", "pipeline", "pipeline2", "pipeline4", "run_final3",
           "run_final4"]
PKGS = ["matbench", "matminer", "pymatgen", "monty", "scikit-learn", "numpy", "scipy", "pandas", "lightgbm",
        "xgboost", "catboost", "optuna", "joblib", "CBFV"]
BOARD_METRIC = {"matbench_steels": "MAE", "matbench_expt_gap": "MAE",
                "matbench_expt_is_metal": "balanced accuracy", "matbench_glass": "balanced accuracy"}
BIB = [
    "@article{dunn2020matbench,\n  author = {Alexander Dunn and Qi Wang and Alex Ganose and Daniel Dopp and Anubhav Jain},\n"
    "  title = {Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm},\n"
    "  journal = {npj Computational Materials},\n  year = {2020},\n  volume = {6},\n  pages = {138},\n"
    "  doi = {10.1038/s41524-020-00406-3}\n}",
    "@article{ward2018matminer,\n  author = {Logan Ward and Alexander Dunn and Alireza Faghaninia and Nils E.R. Zimmermann and "
    "Saurabh Bajaj and Qi Wang and Joseph Montoya and Jiming Chen and Kyle Bystrom and Maxwell Dylla and Kyle Chard and "
    "Mark Asta and Kristin A. Persson and G. Jeffrey Snyder and Ian Foster and Anubhav Jain},\n"
    "  title = {Matminer: An open source toolkit for materials data mining},\n  journal = {Computational Materials Science},\n"
    "  year = {2018},\n  volume = {152},\n  pages = {60--69},\n  doi = {10.1016/j.commatsci.2018.05.018}\n}",
]


def fmt(task, v):
    return f"{v:.2f}" if task == "matbench_steels" else f"{v:.4f}"


def load_mb(path):
    raw = pathlib.Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    return MatbenchBenchmark.from_dict(d), d


def per_fold_mean(csv, cols, arm="stack_local"):
    d = pd.read_csv(csv)
    d = d[d["arm"] == arm]
    return {t: float(d[d["task"] == t][cols["regression" if BOARD_METRIC[t] == "MAE" else "classification"]].mean())
            for t in TASKS}


def copy_modules():
    pkg = SUB / "xstack"
    pkg.mkdir(parents=True, exist_ok=True)
    for old in pkg.glob("*.py"):
        old.unlink()
    for m in MODULES:
        shutil.copy2(ROOT / "src" / f"{m}.py", pkg / f"{m}.py")
    (pkg / "__init__.py").write_text(
        '"""Verbatim copies of the frozen research modules of the XStack entry (Amendment 4, composition-only).\n'
        "The DFT feature builder of the earlier, DFT-informed runs (external_dft.py) is deliberately not shipped;\n"
        'features_all.py imports it only inside a branch that this entry never reaches."""\n', encoding="utf-8")
    assert not (pkg / "external_dft.py").exists()


# The PC slept during the run (Windows Kernel-Power event 42 at 2026-11-17 19:30:30, resume at 2026-11-18 09:46:52
# local). Jobs running at that time resumed unchanged; their raw timers include the sleep. The raw values stay in
# results/final_a4; the submission reports active time = raw - sleep and keeps the raw value next to it.
SLEEP_S = 51382.0
SLEPT = {("matbench_glass", 4), ("matbench_expt_is_metal", 0), ("matbench_expt_is_metal", 1),
         ("matbench_expt_is_metal", 2), ("matbench_expt_is_metal", 3)}


def active_wall(task, k, meta):
    w = float(meta["compute"]["selection_job"]["wall_s"])
    return w - SLEEP_S if (task, k) in SLEPT else w


def fold_params(meta, task, k):
    sel = meta["compute"]["selection_job"]
    ds = meta["design_selection"]
    es = {e: float(s) for e, s in meta["expert_seconds"].items()}
    p = {"pipeline": PIPELINE, "seed": 0, "wall_time_s": round(active_wall(task, k, meta), 1),
         "n_workers": int(sel["n_workers"]), "threads_per_worker": int(sel["threads_per_worker"]),
         "cpu_threads_used": 4, "cpu_count": sel["cpu_count"], "cpu_model": sel["cpu_model"], "gpu": "none"}
    if (task, k) in SLEPT:
        assert float(sel["wall_s"]) > SLEEP_S
        big = [e for e, s in es.items() if s > SLEEP_S]
        assert len(big) <= 1
        for e in big:
            es[e] -= SLEEP_S
        p["wall_time_s_raw"] = round(float(sel["wall_s"]), 1)
        p["timing_note"] = (f"the machine was asleep for {SLEEP_S:.0f} s during this job; wall_time_s = raw - sleep"
                            + (f"; the same was subtracted from expert_fit_seconds['{big[0]}']" if big else ""))
    else:
        assert all(s < 20000 for s in es.values()) and float(sel["wall_s"]) < 20000
    p.update({"expert_fit_seconds": {e: round(s, 1) for e, s in es.items()},
              "selected_candidate": ds["selected"],
              "candidate_inner_cv_scores": {r["candidate"]: round(float(r["stack_oof_score"]), 6) for r in ds["candidates"]},
              "imputation": meta["imputation"]})
    return p


def build_results(metas):
    mb, d0 = load_mb(A4 / "matbench_submission" / "labels" / "results.json.gz")
    assert mb.is_valid and mb.is_recorded
    before = {t.dataset_name: ({k: dict(v["data"]) for k, v in t.results.items()}, dict(t.scores)) for t in mb.tasks}
    for task in mb.tasks:
        for k in range(5):
            task.results[f"fold_{k}"]["parameters"] = fold_params(metas[task.dataset_name][k], task.dataset_name, k)
    mb.add_metadata({"algorithm": ALGO})
    d = mb.as_dict()
    d["datestamp"] = d0["datestamp"]
    d.pop("hash")
    d["hash"] = hash_dictionary(d)
    with gzip.GzipFile(SUB / "results.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(d).encode("utf-8"))
    mb2, _ = load_mb(SUB / "results.json.gz")
    assert mb2.is_valid and mb2.is_recorded
    for t in mb2.tasks:
        data0, scores0 = before[t.dataset_name]
        assert all(dict(v["data"]) == data0[fk] for fk, v in t.results.items()) and dict(t.scores) == scores0
    return mb2


def main():
    metas = {t: [json.load(open(A4 / t / f"fold{k}_meta.json")) for k in range(5)] for t in TASKS}
    assert not any("dft" in r["candidate"].lower() or "F_red" in r["candidate"]
                   for t in TASKS for m in metas[t] for r in m["design_selection"]["candidates"])
    copy_modules()
    mb = build_results(metas)
    sc = {}
    for t in mb.tasks:
        s = t.scores
        sc[t.dataset_name] = (("MAE", s["mae"]["mean"], s["mae"]["std"]) if t.metadata.task_type == "regression" else
                              ("ROC-AUC as computed by matbench from hard labels (= balanced accuracy)",
                               s["rocauc"]["mean"], s["rocauc"]["std"]))
    stats = json.load(open(A4 / "stats.json"))
    summ = json.load(open(A4 / "summary.json"))
    bm = {"regression": "board_metric", "classification": "board_metric"}
    prev = {"A3": per_fold_mean(A3 / "metrics_per_fold.csv", bm), "A2": per_fold_mean(A2 / "metrics_per_fold.csv", bm),
            "A1": per_fold_mean(A1 / "metrics_per_fold.csv", {"regression": "mae", "classification": "balanced_accuracy_innerthr"}),
            "A2noDFT": {t: stats[t]["a4_vs_a2nodft"]["prev_mean"] for t in TASKS}}
    txt = lambda lab: "; ".join(f"{t} {fmt(t, prev[lab][t])}" for t in TASKS)
    a4_txt = "; ".join(f"{t} {fmt(t, sc[t][1])}" for t in TASKS)
    kept = sum(bool(m["gate_stack_local"]["kept"]) for t in TASKS for m in metas[t])
    sel_txt = "; ".join(f"{t}: " + ", ".join(f"fold {k} {m['design_selection']['selected']}" for k, m in enumerate(metas[t]))
                        for t in TASKS)
    cand_txt = "; ".join(f"{t}: {len(metas[t][0]['design_selection']['candidates'])}" for t in TASKS)
    nans = json.load(open(A4 / "raw_block_nan_counts.json"))
    nan_txt = "; ".join(f"{t}: {n['extra']['nan_cells'] + n['cbfv_oliynyk']['nan_cells']} cells in "
                        f"{n['n_rows']} rows x {n['extra']['shape'][1] + n['cbfv_oliynyk']['shape'][1]} columns"
                        for t, n in nans.items())
    auc_txt = ", ".join(f"{t} {summ[t]['stack_local']['rocauc']['mean']:.4f} ± {summ[t]['stack_local']['rocauc']['std']:.4f}"
                        for t in ("matbench_expt_is_metal", "matbench_glass"))
    walls = {t: [active_wall(t, k, m) for k, m in enumerate(metas[t])] for t in TASKS}
    tot_h = sum(sum(w) for w in walls.values()) / 3600
    wall_txt = ", ".join(f"{t} {sum(w) / len(w) / 60:.0f} min" for t, w in walls.items())
    cpu = metas[TASKS[0]][0]["compute"]["selection_job"]["cpu_model"]
    nw = metas[TASKS[0]][0]["compute"]["selection_job"]["n_workers"]
    reqs = [f"{p}=={version(p)}" for p in PKGS]

    algorithm_long = (
        "XStack: a cross-fitted, simplex-weighted stack of composition experts for the four Matbench v0.1 composition "
        "tasks. COMPOSITION-ONLY: the only inputs are label-free descriptors computed from each row's own formula; no "
        "external property data of any kind is used (no DFT values, no pretrained model, no data from other Matbench "
        "tasks). The only labels seen are the task's own training-fold labels. "
        "Features: matminer/pymatgen composition descriptors (Magpie and other ElementProperty presets, element "
        "fractions, Stoichiometry, ValenceOrbital, IonProperty, BandCenter, AtomicOrbitals, Meredig, Miedema, "
        "YangSolidSolution, AtomicPackingEfficiency, WenAlloys, TMetalFraction) and CBFV Oliynyk statistics. "
        "Experts: Optuna-tuned LightGBM, XGBoost, CatBoost, extra trees (two feature sets and element fractions), "
        "kernel ridge (regression) or RBF SVM (classification), k-nearest neighbours and an MLP; for matbench_expt_gap "
        "additionally hurdle experts for the zero-inflated band gap (a classifier for gap > 0 combined with a regressor "
        "fitted on the non-zero gaps). "
        "Protocol: nested cross-validation inside each official training fold T_k; the test rows are only predicted and "
        "recorded. Five inner folds give out-of-fold predictions for every expert. Chosen on T_k only: the Optuna "
        "hyperparameters; the non-negative stack weights summing to one (L1 loss for regression, Brier score for "
        "classification); an optional novelty-damped local out-of-fold residual correction, kept only if a further "
        f"nested inner-CV check confirms it (kept in {kept} of 20 task-folds); the classification threshold; and the "
        "feature-block set (F_min = Magpie + element fractions + 21 extra descriptors, or F_all = F_min + the remaining "
        "descriptor blocks) together with, for matbench_expt_gap, the use of the hurdle experts and of the "
        "element-fraction extra-trees expert. The last two are picked per fold from a small candidate set by the "
        f"cross-fitted inner-CV score of the uncorrected stack on T_k, ties to the simpler candidate (candidates: {cand_txt}). "
        f"Selected per fold: {sel_txt}. "
        "IMPUTATION: missing descriptor cells are filled with column medians computed inside every model fit from that "
        "fit's training rows only: T_k for the final refit, the inner-training part for every inner-CV fit (cross-fitting "
        "and every Optuna trial). No statistic is computed over test rows or over inner-validation rows. "
        "NOT fold-internal, disclosed: the candidate set itself and a few coarse settings (Optuna budget of 20 trials; "
        "random forest, CrabNet and MODNet dropped; CatBoost restricted to the smaller feature set; fixed SVC C and "
        "gamma) were decided in a design phase on fold 0's training rows; and earlier versions of the pipeline were "
        "scored on the official folds before this one (see notes). "
        "Classification: hard labels are recorded at the threshold that maximises balanced accuracy on the training-fold "
        "out-of-fold predictions, so the board's ROC-AUC for these tasks equals balanced accuracy."
    )
    notes = (
        f"CPU-only (no GPU). Every expert was fitted from scratch on one {cpu} workstation ({nw} parallel jobs x 4 "
        f"threads); no cached fit of an earlier run was reused. Total active job wall time {tot_h:.1f} h over the 20 "
        f"task-folds; the machine slept for {SLEEP_S / 3600:.1f} h during the run, five jobs resumed unchanged and their "
        f"recorded times have the sleep subtracted (raw value kept as wall_time_s_raw) "
        f"(mean per fold: {wall_txt}). Per-fold compute is recorded in the parameters of every fold in results.json.gz "
        "(wall_time_s, n_workers, threads_per_worker, cpu_threads_used, cpu_model, gpu, expert_fit_seconds per expert, "
        "selected_candidate, candidate_inner_cv_scores, imputation). "
        f"Cells that are missing before imputation, per task: {nan_txt}. A descriptor column is dropped only if it is "
        "missing in every row of the task (it then has no value anywhere). "
        "PRIOR RUNS ON THE OFFICIAL FOLDS, all pre-registered, each scored once, all reported in the "
        "research repository (board metric: MAE for steels and expt_gap, balanced accuracy for expt_is_metal and glass). "
        f"(1) Amendment 1 (2026-10-18; composition-only, smaller expert pool): {txt('A1')}. "
        f"(2) Amendment 2 (2026-10-19; USED EXTERNAL DATA: Materials Project PBE DFT band gaps and formation energies "
        f"matched by reduced formula; design choices fixed from fold 0's training rows): {txt('A2')}. "
        f"(3) Amendment 3 (2026-10-24; same external DFT data, design choices selected per fold): {txt('A3')}. "
        f"(4) A post-hoc ablation of Amendment 2 without the DFT block (2026-10-19): {txt('A2noDFT')}. "
        f"(5) The present entry, Amendment 4 (2026-11-17; no DFT block, train-only imputation): {a4_txt}. "
        "Amendments 2 and 3 are NOT composition-only and are not part of this entry: the DFT gap and metal flag are "
        "close proxies of the matbench_expt_gap and matbench_expt_is_metal targets. They were the earlier contents of "
        "this pull request and were withdrawn at the maintainers' request. Because of these earlier runs the design of "
        "this entry was not blind to test-fold results; no choice in Amendment 4 was made from test scores (its two "
        "changes are the ones the maintainers asked for). "
        "LABELS VARIANT: results.json.gz records hard labels for the two classification tasks; this was fixed in the "
        f"protocol before scoring. The probability ROC-AUC of the same predictions is {auc_txt}. "
        "REPRODUCIBILITY: same-machine reruns are bit-identical. Across CPU architectures XGBoost (tree_method=hist) "
        "and, on the small steels task, the Optuna-tuned LightGBM parameters were found not to be bitwise reproducible "
        "in an earlier check (Windows i7 vs Linux EPYC); the other experts agreed to 1e-9. "
        "matbench==0.6 was installed with --no-deps because its scikit-learn==1.0 / monty pins do not build on Python "
        "3.12. run_submission.py plus the xstack/ package (verbatim copies of the frozen research modules) reproduce "
        f"results.json.gz. Research repository (analysis plan with all amendments, freeze hashes, per-fold predictions): {REPO}. "
        "The matbench 0.6 wheel writes plain JSON even for a .gz filename; the file here is that JSON gzip-compressed."
    )
    info = {"authors": "Bhaskar Gurram", "algorithm": ALGO, "algorithm_long": algorithm_long, "bibtex_refs": BIB,
            "notes": notes, "requirements": {"python": reqs}}
    (SUB / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (SUB / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")

    (SUB / "README.md").write_text(f"""# XStack (matbench_v0.1)

Composition-only, cross-fitted, simplex-weighted stack of composition experts, evaluated on the four Matbench v0.1
composition tasks with nested cross-validation inside each official training fold. No external data: the inputs are
label-free descriptors of each row's own formula. Missing descriptor cells are imputed with medians of the training
rows of each fit, inner-CV fits included. See `info.json` for the full description and for the earlier runs on the
official folds (two of which used external DFT data and are not part of this entry).

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as hard labels). The
  `parameters` of every fold carry the compute (`wall_time_s`, `n_workers`, `threads_per_worker`, `cpu_threads_used`,
  `cpu_model`, `gpu`, `expert_fit_seconds`), the per-fold selection (`selected_candidate`,
  `candidate_inner_cv_scores`) and the imputation counts.
- `info.json`: metadata, references, notes, pinned requirements.
- `run_submission.py`: reproduces the run and records every fold with `task.record`
  (`python run_submission.py [task ...]`, output `results_rerun.json.gz`, which is not shipped).
- `xstack/`: verbatim copies of the frozen research modules. The code path of this entry is `run_final4.py`
  (per-fold candidate selection) -> `features4.py` (features without any cross-row fill) -> `experts4.py`
  (train-only median imputation around the `experts2.py` pool) -> `pipeline4.py` / `pipeline2.py` / `pipeline.py`
  (tuning, cross-fitting, stack, gated correction, thresholds) with `frozen_config4.py`. `run_final3.py`,
  `frozen_config3.py`, `features_all.py`, `design.py` and `experts.py` are imported for helper functions only. No DFT
  feature builder is shipped. Docstrings refer to the analysis plan in the research repository ({REPO}).
- `requirements.txt`: the same pinned versions as `info.json`.

## Reproduce
```
pip install -r requirements.txt   # matbench==0.6 needs --no-deps on Python 3.12, see info.json notes
python run_submission.py
```
CPU-only, about {tot_h:.0f} hours of job time at 4 threads per job. Data are downloaded on first use into
`./data/cache/`; expert predictions are cached in `./rerun_cache/` so that an interrupted run resumes.
""", encoding="utf-8")

    rows = "\n".join(f"| `{t}` | {sc[t][0]} | {sc[t][1]:.4f} ± {sc[t][2]:.4f} | {fmt(t, prev['A3'][t])} |" for t in TASKS)
    (ROOT / "leaderboard_submission" / "PR_BODY.md").write_text(f"""<!-- PR description. Label: new_benchmark -->
# New benchmark: XStack (composition-only) on the 4 matbench_v0.1 composition tasks

**Label:** `new_benchmark`

**Changed on 2026-11-17 at the maintainers' request.** This entry is now composition-only. The Materials Project DFT
feature block of the earlier versions of this PR is removed, the entry is renamed from DFT-Stack to XStack
(`benchmarks/matbench_v0.1_XStack/`), and missing descriptor cells are imputed from training rows only.

## Brief description of the algorithm
A cross-fitted, simplex-weighted stack of composition experts (Optuna-tuned LightGBM, XGBoost, CatBoost, extra
trees, kernel ridge / RBF SVM, k-NN, MLP; hurdle experts for the zero-inflated `matbench_expt_gap`) on label-free
matminer/CBFV composition descriptors. **No external data**: every input is computed from the row's own formula.

**Fold use, stated exactly.** Every choice is made inside the training rows of each official fold: the Optuna
hyperparameters, the simplex stack weights, the optional local residual correction and its gate (kept in {kept} of
20 task-folds), the classification thresholds, and the feature-block set (for `matbench_expt_gap` also the hurdle
experts and the element-fraction extra-trees expert), selected per fold from a small candidate set by the inner-CV
score of the stack on the training rows (candidates: {cand_txt}). Selected per fold: {sel_txt}.

**Imputation.** Missing descriptor cells ({nan_txt}) are filled with column medians computed inside every model fit
from that fit's training rows only, inner-CV fits and Optuna trials included.

**Disclosed.** The candidate set and a few coarse settings (Optuna budget, dropped experts, CatBoost feature set,
SVC C and gamma) come from a design phase on fold 0's training rows. Earlier versions were scored on the official
folds before this one, each pre-registered and scored once: Amendment 1 (composition-only, smaller pool; {txt('A1')});
Amendments 2 and 3 (with the external DFT block, the earlier contents of this PR; {txt('A2')} and {txt('A3')}); a
no-DFT ablation of Amendment 2 ({txt('A2noDFT')}). So this entry was not designed blind to test-fold results; its two
changes against Amendment 3 are the ones requested here. Classification is recorded as hard labels at the inner-CV
balanced-accuracy threshold, so the board's ROC-AUC equals balanced accuracy; the probability ROC-AUC of the same
predictions is {auc_txt}.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score (this entry, composition-only) | For reference: withdrawn DFT-informed version |
|---|---|---|---|
{rows}

## Included files (`benchmarks/matbench_v0.1_XStack/`)
- `results.json.gz`, `info.json`: required files (`is_valid=True`, `is_recorded=True` for all 4 tasks).
- `run_submission.py`: reproduces the run with `task.record` calls.
- `xstack/`: the pipeline modules imported by `run_submission.py` (no DFT feature builder).
- `README.md`, `requirements.txt`.

## Compute
CPU-only, no GPU. Every expert fitted from scratch on one {cpu} workstation ({nw} jobs x 4 threads), {tot_h:.1f} h of
job wall time in total; no cached fit of an earlier run reused. Per-fold wall time, threads, CPU model and
per-expert fit seconds are in the `parameters` of every fold in `results.json.gz`.

## Paper and code
No published paper or preprint is associated with this entry. Research repository (analysis plan with all
amendments, freeze hashes, per-fold predictions): {REPO}
""", encoding="utf-8")
    print("built", SUB)
    for t in TASKS:
        print(t, sc[t], "| A3", fmt(t, prev["A3"][t]), "| selected", [m["design_selection"]["selected"] for m in metas[t]])


if __name__ == "__main__":
    main()
