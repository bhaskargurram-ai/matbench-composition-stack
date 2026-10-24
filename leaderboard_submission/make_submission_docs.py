"""Generate info.json, README.md and PR_BODY.md for the Matbench leaderboard submission folder.

All numbers are read from files (the submission's results.json.gz, results/final_a2 timing/meta files and the
local DFT-feature cache); nothing is typed by hand. Run from the repository root:
    python leaderboard_submission/make_submission_docs.py
"""
from __future__ import annotations
import glob
import gzip
import json
import os
import pathlib
import statistics
from importlib.metadata import version

import pandas as pd
from matbench.bench import MatbenchBenchmark

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "leaderboard_submission" / "matbench_v0.1_DFTStack"
A2 = ROOT / "results" / "final_a2"
TASKS = ["matbench_steels", "matbench_expt_gap", "matbench_expt_is_metal", "matbench_glass"]
REPO = "https://github.com/bhaskargurram-ai/matbench-composition-stack"

PKGS = ["matbench", "matminer", "pymatgen", "monty", "scikit-learn", "numpy", "scipy", "pandas", "lightgbm",
        "xgboost", "catboost", "optuna", "joblib", "CBFV"]
REQS = [f"{p}=={version(p)}" for p in PKGS]

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
    "@article{jain2013mp,\n  author = {Anubhav Jain and Shyue Ping Ong and Geoffroy Hautier and Wei Chen and William Davidson Richards "
    "and Stephen Dacek and Shreyas Cholia and Dan Gunter and David Skinner and Gerbrand Ceder and Kristin A. Persson},\n"
    "  title = {Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},\n"
    "  journal = {APL Materials},\n  year = {2013},\n  volume = {1},\n  pages = {011002},\n  doi = {10.1063/1.4812323}\n}",
    "@unpublished{gurram_dftstack_2026,\n  author = {Bhaskar Gurram},\n"
    "  title = {Reusing computed density functional theory data to predict experimentally measured materials properties from composition: a pre-registered validation case study},\n"
    "  note = {Manuscript in preparation for journal submission; preprint to follow. Placeholder entry: no DOI or preprint identifier yet; to be updated on publication},\n"
    "  year = {2026}\n}",
]
A1 = ROOT / "results" / "final"
BOARD_METRIC = {"matbench_steels": "MAE", "matbench_expt_gap": "MAE",
                "matbench_expt_is_metal": "balanced accuracy", "matbench_glass": "balanced accuracy"}
# design choices fixed in dftstack/frozen_config2.py / experts2.py from the fold-0 design phase (DESIGN_LOG_A2.md)
FIXED_CHOICES = ("the feature-block set BLOCKS2; for matbench_expt_gap the hurdle experts and the removal of the "
                 "element-fraction extra-trees expert; the Optuna budget of 20 trials; dropping the random-forest expert; "
                 "restricting CatBoost to the F1 feature set; the fixed SVC C and gamma")


def fmt(task, v):
    return f"{v:.2f}" if task == "matbench_steels" else f"{v:.4f}"


def per_fold(csv, col_by_type):
    """Board-metric value of the final arm (stack_local) per fold, from a metrics_per_fold.csv."""
    d = pd.read_csv(csv)
    d = d[d["arm"] == "stack_local"]
    out = {}
    for t in TASKS:
        col = col_by_type["regression" if BOARD_METRIC[t] == "MAE" else "classification"]
        s = d[d["task"] == t].set_index("fold")[col]
        out[t] = (float(s.loc[0]), float(s.mean()), [float(s.loc[k]) for k in range(5)])
    return out


def prob_auc():
    """Probability ROC-AUC of the final arm (sklearn roc_auc_score on the test-fold probabilities), A2 and A1."""
    out = {}
    for lab, folder in (("A2", A2), ("A1", A1)):
        s = json.load(open(folder / "summary.json"))
        out[lab] = {t: (s[t]["stack_local"]["rocauc"]["mean"], s[t]["stack_local"]["rocauc"]["std"])
                    for t in ("matbench_expt_is_metal", "matbench_glass")}
    return out


def prob_variant_board():
    """What matbench computes for the probabilities variant (it thresholds probabilities at 0.5)."""
    raw = open(A2 / "matbench_submission" / "probabilities" / "results.json.gz", "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    mb = MatbenchBenchmark.from_dict(json.loads(raw))
    return {t.dataset_name: t.scores["rocauc"]["mean"] for t in mb.tasks if t.metadata.task_type != "regression"}


def nan_counts():
    return json.load(open(A2 / "imputation_nan_counts.json"))


def scores():
    # upstream matbench (main) reads *.gz with gzip.open; the pinned matbench 0.6 wheel cannot, so load the same way
    mb = MatbenchBenchmark.from_dict(json.loads(gzip.open(SUB / "results.json.gz").read().decode("utf-8")))
    out = {}
    for t in mb.tasks:
        s = t.scores
        if t.metadata.task_type == "regression":
            out[t.dataset_name] = ("MAE", s["mae"]["mean"], s["mae"]["std"], s["rmse"]["mean"])
        else:
            out[t.dataset_name] = ("ROC-AUC as computed by matbench from hard labels (= balanced accuracy)",
                                   s["rocauc"]["mean"], s["rocauc"]["std"], s["f1"]["mean"])
    return mb, out


def timing():
    rows = [json.loads(l) for l in open(A2 / "timing.jsonl") if l.strip()]
    last = {}
    for r in rows:                       # last successful record per (task, fold)
        if r.get("returncode") == 0:
            last[(r["task"], r["fold"])] = r
    tot_h = sum(r["wall_s"] for r in last.values()) / 3600
    med = {t: statistics.median(r["wall_s"] for (tt, _), r in last.items() if tt == t) / 60 for t in TASKS}
    w = next(iter(last.values()))
    return tot_h, med, w["workers"], w["threads_per_worker"]


def gate_kept():
    metas = sorted(glob.glob(str(A2 / "*" / "fold*_meta.json")))
    kept = sum(bool(json.load(open(m)).get("gate_stack_local", {}).get("kept")) for m in metas)
    return kept, len(metas)


def dft_coverage():
    cov = {}
    for t in TASKS:
        p = ROOT / "data" / "cache" / f"{t}_features_dft.pkl"
        if p.exists():
            X = pd.read_pickle(p)
            cov[t] = (int(X["dft_matched"].sum()), len(X))
    return cov


def main():
    mb, sc = scores()
    assert mb.is_valid and mb.is_recorded
    tot_h, med, nw, nth = timing()
    kept, nmeta = gate_kept()
    cov = dft_coverage()
    cov_txt = "; ".join(f"{t}: {m}/{n} ({100 * m / n:.0f}%)" for t, (m, n) in cov.items())
    med_txt = ", ".join(f"{t} {med[t]:.0f} min" for t in TASKS)
    pf2 = per_fold(A2 / "metrics_per_fold.csv", {"regression": "board_metric", "classification": "board_metric"})
    pf1 = per_fold(A1 / "metrics_per_fold.csv", {"regression": "mae", "classification": "balanced_accuracy_innerthr"})
    pauc, pvb, nans = prob_auc(), prob_variant_board(), nan_counts()
    fold0_txt = "; ".join(f"{t} {fmt(t, pf2[t][0])} vs {fmt(t, pf2[t][1])} {BOARD_METRIC[t]}" for t in TASKS)
    a1_txt = "; ".join(f"{t} {fmt(t, pf1[t][1])} {BOARD_METRIC[t]}" for t in TASKS)
    a1_auc = ", ".join(f"{t} {pauc['A1'][t][0]:.4f}" for t in pauc["A1"])
    a2_auc = ", ".join(f"{t} {pauc['A2'][t][0]:.4f} ± {pauc['A2'][t][1]:.4f}" for t in pauc["A2"])
    pvb_txt = ", ".join(f"{t} {v:.4f}" for t, v in pvb.items())
    nan_txt = "; ".join(
        f"{t}: CBFV {n['cbfv_oliynyk']['nan_cells']} of {n['n_rows'] * n['cbfv_oliynyk']['shape'][1]} cells "
        f"({n['cbfv_oliynyk']['rows_with_nan']} rows), extra block {n['extra']['nan_cells']} of "
        f"{n['n_rows'] * n['extra']['shape'][1]} cells ({n['extra']['rows_with_nan']} rows)" for t, n in nans.items())
    p0 = next(t for t in mb.tasks if t.dataset_name == "matbench_steels").results["fold_0"]["parameters"]
    cpu = p0["cpu_model"]

    algorithm_long = (
        "DFT-Stack: a cross-fitted, simplex-weighted stack of composition experts for the four Matbench v0.1 composition "
        "tasks, using Materials Project DFT data as transfer-learning input features. "
        "EXTERNAL DATA DISCLOSURE: besides the formula-derived descriptors, every expert receives a block of Materials "
        "Project PBE DFT features taken from matminer's matbench_mp_gap (PBE band gap) and matbench_mp_e_form (PBE "
        "formation energy) datasets, matched to each task composition by reduced formula (most stable polymorph by "
        "formation energy, plus polymorph aggregates; unmatched compositions get missing values and a dft_matched = 0 "
        "indicator). These DFT values are STRUCTURE-DERIVED: every Materials Project entry is a DFT calculation on a "
        "crystal structure, and per reduced formula we take the entry (polymorph) with the lowest formation energy per "
        "atom plus aggregates over all polymorphs of that formula (min/median/max gap, metallic fraction, number of "
        "polymorphs). The entry is therefore 'DFT, structure-derived, matched by formula', and the DFT gap / metal flag "
        "are close proxies of the matbench_expt_gap and matbench_expt_is_metal targets. No experimental external "
        "labels are used: the only experimental labels seen are the task's own training-fold labels. "
        "Features: matminer/pymatgen composition descriptors (Magpie and other ElementProperty presets, element "
        "fractions, Stoichiometry, ValenceOrbital, IonProperty, BandCenter, AtomicOrbitals, Meredig, Miedema, "
        "YangSolidSolution, AtomicPackingEfficiency, WenAlloys, TMetalFraction) and CBFV Oliynyk statistics; all are "
        "label-free functions of the formula. "
        "Experts: Optuna-tuned LightGBM, XGBoost, CatBoost, extra trees (two feature sets and element fractions), "
        "kernel ridge (regression) or RBF SVM (classification), k-nearest neighbours and an MLP; for matbench_expt_gap "
        "additionally hurdle experts for the zero-inflated band gap (a classifier for gap > 0 combined with a regressor "
        "fitted on the non-zero gaps). "
        "Protocol: nested cross-validation inside each official training fold. Five inner folds produce out-of-fold "
        "predictions for every expert; hyperparameters, the non-negative stack weights (summing to one; L1 loss for "
        "regression, Brier score for classification) and an optional novelty-damped local out-of-fold residual "
        "correction are all chosen on these inner folds, and the correction is kept only if a further nested inner-CV "
        f"check confirms it (it was kept in {kept} of {nmeta} task-folds). "
        "FOLD USE, STATED EXACTLY. Fold-internal (fitted on the training rows T_k of each official fold k only, "
        "test rows only predicted): the Optuna hyperparameters, the simplex stack weights, the correction gate and "
        "the classification thresholds. NOT fold-internal for folds 1-4: the coarse design choices that are fixed in "
        f"the shipped configuration ({FIXED_CHOICES}) were made in a design phase on fold 0's training rows (inner "
        "pseudo-folds, no fold-0 test rows). Fold 0's training rows contain the test rows of folds 1-4, so for folds "
        "1-4 these fixed choices were informed by rows that are later test rows; fold 0 is the only fold whose every "
        f"choice was made without its test rows. Fold-0 vs 5-fold mean of the recorded entry: {fold0_txt}. "
        "Classification: hard labels are recorded at a decision threshold that maximises balanced accuracy on the "
        "training-fold out-of-fold predictions (inner-CV threshold), so the board's ROC-AUC for these tasks equals "
        "balanced accuracy. "
        "The configuration was frozen (SHA-256 hashes of the code) before the official test folds were predicted "
        "with it, and the test folds were predicted once with this configuration."
    )
    notes = (
        f"CPU-only (no GPU): {nw} parallel workers x {nth} threads on one {cpu} workstation (16 cores / 24 threads); "
        f"{tot_h:.1f} worker-hours for all 20 task-folds; median wall time per fold: {med_txt}. Per-fold compute is "
        "recorded in the parameters of every fold in results.json.gz (wall_time_s, n_workers, threads_per_worker, "
        "cpu_threads_used, cpu_model, gpu, expert_fit_seconds per expert). "
        f"DFT-block coverage (compositions matched by reduced formula): {cov_txt}. "
        "PRIOR RUN ON THE OFFICIAL FOLDS: an earlier pre-registered configuration (Amendment 1: no DFT block, no "
        "hurdle experts, smaller expert pool, fully fold-internal candidate selection) was scored on the official "
        f"test folds on 2026-10-18, before this configuration (Amendment 2) was designed and pre-registered. Its "
        f"results, which are reported in the manuscript and in the research repository (results/final/): {a1_txt}; "
        f"probability ROC-AUC {a1_auc}. The present configuration was then scored once; there was no further "
        "iteration on test scores. "
        "LABELS VARIANT: results.json.gz records hard labels for the two classification tasks (the 'labels' variant). "
        "A 'probabilities' variant (same predictions, probabilities recorded) was written at the same scoring moment "
        "and is kept in the research repository; recording labels was fixed in the pre-registered Amendment-2 "
        "protocol before scoring, because the board compares label-based entries by balanced accuracy. For reference, "
        f"the probability ROC-AUC of the same predictions is {a2_auc}, and matbench itself would score the probabilities "
        f"file (thresholded at 0.5) at {pvb_txt}. "
        "TRANSDUCTIVE BUT LABEL-FREE IMPUTATION: two feature builders fill rare missing descriptor values with column "
        "medians computed over all rows of a task (training and test rows together; no labels): CBFV generate_features "
        f"for the Oliynyk block and the 'extra' matminer block in dftstack/data.py. Cells filled, per task: {nan_txt}. "
        "Columns that are entirely missing are dropped on the same basis. "
        "matbench==0.6 was installed with --no-deps because its scikit-learn==1.0 / monty pins do not build on "
        "Python 3.12; the newer versions listed in requirements were used. "
        "The folder is self-contained: run_submission.py plus the dftstack/ package (verbatim copies of the frozen "
        "research modules) reproduce results.json.gz and record every fold with task.record; the Matbench tasks and "
        "the matminer DFT datasets are downloaded on first use. "
        f"Full research repository (analysis plan, freeze hashes, per-fold predictions, ablations): {REPO}. "
        "The matbench 0.6 wheel writes plain JSON even for a .gz filename; the file here is the "
        "byte-identical JSON gzip-compressed, which is what MatbenchBenchmark.from_file on the current main branch reads "
        "(run_submission.py compresses its output the same way)."
    )
    info = {"authors": "Bhaskar Gurram", "algorithm": "DFT-Stack", "algorithm_long": algorithm_long,
            "bibtex_refs": BIB, "notes": notes, "requirements": {"python": REQS}}
    (SUB / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (SUB / "requirements.txt").write_text("\n".join(REQS) + "\n", encoding="utf-8")

    readme = f"""# DFT-Stack (matbench_v0.1)

Cross-fitted, simplex-weighted stack of composition experts with a disclosed block of Materials Project PBE DFT
features (matminer `matbench_mp_gap` / `matbench_mp_e_form`, matched by reduced formula), evaluated on the four
Matbench v0.1 composition tasks with nested cross-validation inside each official training fold.
See `info.json` for the full description and the external-data disclosure.

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as hard labels). The
  `parameters` of every fold carry the compute of the original run: `wall_time_s`, `n_workers`,
  `threads_per_worker`, `cpu_threads_used`, `cpu_model`, `gpu`, `expert_fit_seconds` (per expert), `pipeline`, `seed`.
- `info.json`: metadata, references, notes (fold use stated exactly, prior-run disclosure, labels variant,
  imputation note), pinned requirements.
- `run_submission.py`: reproduces the run and records every fold with `task.record` (with the same per-fold
  compute fields, measured on the rerun machine), then writes `results_rerun.json.gz`
  (`python run_submission.py [task ...]`). `results_rerun.json.gz` is only the output name of such a user rerun;
  it is not shipped in this folder, and no complete independent rerun has been performed so far.
- `dftstack/`: verbatim copies of the frozen research modules (pipeline, experts, feature blocks, DFT
  feature builder, frozen configuration). Their docstrings refer to the analysis plan and design logs, which are
  in the research repository ({REPO}).
- `requirements.txt`: the same pinned versions as `info.json`.

## Reproduce
```
pip install -r requirements.txt   # matbench==0.6 needs --no-deps on Python 3.12, see info.json notes
python run_submission.py
```
CPU-only. Data are downloaded on first use into `./data/cache/`; expert predictions are cached in
`./rerun_cache/` so that an interrupted run resumes.
"""
    (SUB / "README.md").write_text(readme, encoding="utf-8")

    rows = "\n".join(
        f"| `{t}` | {sc[t][0]} | {sc[t][1]:.4f} ± {sc[t][2]:.4f} |" for t in TASKS)
    pr = f"""<!-- Draft PR description (not submitted). Label to request: new_benchmark -->
# New benchmark: DFT-Stack on the 4 matbench_v0.1 composition tasks

**Label:** `new_benchmark` (please add it; I cannot set labels on this repository)

## External data disclosure (please read first)
This entry **uses external computed data**: Materials Project PBE DFT band gaps and formation energies, from
matminer's `matbench_mp_gap` and `matbench_mp_e_form` datasets, matched to each composition by reduced formula
and used as transfer-learning input features. **No experimental external labels** are used. On
`matbench_expt_gap` and `matbench_expt_is_metal` most compositions are matched; on `matbench_steels` and
`matbench_glass` almost none are ({cov_txt}). The entry is therefore not a like-for-like comparison with
composition-only entries that use no external data, and we would welcome it being flagged as such on the board.

## Brief description of the algorithm
A cross-fitted, simplex-weighted stack of composition experts (Optuna-tuned LightGBM, XGBoost, CatBoost, extra
trees, kernel ridge / RBF SVM, k-NN, MLP; hurdle experts for the zero-inflated `matbench_expt_gap`) on
label-free matminer/CBFV composition descriptors plus the DFT block above. The DFT values are structure-derived
(each Materials Project entry is a DFT calculation on a crystal structure; per reduced formula the most stable
polymorph plus polymorph aggregates are used), so the DFT gap / metal flag are close proxies of the
`matbench_expt_gap` / `matbench_expt_is_metal` targets.

**Fold use, stated exactly.** Fold-internal (training rows of each official fold only): the Optuna hyperparameters,
the simplex stack weights, the optional local residual correction and its gate (kept in {kept} of {nmeta}
task-folds) and the classification decision thresholds; test folds are only predicted and recorded. Not
fold-internal for folds 1-4: the coarse design choices fixed in the shipped configuration ({FIXED_CHOICES}) were
made in a design phase on fold 0's training rows, which contain the test rows of folds 1-4. Fold 0 is the only
fold whose every choice was made without its test rows; fold-0 vs 5-fold mean: {fold0_txt}.
An earlier pre-registered configuration (Amendment 1, no DFT block) was scored on the official folds on
2026-10-18 before this one was designed; its results ({a1_txt}) are reported in the manuscript and kept in the
research repository. Classification tasks are recorded as hard labels at the inner-CV balanced-accuracy
threshold (pre-specified before scoring), so the board's ROC-AUC equals balanced accuracy; the probability
ROC-AUC of the same predictions is {a2_auc}. The configuration was frozen by SHA-256 hash before the test folds
were predicted with it, and they were predicted once.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score |
|---|---|---|
{rows}

## Included files (`benchmarks/matbench_v0.1_DFTStack/`)
- `results.json.gz`, `info.json`: required files (`MatbenchBenchmark.from_file(...)`: `is_valid=True`,
  `is_recorded=True` for all 4 tasks).
- `run_submission.py`: source that reproduces the run with `task.record` calls.
- `dftstack/`: the pipeline, experts, feature blocks, DFT feature builder and frozen configuration (small `.py`
  files) imported by `run_submission.py`.
- `README.md`, `requirements.txt`.

## Compute
CPU-only ({cpu}, 16 cores / 24 threads, no GPU), {nw} workers x {nth} threads; {tot_h:.1f} worker-hours for all
20 task-folds. Per-fold wall time, threads, CPU model and per-expert fit seconds are recorded in the `parameters`
of every fold in `results.json.gz`.

## Paper and code
Manuscript in preparation for journal submission; preprint to follow (no DOI yet). Research repository: {REPO}
"""
    (ROOT / "leaderboard_submission" / "PR_BODY.md").write_text(pr, encoding="utf-8")
    print("wrote info.json, requirements.txt, README.md, PR_BODY.md")
    for t in TASKS:
        print(t, sc[t])


if __name__ == "__main__":
    main()
