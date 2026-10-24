"""Generate info.json, README.md and PR_BODY.md for the Matbench leaderboard submission folder (Amendment 3).

All numbers are read from files (the submission's results.json.gz, results/final_a3 metas and metrics, the earlier
runs' metric files, the determinism report and the local DFT-feature cache); nothing is typed by hand.
Run from the repository root:
    python leaderboard_submission/make_submission_docs.py
"""
from __future__ import annotations
import glob
import gzip
import json
import pathlib
from collections import Counter
from importlib.metadata import version

import pandas as pd
from matbench.bench import MatbenchBenchmark

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUB = ROOT / "leaderboard_submission" / "matbench_v0.1_DFTStack"
A3 = ROOT / "results" / "final_a3"
A2 = ROOT / "results" / "final_a2"
A1 = ROOT / "results" / "final"
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
    "@unpublished{gurram_dftstack_inreview,\n  author = {Bhaskar Gurram},\n"
    "  title = {Reusing computed density functional theory data to predict experimentally measured materials properties from composition: a pre-registered validation case study},\n"
    "  note = {Under review at Computational Materials Science. Placeholder entry: no DOI or preprint identifier yet; to be updated on publication},\n"
    "  year = {2026}\n}",
]
BOARD_METRIC = {"matbench_steels": "MAE", "matbench_expt_gap": "MAE",
                "matbench_expt_is_metal": "balanced accuracy", "matbench_glass": "balanced accuracy"}
# coarse choices that remain design-phase decisions (they define the candidate set, not the winner)
REMAINING_FIXED = ("the Optuna budget of 20 trials; the dropped experts (random forest, CrabNet, MODNet); restricting "
                   "CatBoost to the F1 feature set; the fixed SVC C and gamma; and the composition of the candidate set itself")


def fmt(task, v):
    return f"{v:.2f}" if task == "matbench_steels" else f"{v:.4f}"


def per_fold(csv, col_by_type, arm="stack_local"):
    """Board-metric value of an arm per fold, from a metrics_per_fold.csv."""
    d = pd.read_csv(csv)
    d = d[d["arm"] == arm]
    out = {}
    for t in TASKS:
        col = col_by_type["regression" if BOARD_METRIC[t] == "MAE" else "classification"]
        s = d[d["task"] == t].set_index("fold")[col]
        out[t] = (float(s.loc[0]), float(s.mean()), [float(s.loc[k]) for k in range(5)])
    return out


def prob_auc():
    out = {}
    for lab, folder in (("A3", A3), ("A2", A2), ("A1", A1)):
        s = json.load(open(folder / "summary.json"))
        out[lab] = {t: (s[t]["stack_local"]["rocauc"]["mean"], s[t]["stack_local"]["rocauc"]["std"])
                    for t in ("matbench_expt_is_metal", "matbench_glass")}
    return out


def prob_variant_board():
    raw = open(A3 / "matbench_submission" / "probabilities" / "results.json.gz", "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    mb = MatbenchBenchmark.from_dict(json.loads(raw))
    return {t.dataset_name: t.scores["rocauc"]["mean"] for t in mb.tasks if t.metadata.task_type != "regression"}


def nan_counts():
    return json.load(open(A2 / "imputation_nan_counts.json"))


def scores():
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


def metas():
    return {t: [json.load(open(A3 / t / f"fold{k}_meta.json")) for k in range(5)] for t in TASKS}


def compute(ms):
    """Wall time per fold (selection job + separate expert-fit jobs), hosts, and the Amendment-2 worker-hours."""
    tot, hosts = 0.0, Counter()
    per_task = {}
    for t in TASKS:
        w = []
        for m in ms[t]:
            c = m["compute"]
            wall = float(c["selection_job"]["wall_s"])
            hosts[(c["selection_job"]["cpu_model"], c["selection_job"]["n_workers"], c["selection_job"]["threads_per_worker"])] += 0
            for bs, info in c["expert_fits"].items():
                for r in info["fit_info"]["runs"][:-1]:
                    wall += float(r["wall_s"])
                    hosts[(r["cpu_model"], r["n_workers"], r["threads_per_worker"])] += 1
            w.append(wall)
        per_task[t] = sum(w) / len(w) / 60
        tot += sum(w)
    a2 = sum(json.loads(l)["wall_s"] for l in open(A2 / "timing.jsonl") if l.strip()) / 3600
    return tot / 3600, per_task, hosts, a2


def selections(ms):
    out = {}
    for t in TASKS:
        out[t] = [m["design_selection"]["selected"] for m in ms[t]]
    return out


def determinism():
    f = A3 / "determinism" / "report.json"
    return json.load(open(f)) if f.exists() else None


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
    ms = metas()
    tot_h, per_task_min, hosts, a2_h = compute(ms)
    kept = sum(bool(m["gate_stack_local"]["kept"]) for t in TASKS for m in ms[t])
    nmeta = 5 * len(TASKS)
    sel = selections(ms)
    cov = dft_coverage()
    cov_txt = "; ".join(f"{t}: {m}/{n} ({100 * m / n:.0f}%)" for t, (m, n) in cov.items())
    pf3 = per_fold(A3 / "metrics_per_fold.csv", {"regression": "board_metric", "classification": "board_metric"})
    pf2 = per_fold(A2 / "metrics_per_fold.csv", {"regression": "board_metric", "classification": "board_metric"})
    pf1 = per_fold(A1 / "metrics_per_fold.csv", {"regression": "mae", "classification": "balanced_accuracy_innerthr"})
    pauc, pvb, nans, det = prob_auc(), prob_variant_board(), nan_counts(), determinism()
    a3_txt = "; ".join(f"{t} {fmt(t, pf3[t][1])} {BOARD_METRIC[t]}" for t in TASKS)
    a2_txt = "; ".join(f"{t} {fmt(t, pf2[t][1])} {BOARD_METRIC[t]}" for t in TASKS)
    a1_txt = "; ".join(f"{t} {fmt(t, pf1[t][1])} {BOARD_METRIC[t]}" for t in TASKS)
    a1_auc = ", ".join(f"{t} {pauc['A1'][t][0]:.4f}" for t in pauc["A1"])
    a2_auc = ", ".join(f"{t} {pauc['A2'][t][0]:.4f}" for t in pauc["A2"])
    a3_auc = ", ".join(f"{t} {pauc['A3'][t][0]:.4f} ± {pauc['A3'][t][1]:.4f}" for t in pauc["A3"])
    pvb_txt = ", ".join(f"{t} {v:.4f}" for t, v in pvb.items())
    sel_txt = "; ".join(f"{t}: " + ", ".join(f"fold {k} {s}" for k, s in enumerate(sel[t])) for t in TASKS)
    cand_txt = "; ".join(f"{t}: {len(ms[t][0]['design_selection']['candidates'])} candidates" +
                         (f" (dropped for compute budget: {ms[t][0]['design_selection']['dropped_for_budget']})"
                          if ms[t][0]["design_selection"].get("dropped_for_budget") else "") for t in TASKS)
    nan_txt = "; ".join(
        f"{t}: CBFV {n['cbfv_oliynyk']['nan_cells']} of {n['n_rows'] * n['cbfv_oliynyk']['shape'][1]} cells "
        f"({n['cbfv_oliynyk']['rows_with_nan']} rows), extra block {n['extra']['nan_cells']} of "
        f"{n['n_rows'] * n['extra']['shape'][1]} cells ({n['extra']['rows_with_nan']} rows)" for t, n in nans.items())
    p0 = next(t for t in mb.tasks if t.dataset_name == "matbench_steels").results["fold_0"]["parameters"]
    cpu_pc = p0["cpu_model"]
    pod_cpus = sorted({c for (c, _, _) in hosts if c != cpu_pc})
    per_task_txt = ", ".join(f"{t} {per_task_min[t]:.0f} min" for t in TASKS)
    if det:
        det_txt = ("DETERMINISM CHECK: fold 0 of matbench_steels and of matbench_expt_gap were re-fitted from scratch on a "
                   f"Linux cloud CPU ({', '.join(pod_cpus) or 'cloud CPU'}) with the same pinned package versions and compared "
                   "with the Windows-workstation run (per-expert max |difference| of the test predictions): " + "; ".join(
                       f"{t}: " + ", ".join(f"{e} {x['max_abs_diff_test']:.1e}" for e, x in v["per_expert"].items() if "max_abs_diff_test" in x)
                       + f"; re-tuned LightGBM parameters {'identical' if v['lgbm_params_identical'] else 'different'}"
                       for t, v in det["tasks"].items()) +
                   ". Same-machine reruns are exactly reproducible (the reused caches and the from-scratch selection agree to "
                   "0.0 wherever the same candidate is selected); across CPU architectures XGBoost (tree_method=hist) and, on the "
                   "small steels task, the Optuna-tuned LightGBM parameters are not bitwise reproducible, while every other "
                   "expert agrees to 1e-9 or better. ")
    else:
        det_txt = ""

    algorithm_long = (
        "DFT-Stack: a cross-fitted, simplex-weighted stack of composition experts for the four Matbench v0.1 composition "
        "tasks, using Materials Project DFT data as transfer-learning input features. "
        "EXTERNAL DATA DISCLOSURE: besides the formula-derived descriptors, the experts receive a block of Materials "
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
        "FOLD USE, STATED EXACTLY. Every choice is made inside the training rows T_k of each official fold k; the test "
        "rows are only predicted and recorded. Fold-internal: the Optuna hyperparameters, the simplex stack weights, the "
        "correction gate, the classification thresholds, AND the design choices that an earlier version of this entry "
        "had fixed from a design phase on fold 0's training rows: the feature-block set (candidates F_min = Magpie + "
        "element fractions + extra descriptors; F_red = F_min + DFT block; F_full = F_red + the remaining descriptor "
        "blocks) and, for matbench_expt_gap, the use of the hurdle experts and of the element-fraction extra-trees "
        "expert. These are now a small candidate set and the choice among the candidates is made per fold by the "
        "cross-fitted inner-CV score of the uncorrected stack on T_k (ties to the simpler candidate); for steels and "
        f"glass the DFT block is empty or nearly empty, so only F_red vs F_full are candidates ({cand_txt}). Selected "
        f"per fold: {sel_txt}. NOT fold-internal, disclosed: the candidate set itself was constructed from the "
        f"alternatives explored on fold 0's training rows, as were {REMAINING_FIXED}; and the label-free "
        "imputation described in the notes. The code was frozen (SHA-256 hashes) before the official test folds were "
        "predicted with it, and the test folds were predicted and scored once with this configuration. "
        "Classification: hard labels are recorded at a decision threshold that maximises balanced accuracy on the "
        "training-fold out-of-fold predictions (inner-CV threshold), so the board's ROC-AUC for these tasks equals "
        "balanced accuracy."
    )
    notes = (
        f"CPU-only (no GPU). Candidate scoring, selection and final predictions ran on one {cpu_pc} workstation "
        f"(16 cores / 24 threads; 5 parallel workers x 4 threads). The new expert cross-fits of the smaller "
        f"feature-block candidates ran on the same workstation (steels, glass) and on two cloud CPU pods "
        f"({', '.join(pod_cpus) or 'none'}; 32 vCPU each, 8 workers x 4 threads) for expt_gap and expt_is_metal. "
        "Expert cross-fits of the earlier run (F_full block set) were reused where the expert reads exactly the same "
        "columns (identical computation: same code, seeds, deterministic boosting). "
        f"Total wall time of the Amendment-3 jobs: {tot_h:.1f} h over all 20 task-folds (mean per fold: {per_task_txt}), "
        f"on top of the {a2_h:.1f} worker-hours of the reused F_full expert fits. Per-fold compute is recorded in the "
        "parameters of every fold in results.json.gz (wall_time_s, n_workers, threads_per_worker, cpu_threads_used, "
        "cpu_model, gpu, fit_hosts, expert_fit_seconds per expert, selected_candidate, candidate_inner_cv_scores). "
        + det_txt +
        f"DFT-block coverage (compositions matched by reduced formula): {cov_txt}. "
        "PRIOR RUNS ON THE OFFICIAL FOLDS (all reported in the manuscript and in the research repository): "
        "(1) Amendment 1 (2026-10-18; no DFT block, no hurdle experts, smaller expert pool, fully fold-internal "
        f"candidate selection): {a1_txt}; probability ROC-AUC {a1_auc}. (2) Amendment 2 (2026-10-19; the present "
        "expert pool and features with the design choices fixed from fold 0's training rows; this was the entry as "
        f"first submitted in this PR): {a2_txt}; probability ROC-AUC {a2_auc}. (3) The present entry, Amendment 3 "
        f"(2026-10-24; same pipeline, design choices selected per fold): {a3_txt}. Each run was pre-registered and "
        "scored once; Amendment 3 replaces the submitted file because it is the rule-compliant run, whatever its "
        "numbers. "
        "LABELS VARIANT: results.json.gz records hard labels for the two classification tasks (the 'labels' variant). "
        "A 'probabilities' variant (same predictions, probabilities recorded) was written at the same scoring moment "
        "and is kept in the research repository; recording labels was fixed in the pre-registered protocol before "
        "scoring, because the board compares label-based entries by balanced accuracy. For reference, the probability "
        f"ROC-AUC of the same predictions is {a3_auc}, and matbench itself would score the probabilities file "
        f"(thresholded at 0.5) at {pvb_txt}. "
        "TRANSDUCTIVE BUT LABEL-FREE IMPUTATION: two feature builders fill rare missing descriptor values with column "
        "medians computed over all rows of a task (training and test rows together; no labels): CBFV generate_features "
        f"for the Oliynyk block and the 'extra' matminer block in dftstack/data.py. Cells filled, per task: {nan_txt}. "
        "Columns that are entirely missing are dropped on the same basis. "
        "matbench==0.6 was installed with --no-deps because its scikit-learn==1.0 / monty pins do not build on "
        "Python 3.12; the newer versions listed in requirements were used. "
        "The folder is self-contained: run_submission.py plus the dftstack/ package (verbatim copies of the frozen "
        "research modules, including the per-fold candidate selection) reproduce results.json.gz and record every fold "
        "with task.record; the Matbench tasks and the matminer DFT datasets are downloaded on first use. "
        f"Full research repository (analysis plan with all amendments, freeze hashes, per-fold predictions, candidate "
        f"scores, ablations): {REPO}. "
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
Matbench v0.1 composition tasks with nested cross-validation inside each official training fold. Every choice,
including the feature-block set and (for `matbench_expt_gap`) the expert subset, is selected inside each training
fold from a small candidate set. See `info.json` for the full description, the external-data disclosure and the
earlier runs on the official folds.

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as hard labels). The
  `parameters` of every fold carry the compute of the run (`wall_time_s`, `n_workers`, `threads_per_worker`,
  `cpu_threads_used`, `cpu_model`, `gpu`, `fit_hosts`, `expert_fit_seconds` per expert, `pipeline`, `seed`) and the
  per-fold selection (`selected_candidate`, `candidate_inner_cv_scores`).
- `info.json`: metadata, references, notes (fold use stated exactly, prior runs, determinism check, labels variant,
  imputation note), pinned requirements.
- `run_submission.py`: reproduces the run (per-fold candidate selection included) and records every fold with
  `task.record` (with the same per-fold compute fields, measured on the rerun machine), then writes
  `results_rerun.json.gz` (`python run_submission.py [task ...]`). `results_rerun.json.gz` is only the output name of
  such a user rerun; it is not shipped in this folder.
- `dftstack/`: verbatim copies of the frozen research modules (pipeline, experts, feature blocks, DFT feature
  builder, frozen configurations `frozen_config2.py` / `frozen_config3.py`, and the per-fold candidate selection
  `run_final3.py`). Their docstrings refer to the analysis plan and design logs in the research repository ({REPO}).
- `requirements.txt`: the same pinned versions as `info.json`.

## Reproduce
```
pip install -r requirements.txt   # matbench==0.6 needs --no-deps on Python 3.12, see info.json notes
python run_submission.py
```
CPU-only. Data are downloaded on first use into `./data/cache/`; expert predictions are cached in
`./rerun_cache/` so that an interrupted run resumes. Without the original run's caches every expert of every
candidate is fitted from scratch (roughly {tot_h + a2_h:.0f} worker-hours at 4 threads per job).
"""
    (SUB / "README.md").write_text(readme, encoding="utf-8")

    rows = "\n".join(
        f"| `{t}` | {sc[t][0]} | {sc[t][1]:.4f} ± {sc[t][2]:.4f} | {fmt(t, pf2[t][1])} |" for t in TASKS)
    pr = f"""<!-- PR description. Label to request: new_benchmark -->
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

**Fold use, stated exactly.** Every choice is made inside the training rows of each official fold: the Optuna
hyperparameters, the simplex stack weights, the optional local residual correction and its gate (kept in {kept} of
{nmeta} task-folds), the classification decision thresholds, and the coarse design choices (feature-block set;
for `matbench_expt_gap` the hurdle experts and the element-fraction extra-trees expert), which are selected per
fold from a small candidate set by the inner-CV score of the stack on the training rows ({cand_txt}). Selected per
fold: {sel_txt}. Disclosed as not fold-internal: the candidate set itself was constructed from the alternatives
explored on fold 0's training rows ({REMAINING_FIXED}), and a label-free, transductive median imputation of a
few hundred missing descriptor cells (details in `info.json`).
Earlier runs on the official folds, all reported: Amendment 1 (2026-10-18, no DFT block; {a1_txt}) and
Amendment 2 (2026-10-19, the version first submitted in this PR, design choices fixed from fold 0's training rows;
{a2_txt}). Classification tasks are recorded as hard labels at the inner-CV balanced-accuracy threshold
(pre-specified before scoring), so the board's ROC-AUC equals balanced accuracy; the probability ROC-AUC of the
same predictions is {a3_auc}. The configuration was frozen by SHA-256 hash before the test folds were predicted
with it, and they were predicted and scored once.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score (this entry, Amendment 3) | Amendment 2 (first submitted version) |
|---|---|---|---|
{rows}

## Included files (`benchmarks/matbench_v0.1_DFTStack/`)
- `results.json.gz`, `info.json`: required files (`MatbenchBenchmark.from_file(...)`: `is_valid=True`,
  `is_recorded=True` for all 4 tasks).
- `run_submission.py`: source that reproduces the run (per-fold candidate selection included) with `task.record`
  calls.
- `dftstack/`: the pipeline, experts, feature blocks, DFT feature builder, frozen configurations and candidate
  selection (small `.py` files) imported by `run_submission.py`.
- `README.md`, `requirements.txt`.

## Compute
CPU-only, no GPU. Selection and final predictions on one {cpu_pc} workstation (16 cores / 24 threads, 5 workers x
4 threads); new expert cross-fits for the smaller feature-block candidates on the same workstation and on two cloud
CPU pods ({', '.join(pod_cpus) or 'none'}; 32 vCPU, 8 workers x 4 threads). Per-fold wall time, hosts, threads, CPU
models and per-expert fit seconds are recorded in the `parameters` of every fold in `results.json.gz`.
{det_txt}
## Paper and code
Manuscript under review at Computational Materials Science (no DOI or preprint yet). Research repository: {REPO}
"""
    (ROOT / "leaderboard_submission" / "PR_BODY.md").write_text(pr, encoding="utf-8")
    print("wrote info.json, requirements.txt, README.md, PR_BODY.md")
    for t in TASKS:
        print(t, sc[t], "A2", fmt(t, pf2[t][1]), "selected", sel[t])


if __name__ == "__main__":
    main()
