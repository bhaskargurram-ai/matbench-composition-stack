# ANALYSIS PLAN (pre-registered; written and frozen before any official test fold is scored)

Project: cross-fitted stacking + novelty-damped local out-of-fold (OOF) residual correction on the Matbench v0.1 composition tasks.
Status at writing:
- No official test-fold prediction has been made or scored.
- All design work used inner CV on fold 0's training rows only (`src/design.py`; results in `results/design/` and `logs/design_*.log`).
- Code is frozen by SHA-256 in `results/freeze_hashes.txt`. Any change after the freeze is logged in the Deviations section below.

## 1. Data and splits
- **Tasks:** matbench_steels (n = 312, MAE, MPa), matbench_expt_gap (4,604, MAE, eV), matbench_expt_is_metal (4,921, ROC-AUC), matbench_glass (5,680, ROC-AUC).
- **Splits:** the official Matbench v0.1 5-fold splits, loaded through `matbench.bench.MatbenchBenchmark`. No re-splitting.
- **Features (label-free):**
  - Magpie ElementProperty (132);
  - element fractions (103);
  - 21 extra matminer descriptors (Stoichiometry, ValenceOrbital, IonProperty(fast), BandCenter, AtomicOrbitals).
  - Featuriser failures (0 / 14 / 14 / 21 rows for steels / expt_gap / is_metal / glass) are imputed with the dataset-wide feature median. This uses no labels, but it uses feature values of all rows (disclosed).
- **Design-data disclosure.** The design choices (feature set, expert pool and hyperparameters, stacking loss, correction grid) were made on fold 0's training rows. The fold-0 test rows were never used. The test rows of folds 1-4 are part of fold 0's training rows, so they did inform these fixed design choices, though no model was ever fitted on them as test data. Fold-0 results are therefore reported separately as the cleanest estimate. **Superseded by Amendment 1 (Section 8):** the choice among the explored candidate configurations is now made inside each official training fold.

## 2. Method (frozen; `src/pipeline.py`, `src/experts.py`, `src/frozen_config.py`)
Everything below is done independently inside each official training fold T_k, and no official test target is ever accessed.

1. **Experts**, cross-fitted with J = 5 inner folds (seed 0), each also refit on all of T_k:
   - LightGBM (all features; for expt_gap an extra L1-objective copy);
   - random forest;
   - extra trees;
   - kernel ridge on element fractions and on standardised Magpie (regression; internal 3-fold grid over alpha and gamma), or RBF-SVC on the same two spaces (classification);
   - MLP (quantile-normalised inputs).
   - Regression predictions are clipped to the training-target range.
2. **Global stack:** simplex weights (w >= 0, sum w = 1) fitted on the OOF predictions.
   - Objective: MAE (linear programme) for regression; Brier (squared error on 0/1) for classification.
   - A cross-fitted copy of the stack (weights fitted without the row's inner fold) gives OOF stack predictions for the training rows.
3. **Naive local correction** (`stack_local`; the method transferred from SIG-Net):
   - yhat(q) = s(q) + alpha * sum_j w_qj r_j / (sum_j w_qj + kappa), with w_qj = exp(-(d_qj/h)^2) over the k nearest training compositions and r_j the OOF stack residual.
   - The pseudo-count kappa damps the correction toward 0 for novel queries (little kernel mass).
   - Classification applies the correction on the logit scale with probability residuals.
   - Selected on training rows (leave-self-out neighbours) by the task metric (MAE or ROC-AUC) over this grid:
     - space in {standardised Magpie (Euclidean), element fractions (L1)};
     - k in {5, 10, 20, 40};
     - h = hmult x median 1-NN distance, hmult in {0.5, 1, 2, 4, 8, 16};
     - kappa in {0, 0.1, 0.5, 2};
     - alpha in {0.25, 0.5, 0.75, 1} (regression) or {0.1, 0.25, 0.5, 1, 2, 4} (classification).
   - alpha = 0 (no correction) is always a candidate, and ties go to alpha = 0.
4. **Fold-matched correction** (`stack_fm`; the departure arm from the IdeaSpark card):
   - The transfer is r_j + F^(-m(j))(q) - F^full(q).
   - Selection uses deployment-matched training instances built from pair-excluded refits F^(-m,-m'), with the global stack weights held fixed.
   - The grid is the same as for the naive correction.
5. **Hard labels** (classification): a threshold maximising balanced accuracy on each arm's training OOF predictions.

6. **Execution.**
   - `src/launch_final.py` runs one process per (task, fold). There are at most 5 concurrent workers, and each is pinned to 4 threads (experts use `n_jobs=4`; OMP/BLAS/MKL are capped at 4).
   - LightGBM uses `deterministic=True` and `force_col_wise=True`, so results do not depend on scheduling. The design runs used LightGBM without the deterministic flag; this changes no design choice.
   - Per-fold outputs go to `results/final/<task>/fold<k>_predictions.npz` and `fold<k>_meta.json`. Per-expert caches make runs resumable.
7. **Design evidence:** `results/design/DESIGN_LOG.md` lists every configuration tried. The last design round (v6, which included the fold-matched arm) was complete for steels only; it was stopped after 1/5 (expt_gap), 4/5 (expt_is_metal) and 3/5 (glass) pseudo-folds to free CPU for the final run, and its partial means are reported as such.

## 3. Arms (all reported)
- `single:<expert>` for every expert;
- `best_single`: the expert with the best OOF score, selected inside T_k;
- `mean_ensemble`;
- `stack`;
- `stack_local` (primary);
- `stack_local_magpie` and `stack_local_frac` (kernel-space ablations);
- `stack_fm` (fold-matched);
- `single_local` (correction without stacking, applied to `best_single`).

## 4. Endpoints and tests
**Primary endpoint (one per task):**
- Delta = mean over the 5 official folds of metric(`stack_local`) - metric(`stack`).
- Metric: MAE for regression; ROC-AUC computed from probabilities for classification.

**Primary test:**
- A paired bootstrap of Delta that resamples test points within each fold (B = 10,000, seed 12345). Report the 95% percentile CI and a two-sided p.
- Holm correction across the 4 tasks.
- A task counts as "improved by the correction" only if Delta is in the favourable direction AND the Holm-adjusted p < 0.05.
- There is no pooled cross-task claim.

**Reported alongside the primary test** (descriptive; 5 folds give low power):
- per-fold Delta and the number of folds improved;
- paired t-test over folds;
- Nadeau-Bengio corrected resampled t-test;
- Cohen's d_z;
- for regression, a Wilcoxon signed-rank test on per-point absolute errors.

**Secondary comparisons** (same statistics, no multiplicity claim): `stack_local` vs `best_single`; `stack` vs `best_single`; `single_local` vs `best_single`; `stack_local` vs `mean_ensemble`; `stack_fm` vs `stack`; `stack_fm` vs `stack_local`; the two kernel-space ablations vs `stack`.

**Further secondary analyses:**
- **Support strata.** Test errors (MAE, or the Brier score for classification) for `stack`, `stack_local`, `stack_fm` and `best_single`, binned by each test composition's distance to its nearest training composition. Bin edges are the training rows' leave-one-out 1-NN distance quartiles within each fold.
- **Calibration** for classification: Brier score, log loss, 10-bin ECE.
- **Selections:** the selected correction configuration per fold, and the label-permutation control. The control is diagnostic only, because shrinkage toward the mean can be selected even under the null.

## 5. Leaderboard comparison
- **Snapshot:** `results/leaderboard_snapshot.json`, taken 2026-10-18.
- **Regression:** compare our mean MAE (+/- std over folds) with every entry, and report the rank we would have.
- **Classification:** most board entries' "rocauc" equals their balanced accuracy, consistent with hard-label submissions. So:
  - like-for-like comparison: our balanced accuracy at the inner-CV threshold vs the board's balanced-accuracy column;
  - our probability ROC-AUC is reported but not ranked against label-based entries.
- **Claim rule:** no state-of-the-art claim unless our mean beats the best entry's mean. Fold std is always shown.
- **Submission files** (two variants: labels and probabilities) are written under `results/final/matbench_submission/` and are NOT submitted.

## 6. Reporting rules
- All 4 tasks, all 5 folds and all arms are reported, wins or losses. There is a single run with seed 0, and no seed or fold is selected after viewing.
- If a bug is found after test scoring, both the original and the corrected results are reported, together with the fix's diff, in Deviations.
- Compute (CPU wall time) is reported per task.
- Skipped components (MODNet, CrabNet/Roost, TabPFN, ElMD), with the reasons: CPU cost, dependency footprint, and the time budget.

## 7. Deviations
- **Line endings.** Freeze hashes were re-recorded on LF-normalised files (commit dd8624b). The code was unchanged.
- **Superseded first run, never scored.** The first final run under the original plan (commit dd8624b) produced test-set predictions for 19/20 folds. None was ever scored: no test metric was computed and no test label was read. That run was stopped and is superseded by Amendment 1. Its files are kept, unscored, in `results/final/<task>/superseded_unscored_run1/`, and they will not be scored for the paper. The steels fold-4 job hung in a joblib/loky worker and was killed.
- **Scheduling-only code changes for Amendment 1** (numerical results unchanged): the internal KRR grid search now uses joblib's threading backend; per-fold logging.
- **Reporting-only scripts added after the first freeze:** `src/make_paper_assets.py` (generates manuscript tables and figures from the result files) and `research/make_bib.py`.
- **Keep-awake helper.** A per-process keep-awake request (Windows `SetThreadExecutionState`) runs during the final run. Without it, Modern Standby suspended the jobs, which inflated wall times. It changes no system setting.

- **Post-freeze bug fix in `src/evaluate.py` (one line).** The first scoring attempt crashed with a KeyError before any metric was output. Its per-fold loop took the arm list from fold 0, but with per-fold design selection the set of single-expert arms differs between folds (for example `single:lgbm_l1_all`). The fix iterates over each fold's own arms. No metric had been displayed before the crash. Arms present in only some folds are averaged over the folds in which they exist.

## 8. Amendment 1 (2026-10-18 22:40 local; written BEFORE any official test fold was scored)
**Why.**
- The original plan made the design choices on fold 0's training rows. Those rows contain the test rows of folds 1–4, so for folds 1–4 the design was not chosen fold-internally.
- Matbench expects all tuning to use each fold's training data only.
- The owner therefore approved making every design choice inside each official training fold. This was decided before any test score existed.

**What changes.** For each official fold k, using only the training rows T_k:
1. **Candidates.** All candidate configurations already explored in the design phase (`results/design/DESIGN_LOG.md`) are listed in `src/frozen_config.py` `CANDIDATES`. Every candidate is cross-fitted (J = 5, seed 0) on T_k.
   - Regression: A = base features + L2 stack (design v3); B = base features + L1 stack + L1-LightGBM expert (v4); D = extra features + L1 stack (v6, steels); C = extra features + L1 stack + L1-LightGBM expert (v5/v6, expt_gap).
   - Classification: A = base features (v3); C = extra features (v6).
2. **Selection criterion.** The same inner-CV criterion as the design phase, made automatic: the cross-fitted OOF score of the global simplex stack on T_k (MAE for regression, ROC-AUC for classification).
   - This criterion uses the uncorrected stack, so it does not favour either correction arm.
   - Ties go to the earlier (simpler) candidate.
3. **Correction.** The correction hyperparameters are then selected inside T_k exactly as in Section 2. The fold-matched arm is always computed for the selected configuration.
4. **Record.** The per-fold choice and every candidate's score are recorded in `fold<k>_meta.json` under `design_selection` and reported.

**What does not change.** The expert hyperparameters and grids, the correction grid, the endpoints, the tests and the reporting rules.

**Disclosure that remains.** The candidate set itself (which configurations exist) was constructed while looking at fold-0 inner-CV results. The choice among the candidates is now fold-internal.

**Caching.** Expert outputs from the superseded run are reused only where the expert configuration and inputs are identical and were computed solely from T_k: same task, fold, feature set, expert name and settings, seed and J. They are stored per fold and per feature set (`fold<k>_experts` = extra features, `fold<k>_experts_base` = base features).

**Execution.** `python -m src.launch_final --workers 5` (5 workers × 4 threads, deterministic LightGBM). Expected wall clock: about 30–60 min. The base-feature candidates need new cross-fits (no pair refits unless one is selected); the extra-feature candidates reuse caches.
