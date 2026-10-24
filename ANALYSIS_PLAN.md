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

## 9. Amendment 2 (protocol part, 2026-10-19 00:10 local; written BEFORE any Amendment-2 test scoring)
**Status and honesty statement.**
- This amendment was written AFTER the Amendment-1 official test results had been scored and seen (commit f422c82).
- Those results are kept and reported unchanged as the first pre-registered result. Amendment 2 does not replace them.
- Owner decision: strengthen the pipeline to try to be competitive with the leaderboard leaders.

**What Amendment 2 changes:** only the expert pool, the feature sets (including one disclosed transfer-learning feature block) and the selection/threshold procedure. Every selection (features, experts, hyperparameters, stack, optional local correction, thresholds) is made by inner cross-validation inside each official training fold T_k only.

**Rules:**
1. The official test folds are scored exactly ONCE under Amendment 2, after the code is re-frozen (new SHA-256 hashes). There is no iteration on test results. Any later bug fix is reported as a deviation, with both versions shown.
2. Design exploration before the freeze uses training rows only (fold 0's training rows and/or inner CV inside each T_k). No official test rows or test labels are used.
3. External data is limited to offline DFT (PBE) databases distributed with matminer. These are used as features, mapped by reduced composition, and disclosed as transfer learning.
   - No experimental labels from any external source.
   - No LLM or pretrained model that could have memorised experimental values.
   - Missing matches are encoded with an indicator column.
4. Experts that exceed about 45 min per official fold on the local CPU budget (5 workers × 4 threads) are dropped and logged.
5. **New primary endpoint (Amendment 2):** per task, the final Amendment-2 pipeline vs the leaderboard leader (snapshot 2026-10-18).
   - The comparison uses the board's own metric semantics: MAE for regression. For classification, balanced accuracy at the inner-CV threshold vs the board's balanced-accuracy column for label-based entries; probability ROC-AUC is also reported.
   - Retained paired tests: stack vs best single expert; final pipeline vs the Amendment-1 primary arm; optional local correction vs stack.

**Details to be fixed before the re-freeze:** the exact expert list, feature blocks, external dataset and tuning budgets will be recorded in Section 10 (the Amendment 2 specification) before the final run.

## 10. Amendment 2 specification (FROZEN before the Amendment-2 final run; no Amendment-2 test scoring before this)
All choices below were made on fold 0's TRAINING rows only (pseudo-outer folds, seed 2026; `results/design_a2/`, `logs/design_a2*`, quick CV scripts) and are fixed in `src/frozen_config2.py`, `src/experts2.py`, `src/pipeline2.py`, `src/features_all.py`, `src/features_more.py`, `src/external_dft.py`, `src/run_final2.py` and `src/evaluate2.py`.

**Features (label-free, computed per formula).**
- Amendment-1 features: Magpie, element fractions, 21 extra descriptors.
- matminer ElementProperty presets: deml, matminer, megnet_el.
- Meredig, Miedema, YangSolidSolution, AtomicPackingEfficiency, WenAlloys, TMetalFraction.
- CBFV Oliynyk statistics.
- Tried and excluded after design (too slow or no gain): CBFV jarvis and mat2vec, ElementProperty matscholar_el.

**Disclosed transfer block (`dft_*`).**
- Source: Materials Project PBE data as distributed by matminer, `matbench_mp_gap` (gap) joined to `matbench_mp_e_form` (formation energy) by reduced formula + number of sites + volume.
- Matching to task compositions is by reduced formula. The "stable" entry is the polymorph with the lowest formation energy per atom.
- Features: stable-entry gap, metal flag and formation energy; min/median/max gap over polymorphs; metallic fraction; number of polymorphs; indicator `dft_matched`.
- Matches: expt_gap 3,347/4,604; expt_is_metal 3,613/4,921; glass 60/5,680; steels 0.
- No experimental labels are used, and no pretrained model or LLM.
- The figshare `mp_nostruct_20181018` table (which has e_hull) could not be downloaded (HTTP 500).

**Expert pool** (`src/experts2.py`; F1 = Amendment-1 features + dft; F2 = F1 + the new blocks):
- LightGBM on F2, tuned by Optuna inside T_k: 20 TPE trials (seed 0), with the default configuration evaluated first; the objective is the mean inner-validation metric over the same J = 5 inner splits.
- XGBoost on F2 (fixed parameters).
- CatBoost on F1 (fixed).
- Extra trees on F1 and on F2.
- Extra trees on element fractions, with max_features chosen by an internal 3-fold grid. Not used for expt_gap.
- Kernel ridge (regression) or RBF-SVC (classification) on element fractions and on Magpie.
- Distance-weighted kNN on element fractions, k chosen by an internal 3-fold grid.
- MLP on F1.
- expt_gap only, three hurdle experts for the zero-inflated target: a LightGBM classifier for y > 0 combined with a LightGBM L1 regressor fitted on the y > 0 rows (prediction p·r, or r if p ≥ 0.5 else 0), and an extra-trees variant (p·r).

**Dropped experts (logged).**
- CrabNet: own CPU training loop around the `crabnet` network; about 23 min per fit and dev MAE 0.419 on expt_gap.
- MODNet 0.4.5 with TensorFlow 2.21: a single fit on 200 steels rows exceeded 40 min, even on 281 features.
- XGBoost Optuna tuning: too slow on CPU.
- Random forest on F2: slow, with near-zero stack weight.
- The CatBoost input was reduced to F1, because on F2 it took about 29 min for a 200-row steels cross-fit.

**Stack and final pipeline.**
- Simplex stack: L1 loss for regression, Brier for classification, with cross-fitted OOF.
- The naive local correction is optional. It is selected as in Amendment 1, and kept only if a NESTED inner-CV check confirms that it improves the honest OOF score (`gated_correction`).
- **Final arm = `stack_local`** (the stack, plus the correction only where the gate keeps it).
- Classification hard labels use the balanced-accuracy-maximising threshold chosen on the training OOF predictions.

**Disclosure.** The OOF predictions of the Optuna-tuned expert are mildly optimistic, because tuning and stacking use the same inner splits. This can affect only the level-1 stack weights. The test folds are never involved.

**Scoring (once).**
- Run: `python -m src.launch_final --amendment2 --workers 5`.
- Then: `python -m src.evaluate2 --submission`.
- Primary analysis per task: final vs leaderboard leader (TPOT-Mat on steels, Darwin on expt_gap and expt_is_metal, MODNet v0.1.12 on glass).
  - Metric: board semantics (MAE; balanced accuracy, with labels or probabilities at 0.5 for the leader, as the board computed).
  - Test: paired bootstrap over test points within folds using the leaders' public per-sample predictions (matbench repository, downloaded 2026-10-19), Holm over the 4 tasks.
  - Claims allowed: "beats the leader" only if the mean is better AND the Holm-adjusted p < 0.05; "on par" if the CI includes 0.
- Retained comparisons: stack vs best single; final A2 vs Amendment-1 primary arm; correction vs stack.
- The probability ROC-AUC is also reported, for us and for leaders that recorded probabilities.

## 11. Amendment-2 execution record and deviations
- **Execution.** Predictions for all 20 folds were committed (aa9db92) before scoring. Scoring ran once (`src.evaluate2`, about 2026-10-19 08:00 local) with no errors and no re-runs. Results are in `results/final_a2/{stats.json, summary.json, metrics_per_fold.csv}`, and the unsubmitted Matbench files in `results/final_a2/matbench_submission/{labels,probabilities}/` (both `is_valid=True`).
- **Reporting-only code added after the Amendment-2 freeze:** `src/make_paper_assets2.py` (manuscript tables and figures from the result files) and changes to `research/make_bib.py` (new verified references). None of the frozen pipeline or scoring code changed.
- **Leader predictions** were downloaded from the public matbench repository on 2026-10-19 and stored under `data/cache/external/leaders/`.

## 12. Post-hoc ablation (written 2026-10-19 after Amendment-2 scoring; NOT used for any selection; the final pipeline is unchanged)
**Label:** post-hoc ablation, written after Amendment-2 scoring and not used for any selection. The final reported pipeline (Amendment 2) and the prepared leaderboard files do NOT change because of it.

**Variants.** Both re-run the frozen Amendment-2 code unchanged (same experts, inner-CV selection inside each T_k, seeds, J = 5, Optuna budget), except for:
- **A2-noDFT** (all 4 tasks): the DFT feature block (`dft_*` columns) is removed from every feature group. On expt_gap the hurdle experts are kept.
- **A2-noHurdle** (expt_gap only): the three hurdle experts are removed and the DFT block is kept. Together with A2-noDFT this separates the two sources of the expt_gap gain.

**Caching.** Expert caches from the Amendment-2 run are reused only for experts whose inputs and configuration are byte-identical:
- in A2-noDFT, the experts that read only Magpie or element-fraction columns;
- in A2-noHurdle, every remaining expert, including the tuned LightGBM parameters.

**Statistics.** Each variant is scored once. It is compared with the Amendment-2 final arm (`stack_local`) by:
- the paired bootstrap over test points within folds (B = 10,000) on the board metric (MAE; balanced accuracy at each pipeline's own inner-CV threshold), reported with the per-fold differences;
- the same comparison against the leader.

There is no multiplicity claim: the analysis is descriptive.

### 12.1 Ablation execution record (appended 2026-10-19, before ablation scoring)
- **Interruption.** The ablation run was interrupted at about 10:42 local time, when all workers were killed. At that point the finished folds were nodft expt_gap 5/5, expt_is_metal 5/5 and glass fold 4, plus nohurdle expt_gap 5/5 (already in 379af22).
- **Checks before any scoring.** Every finished fold was checked: the npz loads, all arms are finite, test ids equal the official fold, and the meta JSON parses. The finished folds were committed in 3ec9272.
- **Rerun.** The 9 interrupted folds (nodft glass 0–3 and steels 0–4) were rerun with the unchanged launcher (`python -m src.launch_ablation`; 5 workers × 4 threads; deterministic LightGBM) under a per-process keep-awake request.
  - Their partially written expert caches had been checked before reuse (they load, have the correct shapes and are finite).
  - The pipeline is resumable and deterministic, so reusing them does not change the predictions.
- **Reporting-only change before scoring.** `src/evaluate_ablation.py` was extended to save per-fold board metrics, mean ± s.d. and probability ROC-AUC. The LaTeX output moved to a new generator, `src/make_ablation_assets.py`, which reads `results/ablation_summary.json`. The statistics are unchanged: the same paired bootstrap as `src.evaluate2`, B = 10,000, rng seed 12345, and the same comparisons. All p-values are unadjusted (descriptive).

## 13. Amendment 3 (fold-internal selection of the fixed design choices; written 2026-10-24 BEFORE any Amendment-3 test prediction)
**Why and status.**
- The matbench PR #372 review (maintainer, 2026-10-24) identified the one step of Amendment 2 that is not fold-internal: the coarse design choices fixed in `src/frozen_config2.py` / `src/experts2.py` (feature-block set; for expt_gap the hurdle experts and the removal of `et_frac`) were made in a design phase on fold 0's training rows, which contain the test rows of folds 1–4.
- Amendment 3 turns those choices into a small candidate set and makes the choice among the candidates inside each official training fold T_k, exactly as Amendment 1 (Section 8) did. Nothing else changes.
- This amendment is written after the Amendment-1 and Amendment-2 official results were scored and seen (Sections 8–11). Both remain reported unchanged. Amendment 3 is scored once; its result replaces the leaderboard submission file whatever it is (better or worse than Amendment 2), because it is the rule-compliant run. The owner approved this on 2026-10-24.
- No Amendment-3 test row has been predicted when this section is frozen. The code (`src/frozen_config3.py`, `src/run_final3.py`, `src/launch_final3.py`, `src/pod_job_a3.py`, `src/evaluate3.py`) was dry-run only on fold-0 TRAINING rows (`run_final3 smoke`: a pseudo split inside T_0, 2 Optuna trials, throw-away folder) and, for the scorer, on random targets (`src/smoke_test_evaluate3.py`).

### 13.1 What stays exactly as frozen in Section 10
Official folds; J = 5 inner folds, seed 0; the expert pool and every expert setting (`src/experts2.py`: Optuna 20 TPE trials for LightGBM with the default evaluated first, fixed XGBoost, CatBoost on F1, extra trees, KRR/SVC with the fixed C and gamma, kNN, MLP, the three hurdle experts for expt_gap); the simplex stack (L1 / Brier); the nested-CV-gated local correction; the balanced-accuracy threshold on training OOF; the final arm `stack_local`; hard labels recorded for classification; the disclosed DFT transfer block (`src/external_dft.py`); the feature builders and their imputation (13.5). `src/pipeline2.py`, `src/experts2.py`, `src/features_all.py`, `src/features_more.py`, `src/external_dft.py`, `src/data.py` are not modified; `run_final3` calls the unchanged `pipeline2.run_split2` for the selected candidate.

Choices that remain design-phase decisions and are disclosed as such (they define the candidate set, not the winner): the Optuna budget, the dropped experts (random forest, CrabNet, MODNet), CatBoost on F1, the SVC C/gamma, and the composition of the candidate set itself (constructed from the alternatives explored on fold 0's training rows, `results/design_a2/DESIGN_LOG_A2.md`). This is the same "candidate set constructed while looking at fold-0 inner-CV results" disclosure that Amendment 1 carried.

### 13.2 Candidate set (`src/frozen_config3.py`)
Feature-block sets (block names as in `features_all.features`):
- `F_min` = ("base", "extra"): Magpie, element fractions, 21 extra descriptors (the Amendment-1 features);
- `F_red` = ("base", "extra", "dft"): F_min plus the DFT transfer block. The DFT block is kept in every candidate except F_min because it is the disclosed external data of this entry, not a design choice under test;
- `F_full` = `BLOCKS2`: F_red plus ep_deml, ep_matminer, ep_megnet, meredig, miedema, yang, ape, wen, tmetal, cbfv_oliynyk (the frozen Amendment-2 set).

The expert feature groups are derived from the candidate's columns as in Amendment 2 (F1 = magpie/frac/extra/dft columns present, F2 = all columns present; `magpie` and `frac` groups unchanged), so under F_red the "F2" experts read the F1 columns.

Expert subsets (expt_gap only): hurdle experts {off, on} x `et_frac` {in, out}. These change only which already-fitted experts enter the stack, so they are scored by re-fitting the stack on the fitted experts' OOF predictions (no new expert fits). For the other three tasks the single subset is the frozen Amendment-2 pool (no hurdle experts, `et_frac` in).

| Task | Block-set candidates (simplest first) | Expert subsets | Candidates |
|---|---|---|---|
| matbench_steels | F_red, F_full (DFT block: 0 matches, so F_min = F_red in information) | 1 | 2 |
| matbench_glass | F_red, F_full (DFT block: 60/5680 matches) | 1 | 2 |
| matbench_expt_is_metal | F_min*, F_red, F_full | 1 | 3 (2 without F_min) |
| matbench_expt_gap | F_min*, F_red, F_full | hurdle {off, on} x et_frac {in, out} = 4 | 12 (8 without F_min) |

(*) F_min is a candidate for the two gap tasks only if its expert fits complete for all 5 folds of that task within the compute budget of 13.6. The decision is made on compute (completion), before any candidate score of that task is viewed, and is recorded per fold in `fold<k>_meta.json["design_selection"]["dropped_for_budget"]` and reported. A candidate is never dropped for a subset of folds.

For expt_gap the Amendment-2 run never fitted `et_frac` (it was dropped by the frozen config); it is cross-fitted now on T_k (element fractions only, identical for all block sets, so fitted once per fold and shared).

### 13.3 Selection rule (fold-internal, automatic)
For each official fold k, using only T_k (`run_final3.select_and_predict`):
1. For every candidate block set, tune (Optuna, 20 trials) and cross-fit the full expert pool on T_k with the unchanged Amendment-2 code (`pipeline2.tune_expert`, `pipeline2.crossfit2`; J = 5, seed 0, the same inner splits for every candidate).
2. Score every candidate (block set, expert subset) by the cross-fitted OOF score of the UNCORRECTED global simplex stack on T_k (`pipeline.crossfitted_stack`; MAE for regression, ROC-AUC for classification), the Amendment-1 criterion. It does not favour either correction arm.
3. Pick the best candidate; ties go to the earlier (simpler) candidate in the order F_min < F_red < F_full, hurdle off < on, et_frac in < out.
4. For the chosen candidate only: `pipeline2.run_split2` unchanged (stack weights, correction selection and nested gate, thresholds, final predictions). The stack OOF score it reports must equal the candidate's score from step 2 (asserted).
5. Record every candidate's score and the choice in `fold<k>_meta.json["design_selection"]`, together with `compute` (13.6).

### 13.4 Expert cache reuse and the determinism check
- Reuse rule: an expert's cached cross-fit (`<name>.npz`: OOF and test predictions, seconds; plus `<name>_params.json` for the tuned expert) is reused only if the expert has the same name AND reads exactly the same input columns (same names, same order) as the cached computation, which was itself computed from T_k alone with the same code, seeds, J and thread count. Then the computation is identical (same code path, deterministic LightGBM/XGBoost/CatBoost/scikit-learn seeds), and refitting it would only reproduce it. Sources: the Amendment-2 caches `results/final_a2/<task>/fold<k>_experts/` (block set F_full) and the other candidates' caches of the same fold. The provenance of every cache file is written to `<cache dir>/cache_provenance.json` and copied into `fold<k>_meta.json`.
- Consequences: the F_full candidate reuses every Amendment-2 expert of the fold (so whenever F_full with the Amendment-2 subset is selected, the fold's predictions are the Amendment-2 predictions); under F_red the experts on `F1`, `magpie` and `frac` columns are identical to Amendment 2 (cat_F1, et_F1, mlp_F1, hurdle_et_F1_exp, krr/svc_frac, krr/svc_magpie, knn_frac, et_frac) and only lgbm_F2_tuned (with its tuning), xgb_F2, et_F2 and the two hurdle LightGBM experts are new; under F_min only the `magpie`/`frac` experts are reused.
- Determinism check (answers the reviewer's reproducibility question and verifies that reuse is legitimate): fold 0 of matbench_steels and fold 0 of matbench_expt_gap are re-fitted from scratch (block set F_full, full pool, `--no-reuse`) on a RunPod CPU pod (Linux, different CPU, same pinned package versions) into `results/final_a3/determinism/<task>/fold0_experts_F_full/`. For every expert the maximum absolute difference between the re-fitted and the cached OOF and test predictions, and whether the re-tuned LightGBM parameters equal the cached ones, is reported (`results/final_a3/determinism/report.json`), whatever it is. Differences at floating-point level from a different BLAS/OS are possible for the KRR/SVC/MLP experts and are reported as such; the check does not alter the run.

### 13.5 Imputation: unchanged and disclosed
The two feature builders that fill rare missing descriptor values with label-free column medians over all task rows (CBFV `generate_features` for the Oliynyk block; the `extra` block in `data.py`) stay as frozen, so that the Amendment-2 caches remain valid computations. The affected cells are counted in `results/final_a2/imputation_nan_counts.json` (expt_gap: CBFV 2782 of 1.42M cells, extra 42 of 96.7k; expt_is_metal 3075 of 1.52M, 42; glass 132 of 1.75M, 63; steels none) and disclosed in `info.json`. The train-only imputation sketched in the draft (`leaderboard_submission/AMENDMENT3_PLAN_DRAFT.md`, Section 4) is NOT part of Amendment 3.

### 13.6 Compute plan and recording
- Expert fits that are new under Amendment 3: F_red (and F_min) for all tasks, and `et_frac` for expt_gap. F_full needs no new fit.
- Split: expt_gap and expt_is_metal expert fits on two RunPod SECURE CPU pods (cpu5c, 32 vCPU each, 8 workers x 4 threads, all 5 folds in parallel; launcher `launch_cpu.py` in the session scratchpad, runner `src/pod_job_a3.py`); the determinism jobs of 13.4 run on the expt_is_metal pod. Cost cap: hard stop at $4 per pod, total <= $8. Shipped to the pod: `src/`, the feature caches of the tasks (`data/cache/*.pkl`, `*_folds.json`), the Amendment-2 expert caches of the tasks, and a pinned requirements file (same versions as `leaderboard_submission/matbench_v0.1_DFTStack/requirements.txt`; matbench/matminer/pymatgen are not needed on the pod because all features are cached). steels and glass expert fits and ALL selection jobs (13.3) run on this PC (`src/launch_final3.py`, 5 workers x 4 threads, at most 20 threads).
- Fallback: if a pod job fails or is stopped by the cost cap, the missing fits are done on this PC by the resumable `select` jobs (same code, same caches); this is recorded. If the F_min fits of a gap task do not complete within budget, F_min is dropped for that task (13.2).
- Recorded per fold in `fold<k>_meta.json["compute"]`: the selection job's wall time, host, CPU model, cpu count, workers and threads; per candidate the `fit_info.json` of its expert fits (host, CPU model, workers, threads, wall time, which experts were fitted there) and the cache provenance; `expert_seconds` per expert (the fit time of the computation actually used, measured where it was computed). `results/final_a3/timing.jsonl` (PC) and `pod_timing.jsonl` (pods) keep the per-job wall times. The submission file's per-fold `parameters` carry the same fields as the Amendment-2 file (wall_time_s = selection-job wall time plus the expert-fit wall time of that fold's new fits, n_workers, threads_per_worker, cpu_threads_used, cpu_model (all hosts used), gpu, expert_fit_seconds, pipeline, seed) plus `selected_candidate`.

### 13.7 Scoring (once) and reporting
- Order: this section is frozen (SHA-256 hashes of `ANALYSIS_PLAN.md` and of `src/frozen_config3.py`, `src/run_final3.py`, `src/launch_final3.py`, `src/pod_job_a3.py`, `src/pod_ship_a3.py`, `src/evaluate3.py`, `src/smoke_test_evaluate3.py`, and of the unchanged `src/pipeline2.py`, `src/experts2.py`, `src/pipeline.py`, `src/features_all.py`, `src/features_more.py`, `src/external_dft.py`, `src/data.py`, `src/frozen_config2.py`, `src/evaluate2.py`, `src/evaluate.py` in `results/freeze_hashes_a3.txt`) and committed BEFORE any Amendment-3 test prediction; then the run; then the 20 fold prediction files and metas are committed BEFORE scoring; then `python -m src.evaluate3 --submission` runs ONCE.
- `src/evaluate3.py` is a reporting-only copy of `src/evaluate2.py` pointed at `results/final_a3`: the same metrics (board semantics: MAE; balanced accuracy at the inner-CV threshold), the same paired bootstrap over test points within folds (B = 10,000, rng seed 12345), Holm over the 4 tasks for the leader comparison, stack vs best single, correction vs stack, probability ROC-AUC. Added: A3 vs the Amendment-2 final arm and A3 vs the Amendment-1 final arm (same paired bootstrap; descriptive), and the per-fold selected candidates with every candidate's inner-CV score.
- Outputs: `results/final_a3/{summary.json, stats.json, metrics_per_fold.csv}` and `results/final_a3/matbench_submission/{labels,probabilities}/results.json.gz`.
- Submission: if the labels file is `is_valid`, it replaces `leaderboard_submission/matbench_v0.1_DFTStack/results.json.gz` (with the per-fold compute parameters), and `dftstack/` plus `run_submission.py` are updated so that the shipped folder reproduces Amendment 3. The Amendment-2 and Amendment-1 numbers stay reported alongside (info.json, README, PR body, manuscript).
- No iteration on test scores. Any later bug fix is reported as a deviation with both versions shown. The result is reported whatever it is, including if Amendment 3 is worse than Amendment 2 on any or all tasks.
