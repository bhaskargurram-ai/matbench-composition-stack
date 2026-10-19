# STATUS

**Current stage (2026-10-19 morning):**
- Amendment 2 is complete: frozen (537728f), run (predictions committed in aa9db92), scored ONCE, and written up.
- The manuscript has been rewritten around both pre-registered runs and compiles: `manuscript/main.pdf` (26 pp) and `supplementary.pdf`.
- Nothing has been submitted or published. The repo is still private.

## AMENDMENT-2 FINAL RESULTS (official folds; all selection inside training folds; scored once)
Final pipeline = cross-fitted simplex stack. The nested-CV gate kept the local correction in 0/20 folds.

The comparison is paired over the same test compositions, using the leaders' public predictions. Classification is scored as balanced accuracy, which is what the board actually reports for these leaders.

| Task | A2 final | Leader (board) | Δ (ours − leader) [95% CI] | p_Holm | Verdict | Rank | A1 primary |
|---|---|---|---|---|---|---|---|
| steels, MAE (MPa) | **80.10 ± 10.19** | TPOT-Mat 79.95 | +0.16 [−4.54, +4.89] | 0.93 | on par | 2/12 | 82.80 |
| expt_gap, MAE (eV) | **0.2352 ± 0.0167** | Darwin 0.2865 | −0.0513 [−0.070, −0.033] | <0.001 | **beats leader (−17.9%)** | **1/13** | 0.3240 |
| expt_is_metal, balanced accuracy | **0.9547 ± 0.0033** | Darwin 0.9598 | −0.0051 [−0.012, +0.002] | 0.30 | on par | 2/8 | 0.9370 |
| glass, balanced accuracy | **0.8937 ± 0.0158** | MODNet v0.1.12 0.9603 | −0.0666 [−0.076, −0.057] | <0.001 | below leader | 2/8 | 0.8841 |

- **Probability ROC-AUC:** expt_is_metal 0.9895, glass 0.9606. MODNet's glass probabilities give 0.9899.
- **Drivers:** the disclosed DFT transfer block (Materials Project PBE data via matminer, matched by reduced formula) plus hurdle experts for the zero-inflated band gap.
- **Stack vs best single:** better only on glass (+0.18% AUC, p = 0.014).
- **Steels caveat:** the best single model (extra trees on element fractions) reached 79.56 MAE, but the pre-registered stack (80.10) is reported as final.
- **Compute:** 5 workers × 4 threads; median per fold 9 / 62 / 48 / 57 min (steels / expt_gap / expt_is_metal / glass); 14.3 worker-hours in total.
- **Dropped experts:** CrabNet (slow and weak) and MODNet (more than 40 min per fit on CPU).
- **Unsubmitted Matbench files:** `results/final_a2/matbench_submission/{labels,probabilities}/results.json.gz`.
- **Honesty notes** (all in the manuscript):
  - Amendment 2 was written after the Amendment-1 results were seen (two looks at the test folds).
  - Fold-0 training rows shaped the candidate space.
  - The Optuna OOF predictions are mildly optimistic for the stack weights only.
  - The leaders' protocols were not audited.

## AMENDMENT-1 RESULTS (first pre-registered result; official Matbench v0.1 folds; per-fold design selection inside each training fold; scored 2026-10-18 ~23:30)
Primary arm: stack + naive novelty-damped local OOF correction. Mean ± std over 5 folds.

| Task | Best single (inner CV) | Stack | **Stack + local corr. (primary)** | Stack + fold-matched | Primary test (local vs stack) | Board leader | Our rank |
|---|---|---|---|---|---|---|---|
| steels MAE (MPa) | 83.41 | 83.12 | **82.80 ± 11.65** | 82.02 | −0.39%, p_Holm = 1.0, 3/5 folds | TPOT-Mat 79.95 (MODNet 87.76) | 3 of 12 |
| expt_gap MAE (eV) | 0.3259 | 0.3264 | **0.3240 ± 0.0140** | 0.3276 | **−0.71%, p_Holm = 0.005, 5/5 folds** | Darwin 0.2865 (MODNet 0.3327) | 2 of 13 |
| expt_is_metal | AUC 0.9819 | AUC 0.9825 | **AUC 0.9825; BA 0.9370** | AUC 0.9826 | Δ ≈ 0 (correction on in 1/5 folds) | Darwin BA 0.9598 (AMMExpress 0.9209) | 2 of 8 (by BA) |
| glass | AUC 0.9548 | AUC 0.9567 | **AUC 0.9567; BA 0.8841** | AUC 0.9559 | Δ = 0 (correction never selected) | MODNet BA 0.9603 (AMMExpress 0.8607) | 2 of 8 (by BA) |

- **Primary endpoint.** 1 of 4 tasks improved (expt_gap, about 0.7% of the MAE; real but small). The leaderboard leader is not beaten on any task.
- **Support strata.** The naive correction hurts in the nearest-neighbour quartile (expt_gap +0.9%, steels +6.8% MAE) and helps at larger distances. The fold-matched variant did not fix this: on expt_gap it was worse than naive (+1.1%, p = 0.005).
- **Permutation control.** The correction is still selected under permuted labels (a shrinkage effect), so this control is diagnostic only.
- **Classification comparison.** The board's classification "rocauc" equals balanced accuracy for label-based entries, so we compare BA with BA. Our probability AUCs are not ranked.
- **Files.**
  - Results: `results/final/{stats.json, summary.json, metrics_per_fold.csv, support_strata.csv}`.
  - Unsubmitted Matbench files: `results/final/matbench_submission/{labels,probabilities}/results.json.gz` (`is_valid=True`).
- **Wall clock** (5 workers × 4 threads, deterministic LightGBM): run 2 about 20 min (caches reused). Run 1 from scratch about 1 h 40 min, including a 35-min Modern Standby pause.

## Done
- **Environment**
  - Experiment dependencies are installed in `.venv`: scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, numpy 2.5.4, pandas 2.3.3, matminer 0.10.1, pymatgen.
  - `matbench==0.6` was installed with `--no-deps`, because its pins (scikit-learn 1.0, monty 2021) cannot build on Python 3.12.
  - matbench ships a stray top-level `scripts/` package. It shadowed idea-spark's own `scripts` package, so I renamed it inside `.venv` to `_matbench_devscripts_unused`. matbench itself still imports fine.
- **Data**
  - All 4 tasks are cached with their official folds (`src/data.py` → `data/cache/`, git-ignored).
  - Magpie (132) + element-fraction (103) features.
  - 21 extra matminer descriptors (Stoichiometry, ValenceOrbital, IonProperty, BandCenter, AtomicOrbitals) are label-free.
    - 14/14/21 rows (expt_gap / is_metal / glass) had a featuriser failure. They were imputed with the dataset-wide feature median (label-free).
- **paper-search** (`research/paper_search/allinone.md`): run 1 was degraded; run 2 used full connectors.
- **scoop-check** (`research/scoop_check/`): **Level 3 — Medium Overlap**.
  - The Miccio 2025 risk is resolved. The owner supplied the PDF; it was read; it is Level 4 (polymer transfer learning, no stacking or residual correction). It will be cited in related work.
- **idea-spark** (`ideaspark_run/matbench-composition-stack/`)
  - Phase 0: 132 records → 74 after the relevance partition, plus 7 connector-verified host refs; full text for 7 of 12 papers.
  - Phase 1: proceed.
  - Phase 2 candidate: "Fold-Matched Local Residual Correction for Cross-Fitted Composition Stacks". It adopts the user's direction. It changes how the residual transfer is done, removing the absorption term M_q by using pair-excluded refits.
  - Coherence gate: no blocking findings. Critique: revise (tactical wording edits).
- **Pipeline** (`src/`)
  - `pipeline.py`: cross-fitting; simplex stack (L2 or L1); naive local correction; fold-matched correction; inner-CV thresholds.
  - `experts.py`, `design.py`, `run_final.py`, `evaluate.py`.
- **Leaderboard verified 2026-10-18** (`results/leaderboard_snapshot.json`). Two corrections to CLAUDE.md:
  - **steels: the leader is TPOT-Mat, 79.95 MAE**, then AutoML-Mat 82.30. MODNet v0.1.12 (87.76) is third.
  - **expt_is_metal: the leader is Darwin, 0.9598** (AMMExpress 0.9209 is second).
  - expt_gap: Darwin 0.2865. glass: MODNet 0.9603.
  - For classification, the board's "rocauc" equals balanced accuracy for these entries. That is consistent with hard-label submissions, so our probability ROC-AUC is not comparable. We will compare balanced accuracy with balanced accuracy.

## Key numbers (DESIGN ONLY: inner CV on fold 0's training rows, 5 pseudo-folds; not official results)
| Task | Best single | Global stack | Stack + naive local correction |
|---|---|---|---|
| steels (v3, MAE) | 85.5 | 87.9 | 87.4 |
| expt_gap (v5: extra features + L1 stack, MAE) | 0.341 | 0.343 | 0.339 |
| glass (v3, prob. AUC) | 0.944 | 0.948 | 0.948 (correction never selected) |
| expt_is_metal (v3, prob. AUC) | 0.971 | 0.974 | 0.974 (never selected) |

## Problems / blockers
- Two design pseudo-folds took about 50 minutes of wall time at the same moment, probably machine sleep or contention. There was no effect on results.
- Firecrawl credits are exhausted, so WebFetch/WebSearch were used instead.

## idea-spark outcome
- The run ended `needs_work`. A navigator inconsistency caused it: the post-revision reviewer is instructed to report empty `confirmed_terms`, but the novelty re-audit gate then requires non-empty terms.
- I ran the term-refresh collision manually (240 hits) and found no fold-matched prior art by title scan.
- The reviewed candidate (post-revision pass) is summarised in `research/STAGE1_SUMMARY.md`. Phase 4 cards were not rendered.

## Next step
1. **Owner review** of `manuscript/main.pdf` and `supplementary.pdf`. Rebuild with `python -m src.make_paper_assets && python -m src.make_paper_assets2`, then `cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
2. **Owner decisions** (not done by autopilot): make the repo public at submission; submit to the Matbench leaderboard (the labels variant matches the board's semantics; the probabilities variant would show a true ROC-AUC); and Stage 4 (ResearchStudio-Reel assets).
3. **Optional.** Verify the journal's metrics and APC on its site (the Elsevier page returned 403 to WebFetch).
