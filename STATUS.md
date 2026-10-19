# STATUS

**Current stage:** Stage 2. The design is FROZEN: `ANALYSIS_PLAN.md` and `results/freeze_hashes.txt` were committed in dd8624b (first freeze commit df504b2; hashes re-recorded on LF bytes). The final official-fold run is in progress.

**Final-run setup.**
- Command: `python -m src.launch_final --workers 5`. It runs one process per (task, fold), with 5 workers × 4 threads (expert `n_jobs=4`; OMP/BLAS/MKL capped at 4).
- LightGBM is deterministic (`deterministic=True`, `force_col_wise=True`).
- Outputs: `results/final/<task>/fold<k>_*`; wall times in `results/final/timing.jsonl`.
- Started 2026-10-18 21:02 (local).
- The machine has 32 GB RAM (about 10 GB free at launch).

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
1. Wait for the final run to finish. To resume after an interruption: `python -m src.launch_final --workers 5`; it skips folds that are already done.
2. Run `python -m src.evaluate --submission`.
3. Write the manuscript in `manuscript/`.
