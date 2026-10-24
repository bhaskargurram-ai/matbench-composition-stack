# Amendment 3 (DRAFT pre-registration; not yet frozen, not yet run)

Purpose: remove the one non-fold-internal step of Amendment 2 that the matbench PR #372 review identified. The
coarse design choices that Amendment 2 fixed from a design phase on fold 0's training rows become a small
candidate set, and the choice among the candidates is made inside each official training fold T_k, exactly as
Amendment 1 (ANALYSIS_PLAN.md Section 8) did. Nothing else changes. If this amendment is run, its results replace
the current leaderboard file whatever they are; the Amendment-1 and Amendment-2 results stay reported.

Status: draft written 2026-10-24, to be frozen (SHA-256 hashes, commit) before any official test fold is
predicted under it. No Amendment-3 code exists yet; no test fold has been touched since Amendment-2 scoring.

## 1. What stays fixed (unchanged from Amendment 2, Section 10)
- Official Matbench v0.1 folds; J = 5 inner folds; seed 0; Optuna 20 TPE trials for LightGBM (default first).
- Expert pool and their fixed settings (XGBoost F2, CatBoost F1, extra trees F1/F2/frac, KRR/SVC frac+magpie with
  the fixed C/gamma, kNN frac, MLP F1; hurdle experts available for expt_gap). No random forest, no CrabNet, no MODNet.
- Simplex stack (L1 / Brier), nested-CV-gated local correction, balanced-accuracy threshold on training OOF,
  final arm `stack_local`, hard labels recorded for classification (labels variant), probabilities kept alongside.
- Disclosed DFT transfer block (matminer matbench_mp_gap / matbench_mp_e_form, matched by reduced formula,
  structure-derived, `dft_matched` indicator).

Fixed items that remain design-phase choices and are disclosed as such (they define the candidate set, not the
winner): the Optuna budget, the dropped experts, CatBoost on F1, the SVC C/gamma. These are coarse budget/speed
decisions; they are the "candidate set constructed while looking at fold-0 inner-CV results" disclosure that
Amendment 1 also carried.

## 2. Candidate set per task (from results/design_a2/DESIGN_LOG_A2.md alternatives)
Feature-block sets, named as in the design log:
- `B0` = base + extra (the Amendment-1 features: Magpie, element fractions, 21 extra descriptors);
- `B1` = B0 + DFT block;
- `B2` = B1 + new blocks (ep_deml, ep_matminer, ep_megnet, meredig, miedema, yang, ape, wen, tmetal,
  cbfv_oliynyk) = the shipped `BLOCKS2`.

| Task | Candidates | Count |
|---|---|---|
| matbench_steels | {B0, B1, B2} | 3 |
| matbench_expt_is_metal | {B0, B1, B2} | 3 |
| matbench_glass | {B0, B1, B2} | 3 |
| matbench_expt_gap | {B0, B1, B2} x hurdle {off, on} x et_frac {in, out} | 12 |

F1/F2 for the experts are derived from the candidate's block set as in Amendment 2 (F1 = base + extra + dft when
present; F2 = all blocks of the candidate). With B0 the DFT-dependent columns are simply absent.

## 3. Selection rule (fold-internal, automatic)
For each official fold k, using only T_k:
1. Cross-fit every expert of every candidate on T_k (J = 5, seed 0, same inner splits for all candidates).
2. Fit the simplex stack on the candidate's OOF predictions; the candidate's score is the OOF score of the
   uncorrected stack on T_k (MAE for regression, ROC-AUC for classification), the same criterion as Amendment 1.
3. Pick the best candidate; ties go to the earlier (simpler) candidate in the order B0 < B1 < B2, hurdle off <
   on, et_frac in < out.
4. Then, for the chosen candidate only: correction selection and nested gate, threshold, final predictions,
   exactly as Amendment 2.
5. Record in `fold<k>_meta.json["design_selection"]` every candidate's OOF stack score and the choice.

## 4. Train-only imputation (new; closes the transductive-imputation point)
- `extra` block (data.py): drop the `fillna(median over all rows)`; keep NaN in the cached block and impute
  inside `run_split` with column medians of the T_k rows (applied to T_k and to the test rows of fold k).
- CBFV Oliynyk block: call `generate_features` per row group so that no cross-row statistic is used, or
  equivalently compute the features with NaN preserved and impute with T_k medians as above. (CBFV's own median
  fill is the only cross-row step; sum/avg/dev/range/mode features are per-formula.)
- All-NaN column dropping is done on T_k only.
- Expected effect: negligible (0-0.2% of cells, see results/final_a2/imputation_nan_counts.json), but the
  pipeline becomes strictly inductive.

## 5. Compute recording
Every `task.record` call stores `wall_time_s`, `n_workers`, `threads_per_worker`, `cpu_threads_used`, `cpu_model`,
`gpu`, `expert_fit_seconds` (per expert, per candidate), `design_selection` (chosen candidate), `pipeline`, `seed`.
The launcher keeps writing `timing.jsonl`.

## 6. Scoring (once)
- Freeze: new `freeze_hashes_amendment3.txt`, commit, then `launch_final --amendment3 --workers 5`.
- Predictions for all 20 folds committed before scoring; `evaluate` run once; both labels and probabilities
  files written; the labels file replaces `leaderboard_submission/matbench_v0.1_DFTStack/results.json.gz`.
- No iteration on test scores. Any later bug fix is reported as a deviation with both versions shown.
- Reporting: Amendment 3 alongside Amendments 1 and 2 and the leaders; same paired bootstrap as Section 10.

## 7. Cost estimate (from results/final_a2/timing.jsonl, 5 workers x 4 threads, i7-14650HX)
Mean wall time per fold of the full Amendment-2 pipeline (one candidate = B2): steels 498 s, expt_gap 3678 s,
expt_is_metal 2795 s, glass 3299 s (14.3 worker-hours for 20 folds).

| Task | Candidates needing new expert fits | Worker-hours (upper bound) |
|---|---|---|
| steels | 3 block sets x 5 folds x 0.14 h | 2.1 |
| expt_is_metal | 3 x 5 x 0.78 h | 11.6 |
| glass | 3 x 5 x 0.92 h | 13.7 |
| expt_gap | 3 block sets x 5 folds x 1.02 h (hurdle on/off and et_frac in/out only add or drop experts whose OOF predictions are cached, so the 12 candidates cost about 3 full fits) | 15.3 |
| total | | about 43 worker-hours (upper bound without caching: about 88) |

B0 and B1 have fewer columns than B2, so the figures above are conservative. With 5 workers that is roughly 9-10 h
of machine time for the expert fits plus the selection, final refits, scoring and checks: about 2 days end to end on
this workstation. Expert caches from Amendment 2 are reused only where the expert's inputs and configuration are
byte-identical and were computed from T_k alone (the B2 candidate before the imputation change does not qualify, so
the estimate assumes no reuse).
