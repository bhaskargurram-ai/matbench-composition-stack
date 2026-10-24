@ardunn thank you for the detailed review. It is correct on every point. Answers to the seven questions, what changed in this push, and two options for you to choose from.

**1. Frozen choices reused on folds 1-4.** Correct. The coarse design choices in `frozen_config2.py` / `experts2.py` (feature-block set, hurdle experts and the `et_frac` drop for `expt_gap`, 20 Optuna trials, no random forest, CatBoost on F1 only, fixed SVC C/gamma) were made on fold 0's training rows, which contain the test rows of folds 1-4. Everything else (Optuna, stack weights, thresholds, correction gate) is fold-internal. I cannot show that folds 1-4 are unchanged without re-running. Fold 0 is the only fully clean fold; fold 0 vs 5-fold mean: steels 93.36 vs 80.10 MAE (the hardest fold for every entry), expt_gap 0.2099 vs 0.2352 MAE, expt_is_metal 0.9573 vs 0.9547, glass 0.9136 vs 0.8937 balanced accuracy. `info.json` now states this exactly; the "never used for fitting or selection" sentence is gone.

**2. Earlier scoring on the official folds.** Yes. Amendment 1 (no DFT block, smaller expert pool, fully fold-internal candidate selection) was scored on the official folds on 2026-10-18, before Amendment 2 was designed, pre-registered (2026-10-19) and scored once. Amendment 1: steels 82.80 MAE, expt_gap 0.3240 MAE, expt_is_metal 0.9370, glass 0.8841 balanced accuracy (probability ROC-AUC 0.9825 / 0.9567). Both runs are in the manuscript; `info.json` now discloses this.

**3. Variants.** Two: `labels` and `probabilities`, both written at the same scoring moment from the same predictions. Recording labels was fixed in the Amendment-2 protocol before scoring (the board compares label entries by balanced accuracy). Probability ROC-AUC of the same predictions: expt_is_metal 0.9895 ± 0.0008, glass 0.9606 ± 0.0080. Scored by matbench (threshold 0.5) the probabilities file would read 0.9559 / 0.8725, slightly higher on is_metal and lower on glass than the labels file.

**4. Per-fold compute.** Added to `params` of all 20 folds in `results.json.gz`: `wall_time_s`, `n_workers`, `threads_per_worker`, `cpu_threads_used`, `cpu_model` (i7-14650HX, 16 cores / 24 threads, no GPU), `gpu`, `expert_fit_seconds` per expert, plus `pipeline`/`seed`. Predictions and scores are byte-identical before and after (checked per fold); `run_submission.py` records the same fields on a rerun.

**5. `results_rerun.json.gz`.** Only the output filename of `run_submission.py`; not shipped, and no complete independent rerun has been done yet. The README now says so.

**6. CBFV.** It fills missing cells with column medians over all rows passed, and we pass all task rows at once; `data.py` L90 does the same for the "extra" block. No scaling, no labels. Cells filled: expt_gap CBFV 2782 of 1.42M cells (422 rows), extra 42 of 96.7k; expt_is_metal 3075 of 1.52M (465 rows), extra 42; glass 132 of 1.75M (22 rows), extra 63; steels none. Disclosed in `info.json`.

**7. Structure-derived.** Added to `info.json`: the MP values are DFT calculations on crystal structures, the most stable polymorph per reduced formula plus polymorph aggregates ("DFT, structure-derived, matched by formula"), and the DFT gap / metal flag are close proxies of the expt_gap / expt_is_metal targets. Please flag the entry accordingly.

**Also in this push:** `user_metadata.algorithm` is now `DFT-Stack`; citation updated (manuscript in preparation, preprint to follow).

**Options for point 1, your choice:**
(i) merge with the disclosure as written in `info.json`; or
(ii) I re-run with the fixed design choices turned into a small candidate set selected per fold by the inner-CV stack score on the fold's training rows, as Amendment 1 did ("Amendment 3", pre-registered before scoring, train-only imputation included). About 45 worker-hours, roughly 2 days on this workstation; the results would replace the current file, whatever they are.

A full independent rerun producing `results_rerun.json.gz` is also possible. I will do whichever you prefer.
