<!-- DRAFT reply for matbench PR #372. NOT posted. Numbers from results/final_a4/stats.json. -->
@liuxiaotong15 thank you, both points are fair. I made both changes, reran all 20 task-folds from scratch and replaced the entry.

**1. Median filling.** Fixed. The two descriptor blocks that were filled with all-row medians (the `extra` matminer block and the CBFV Oliynyk block) are now built with the missing cells left empty. The medians are computed inside every model fit from that fit's training rows only: the training fold for the final refit, and the inner-training part for every inner-CV fit, including cross-fitting and every Optuna trial (`xstack/experts4.py`, `xstack/features4.py`). No statistic is computed over test rows or inner-validation rows. Cells affected: steels 0; `expt_gap` 2,824; `expt_is_metal` 3,117; glass 195.

**2. DFT block.** Removed. No candidate contains it, the DFT module is not shipped, and no data outside the task's own training fold is used. The entry is now composition-only, so I renamed it from DFT-Stack to **XStack** (`benchmarks/matbench_v0.1_XStack/`). The DFT-informed numbers are withdrawn from this PR and only mentioned in `info.json` as earlier runs that used external data.

**Results (mean over the 5 official folds):**

| Task | Metric | XStack (this push) | Previous file (with DFT) |
|---|---|---|---|
| `matbench_steels` | MAE (MPa) | 80.20 | 80.32 |
| `matbench_expt_gap` | MAE (eV) | 0.2773 | 0.2352 |
| `matbench_expt_is_metal` | balanced accuracy | 0.9439 | 0.9547 |
| `matbench_glass` | balanced accuracy | 0.8902 | 0.8937 |

As expected, `expt_gap` and `expt_is_metal` are clearly worse without the DFT block. Classification is recorded as hard labels, so the board's ROC-AUC equals balanced accuracy; the probability ROC-AUC of the same predictions is 0.9850 (`expt_is_metal`) and 0.9608 (glass).

**What else changed: nothing.** Everything else is the previous protocol. Each fold still chooses, by inner CV on its training rows, between a small and a full composition-only feature set (for `expt_gap` also the hurdle experts); the selected candidate and all candidates' inner-CV scores are in each fold's `parameters`. The bot suggested the small set for all folds; it is a candidate in every fold and was selected in 4 of 20.

**Disclosed.** This is the fourth scored run on the official folds (all in `info.json`), so it is not blind to test results; its two changes are exactly the ones you asked for, it was frozen by hash before any test prediction, and it was scored once. The optional local correction was rejected by its inner-CV gate in all 20 folds, so the entry is the plain stack.

**Compute.** CPU only, one i7-14650HX, every model fitted from scratch, 27.6 h of job time. The machine slept for 14.3 h during the run; five jobs resumed unchanged, and their `wall_time_s` has the sleep subtracted, with the raw value kept as `wall_time_s_raw`.

**In this push:** folder renamed to `matbench_v0.1_XStack`; new `results.json.gz` (`is_valid`, `is_recorded`), `info.json`, `run_submission.py`, `xstack/`, README and PR description.
