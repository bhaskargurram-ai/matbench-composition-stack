# DFT-informed stacked ensembles on Matbench composition tasks

Code and results for a pre-registered, fully nested evaluation of cross-fitted stacked ensembles on the four
composition tasks of Matbench v0.1: `matbench_steels`, `matbench_expt_gap`, `matbench_expt_is_metal` and
`matbench_glass`.

## Protocol
- Only the official Matbench folds are used.
- Every model, feature, hyperparameter, stacking and threshold choice is made by cross-validation inside each official training fold.
- The analysis plan and the code were frozen before any test scoring. The SHA-256 hashes are in `results/freeze_hashes*.txt`, and the plan is in `ANALYSIS_PLAN.md` (original plan plus Amendments 1 and 2, and a post-hoc ablation in Section 12).
- Test-fold predictions were committed before they were scored. The commit history is the audit trail.

## Final results (Amendment 2, official folds, mean over 5 folds)

| Task | Metric | Result |
|---|---|---|
| matbench_steels | MAE (MPa) | 80.10 |
| matbench_expt_gap | MAE (eV) | 0.2352 |
| matbench_expt_is_metal | balanced accuracy | 0.9547 |
| matbench_glass | balanced accuracy | 0.8937 |

**External data.** A block of Materials Project PBE band gaps and formation energies, taken from matminer's
`matbench_mp_gap` and `matbench_mp_e_form` and matched by reduced formula, is used as transfer-learning input. No
experimental labels from external sources are used. `results/ablation_summary.json` reports the pipeline without
this block.

## Layout
| Path | Contents |
|---|---|
| `src/` | Data loading and features (`data.py`, `features_*.py`, `external_dft.py`), experts (`experts*.py`), cross-fitting and stacking (`pipeline*.py`), final runs (`run_final*.py`, `launch_*.py`), scoring (`evaluate*.py`) |
| `results/final/` | Pre-registered run 1 (Amendment 1): predictions, metrics and statistics |
| `results/final_a2/` | Pre-registered run 2 (Amendment 2): predictions, metrics, statistics, and Matbench result files |
| `results/ablation_*` | Post-hoc ablations (no DFT block; no hurdle experts) |
| `results/design*/` | Design-phase logs (inner CV on training rows only) |
| `leaderboard_submission/` | Self-contained Matbench leaderboard submission (`matbench_v0.1_DFTStack`) |

## Reproduce
Python 3.12, CPU only. Install the pinned requirements in
`leaderboard_submission/matbench_v0.1_DFTStack/requirements.txt`, then run
`python leaderboard_submission/matbench_v0.1_DFTStack/run_submission.py`.
The full run took about 14 worker-hours (5 workers × 4 threads).

## Author
Bhaskar Gurram, Independent Researcher. A manuscript describing this work is in preparation.
