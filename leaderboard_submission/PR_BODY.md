<!-- PR description. Label: new_benchmark -->
# New benchmark: XStack (composition-only) on the 4 matbench_v0.1 composition tasks

**Label:** `new_benchmark`

**Changed on 2026-11-17 at the maintainers' request.** This entry is now composition-only. The Materials Project DFT
feature block of the earlier versions of this PR is removed, the entry is renamed from DFT-Stack to XStack
(`benchmarks/matbench_v0.1_XStack/`), and missing descriptor cells are imputed from training rows only.

## Brief description of the algorithm
A cross-fitted, simplex-weighted stack of composition experts (Optuna-tuned LightGBM, XGBoost, CatBoost, extra
trees, kernel ridge / RBF SVM, k-NN, MLP; hurdle experts for the zero-inflated `matbench_expt_gap`) on label-free
matminer/CBFV composition descriptors. **No external data**: every input is computed from the row's own formula.

**Fold use, stated exactly.** Every choice is made inside the training rows of each official fold: the Optuna
hyperparameters, the simplex stack weights, the optional local residual correction and its gate (kept in 0 of
20 task-folds), the classification thresholds, and the feature-block set (for `matbench_expt_gap` also the hurdle
experts and the element-fraction extra-trees expert), selected per fold from a small candidate set by the inner-CV
score of the stack on the training rows (candidates: matbench_steels: 2; matbench_expt_gap: 8; matbench_expt_is_metal: 2; matbench_glass: 2). Selected per fold: matbench_steels: fold 0 F_min|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_min|hurdle=off|et_frac=in, fold 4 F_min|hurdle=off|et_frac=in; matbench_expt_gap: fold 0 F_all|hurdle=on|et_frac=in, fold 1 F_all|hurdle=on|et_frac=in, fold 2 F_min|hurdle=on|et_frac=in, fold 3 F_all|hurdle=on|et_frac=in, fold 4 F_all|hurdle=on|et_frac=in; matbench_expt_is_metal: fold 0 F_all|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_all|hurdle=off|et_frac=in, fold 4 F_all|hurdle=off|et_frac=in; matbench_glass: fold 0 F_all|hurdle=off|et_frac=in, fold 1 F_all|hurdle=off|et_frac=in, fold 2 F_all|hurdle=off|et_frac=in, fold 3 F_all|hurdle=off|et_frac=in, fold 4 F_all|hurdle=off|et_frac=in.

**Imputation.** Missing descriptor cells (matbench_steels: 0 cells in 312 rows x 329 columns; matbench_expt_gap: 2824 cells in 4604 rows x 329 columns; matbench_expt_is_metal: 3117 cells in 4921 rows x 329 columns; matbench_glass: 195 cells in 5680 rows x 329 columns) are filled with column medians computed inside every model fit
from that fit's training rows only, inner-CV fits and Optuna trials included.

**Disclosed.** The candidate set and a few coarse settings (Optuna budget, dropped experts, CatBoost feature set,
SVC C and gamma) come from a design phase on fold 0's training rows. Earlier versions were scored on the official
folds before this one, each pre-registered and scored once: Amendment 1 (composition-only, smaller pool; matbench_steels 82.80; matbench_expt_gap 0.3240; matbench_expt_is_metal 0.9370; matbench_glass 0.8841);
Amendments 2 and 3 (with the external DFT block, the earlier contents of this PR; matbench_steels 80.10; matbench_expt_gap 0.2352; matbench_expt_is_metal 0.9547; matbench_glass 0.8937 and matbench_steels 80.32; matbench_expt_gap 0.2352; matbench_expt_is_metal 0.9547; matbench_glass 0.8937); a
no-DFT ablation of Amendment 2 (matbench_steels 80.06; matbench_expt_gap 0.2724; matbench_expt_is_metal 0.9425; matbench_glass 0.8922). So this entry was not designed blind to test-fold results; its two
changes against Amendment 3 are the ones requested here. Classification is recorded as hard labels at the inner-CV
balanced-accuracy threshold, so the board's ROC-AUC equals balanced accuracy; the probability ROC-AUC of the same
predictions is matbench_expt_is_metal 0.9850 ± 0.0020, matbench_glass 0.9608 ± 0.0080.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score (this entry, composition-only) | For reference: withdrawn DFT-informed version |
|---|---|---|---|
| `matbench_steels` | MAE | 80.2043 ± 9.3679 | 80.32 |
| `matbench_expt_gap` | MAE | 0.2773 ± 0.0131 | 0.2352 |
| `matbench_expt_is_metal` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.9439 ± 0.0034 | 0.9547 |
| `matbench_glass` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.8902 ± 0.0145 | 0.8937 |

## Included files (`benchmarks/matbench_v0.1_XStack/`)
- `results.json.gz`, `info.json`: required files (`is_valid=True`, `is_recorded=True` for all 4 tasks).
- `run_submission.py`: reproduces the run with `task.record` calls.
- `xstack/`: the pipeline modules imported by `run_submission.py` (no DFT feature builder).
- `README.md`, `requirements.txt`.

## Compute
CPU-only, no GPU. Every expert fitted from scratch on one Intel(R) Core(TM) i7-14650HX workstation (5 jobs x 4 threads), 27.6 h of
job wall time in total; no cached fit of an earlier run reused. Per-fold wall time, threads, CPU model and
per-expert fit seconds are in the `parameters` of every fold in `results.json.gz`.

## Paper and code
Manuscript under review at Computational Materials Science (no DOI or preprint yet). Research repository: https://github.com/bhaskargurram-ai/matbench-composition-stack
