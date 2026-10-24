<!-- PR description. Label to request: new_benchmark -->
# New benchmark: DFT-Stack on the 4 matbench_v0.1 composition tasks

**Label:** `new_benchmark` (please add it; I cannot set labels on this repository)

## External data disclosure (please read first)
This entry **uses external computed data**: Materials Project PBE DFT band gaps and formation energies, from
matminer's `matbench_mp_gap` and `matbench_mp_e_form` datasets, matched to each composition by reduced formula
and used as transfer-learning input features. **No experimental external labels** are used. On
`matbench_expt_gap` and `matbench_expt_is_metal` most compositions are matched; on `matbench_steels` and
`matbench_glass` almost none are (matbench_steels: 0/312 (0%); matbench_expt_gap: 3347/4604 (73%); matbench_expt_is_metal: 3613/4921 (73%); matbench_glass: 60/5680 (1%)). The entry is therefore not a like-for-like comparison with
composition-only entries that use no external data, and we would welcome it being flagged as such on the board.

## Brief description of the algorithm
A cross-fitted, simplex-weighted stack of composition experts (Optuna-tuned LightGBM, XGBoost, CatBoost, extra
trees, kernel ridge / RBF SVM, k-NN, MLP; hurdle experts for the zero-inflated `matbench_expt_gap`) on
label-free matminer/CBFV composition descriptors plus the DFT block above. The DFT values are structure-derived
(each Materials Project entry is a DFT calculation on a crystal structure; per reduced formula the most stable
polymorph plus polymorph aggregates are used), so the DFT gap / metal flag are close proxies of the
`matbench_expt_gap` / `matbench_expt_is_metal` targets.

**Fold use, stated exactly.** Every choice is made inside the training rows of each official fold: the Optuna
hyperparameters, the simplex stack weights, the optional local residual correction and its gate (kept in 0 of
20 task-folds), the classification decision thresholds, and the coarse design choices (feature-block set;
for `matbench_expt_gap` the hurdle experts and the element-fraction extra-trees expert), which are selected per
fold from a small candidate set by the inner-CV score of the stack on the training rows (matbench_steels: 2 candidates; matbench_expt_gap: 12 candidates; matbench_expt_is_metal: 3 candidates; matbench_glass: 2 candidates). Selected per
fold: matbench_steels: fold 0 F_red|hurdle=off|et_frac=in, fold 1 F_red|hurdle=off|et_frac=in, fold 2 F_red|hurdle=off|et_frac=in, fold 3 F_red|hurdle=off|et_frac=in, fold 4 F_red|hurdle=off|et_frac=in; matbench_expt_gap: fold 0 F_full|hurdle=on|et_frac=in, fold 1 F_full|hurdle=on|et_frac=in, fold 2 F_full|hurdle=on|et_frac=out, fold 3 F_full|hurdle=on|et_frac=in, fold 4 F_full|hurdle=on|et_frac=in; matbench_expt_is_metal: fold 0 F_full|hurdle=off|et_frac=in, fold 1 F_full|hurdle=off|et_frac=in, fold 2 F_full|hurdle=off|et_frac=in, fold 3 F_full|hurdle=off|et_frac=in, fold 4 F_full|hurdle=off|et_frac=in; matbench_glass: fold 0 F_full|hurdle=off|et_frac=in, fold 1 F_full|hurdle=off|et_frac=in, fold 2 F_full|hurdle=off|et_frac=in, fold 3 F_full|hurdle=off|et_frac=in, fold 4 F_full|hurdle=off|et_frac=in. Disclosed as not fold-internal: the candidate set itself was constructed from the alternatives
explored on fold 0's training rows (the Optuna budget of 20 trials; the dropped experts (random forest, CrabNet, MODNet); restricting CatBoost to the F1 feature set; the fixed SVC C and gamma; and the composition of the candidate set itself), and a label-free, transductive median imputation of a
few hundred missing descriptor cells (details in `info.json`).
Earlier runs on the official folds, all reported: Amendment 1 (2026-10-18, no DFT block; matbench_steels 82.80 MAE; matbench_expt_gap 0.3240 MAE; matbench_expt_is_metal 0.9370 balanced accuracy; matbench_glass 0.8841 balanced accuracy) and
Amendment 2 (2026-10-19, the version first submitted in this PR, design choices fixed from fold 0's training rows;
matbench_steels 80.10 MAE; matbench_expt_gap 0.2352 MAE; matbench_expt_is_metal 0.9547 balanced accuracy; matbench_glass 0.8937 balanced accuracy). Classification tasks are recorded as hard labels at the inner-CV balanced-accuracy threshold
(pre-specified before scoring), so the board's ROC-AUC equals balanced accuracy; the probability ROC-AUC of the
same predictions is matbench_expt_is_metal 0.9895 ± 0.0008, matbench_glass 0.9606 ± 0.0080. The configuration was frozen by SHA-256 hash before the test folds were predicted
with it, and they were predicted and scored once.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score (this entry, Amendment 3) | Amendment 2 (first submitted version) |
|---|---|---|---|
| `matbench_steels` | MAE | 80.3217 ± 9.1998 | 80.10 |
| `matbench_expt_gap` | MAE | 0.2352 ± 0.0149 | 0.2352 |
| `matbench_expt_is_metal` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.9547 ± 0.0030 | 0.9547 |
| `matbench_glass` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.8937 ± 0.0141 | 0.8937 |

## Included files (`benchmarks/matbench_v0.1_DFTStack/`)
- `results.json.gz`, `info.json`: required files (`MatbenchBenchmark.from_file(...)`: `is_valid=True`,
  `is_recorded=True` for all 4 tasks).
- `run_submission.py`: source that reproduces the run (per-fold candidate selection included) with `task.record`
  calls.
- `dftstack/`: the pipeline, experts, feature blocks, DFT feature builder, frozen configurations and candidate
  selection (small `.py` files) imported by `run_submission.py`.
- `README.md`, `requirements.txt`.

## Compute
CPU-only, no GPU. Selection and final predictions on one Intel(R) Core(TM) i7-14650HX workstation (16 cores / 24 threads, 5 workers x
4 threads); new expert cross-fits for the smaller feature-block candidates on the same workstation and on two cloud
CPU pods (AMD EPYC 9575F 64-Core Processor; 32 vCPU, 8 workers x 4 threads). Per-fold wall time, hosts, threads, CPU
models and per-expert fit seconds are recorded in the `parameters` of every fold in `results.json.gz`.
DETERMINISM CHECK: fold 0 of matbench_steels and of matbench_expt_gap were re-fitted from scratch on a Linux cloud CPU (AMD EPYC 9575F 64-Core Processor) with the same pinned package versions and compared with the Windows-workstation run (per-expert max |difference| of the test predictions): matbench_expt_gap: cat_F1 0.0e+00, et_F1 8.9e-16, et_F2 8.9e-16, hurdle_et_F1_exp 8.9e-16, hurdle_lgbm_F2_exp 1.7e-21, hurdle_lgbm_F2_median 0.0e+00, knn_frac 0.0e+00, krr_frac 3.9e-13, krr_magpie 5.2e-13, lgbm_F2_tuned 0.0e+00, mlp_F1 1.1e-14, xgb_F2 8.8e-01; re-tuned LightGBM parameters identical; matbench_steels: cat_F1 0.0e+00, et_F1 2.3e-13, et_F2 4.5e-13, et_frac 4.5e-13, knn_frac 0.0e+00, krr_frac 5.4e-11, krr_magpie 1.5e-11, lgbm_F2_tuned 7.8e-02, mlp_F1 1.8e-12, xgb_F2 6.6e+01; re-tuned LightGBM parameters different. Same-machine reruns are exactly reproducible (the reused caches and the from-scratch selection agree to 0.0 wherever the same candidate is selected); across CPU architectures XGBoost (tree_method=hist) and, on the small steels task, the Optuna-tuned LightGBM parameters are not bitwise reproducible, while every other expert agrees to 1e-9 or better. 
## Paper and code
Manuscript under review at Computational Materials Science (no DOI or preprint yet). Research repository: https://github.com/bhaskargurram-ai/matbench-composition-stack
