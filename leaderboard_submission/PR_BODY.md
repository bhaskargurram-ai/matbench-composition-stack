<!-- Draft PR description (not submitted). Label to request: new_benchmark -->
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

**Fold use, stated exactly.** Fold-internal (training rows of each official fold only): the Optuna hyperparameters,
the simplex stack weights, the optional local residual correction and its gate (kept in 0 of 20
task-folds) and the classification decision thresholds; test folds are only predicted and recorded. Not
fold-internal for folds 1-4: the coarse design choices fixed in the shipped configuration (the feature-block set BLOCKS2; for matbench_expt_gap the hurdle experts and the removal of the element-fraction extra-trees expert; the Optuna budget of 20 trials; dropping the random-forest expert; restricting CatBoost to the F1 feature set; the fixed SVC C and gamma) were
made in a design phase on fold 0's training rows, which contain the test rows of folds 1-4. Fold 0 is the only
fold whose every choice was made without its test rows; fold-0 vs 5-fold mean: matbench_steels 93.36 vs 80.10 MAE; matbench_expt_gap 0.2099 vs 0.2352 MAE; matbench_expt_is_metal 0.9573 vs 0.9547 balanced accuracy; matbench_glass 0.9136 vs 0.8937 balanced accuracy.
An earlier pre-registered configuration (Amendment 1, no DFT block) was scored on the official folds on
2026-10-18 before this one was designed; its results (matbench_steels 82.80 MAE; matbench_expt_gap 0.3240 MAE; matbench_expt_is_metal 0.9370 balanced accuracy; matbench_glass 0.8841 balanced accuracy) are reported in the manuscript and kept in the
research repository. Classification tasks are recorded as hard labels at the inner-CV balanced-accuracy
threshold (pre-specified before scoring), so the board's ROC-AUC equals balanced accuracy; the probability
ROC-AUC of the same predictions is matbench_expt_is_metal 0.9895 ± 0.0008, matbench_glass 0.9606 ± 0.0080. The configuration was frozen by SHA-256 hash before the test folds
were predicted with it, and they were predicted once.

## Results (mean ± std over the 5 official folds, as computed by matbench, i.e. population std)
| Task | Metric | Score |
|---|---|---|
| `matbench_steels` | MAE | 80.1045 ± 9.1165 |
| `matbench_expt_gap` | MAE | 0.2352 ± 0.0149 |
| `matbench_expt_is_metal` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.9547 ± 0.0030 |
| `matbench_glass` | ROC-AUC as computed by matbench from hard labels (= balanced accuracy) | 0.8937 ± 0.0141 |

## Included files (`benchmarks/matbench_v0.1_DFTStack/`)
- `results.json.gz`, `info.json`: required files (`MatbenchBenchmark.from_file(...)`: `is_valid=True`,
  `is_recorded=True` for all 4 tasks).
- `run_submission.py`: source that reproduces the run with `task.record` calls.
- `dftstack/`: the pipeline, experts, feature blocks, DFT feature builder and frozen configuration (small `.py`
  files) imported by `run_submission.py`.
- `README.md`, `requirements.txt`.

## Compute
CPU-only (Intel(R) Core(TM) i7-14650HX, 16 cores / 24 threads, no GPU), 5 workers x 4 threads; 14.3 worker-hours for all
20 task-folds. Per-fold wall time, threads, CPU model and per-expert fit seconds are recorded in the `parameters`
of every fold in `results.json.gz`.

## Paper and code
Manuscript in preparation for journal submission; preprint to follow (no DOI yet). Research repository: https://github.com/bhaskargurram-ai/matbench-composition-stack
