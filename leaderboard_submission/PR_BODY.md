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
label-free matminer/CBFV composition descriptors plus the DFT block above. All hyperparameters, stack weights,
an optional local residual correction (kept in 0 of 20 task-folds) and the classification decision
thresholds are chosen by nested cross-validation inside each official training fold; test folds are only
predicted and recorded. Classification tasks are recorded as hard labels at the inner-CV balanced-accuracy
threshold, so the board's ROC-AUC equals balanced accuracy. The configuration was frozen by SHA-256 hash before
the test folds were predicted.

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
CPU-only, 5 workers x 4 threads; 14.3 worker-hours for all 20 task-folds.

## Paper and code
Manuscript under review at Computational Materials Science (no DOI or preprint yet). Research repository: https://github.com/bhaskargurram-ai/matbench-composition-stack
