# DFT-Stack (matbench_v0.1)

Cross-fitted, simplex-weighted stack of composition experts with a disclosed block of Materials Project PBE DFT
features (matminer `matbench_mp_gap` / `matbench_mp_e_form`, matched by reduced formula), evaluated on the four
Matbench v0.1 composition tasks with nested cross-validation inside each official training fold.
See `info.json` for the full description and the external-data disclosure.

## Files
- `results.json.gz`: recorded benchmark (4 tasks x 5 folds; classification recorded as hard labels). The
  `parameters` of every fold carry the compute of the original run: `wall_time_s`, `n_workers`,
  `threads_per_worker`, `cpu_threads_used`, `cpu_model`, `gpu`, `expert_fit_seconds` (per expert), `pipeline`, `seed`.
- `info.json`: metadata, references, notes (fold use stated exactly, prior-run disclosure, labels variant,
  imputation note), pinned requirements.
- `run_submission.py`: reproduces the run and records every fold with `task.record` (with the same per-fold
  compute fields, measured on the rerun machine), then writes `results_rerun.json.gz`
  (`python run_submission.py [task ...]`). `results_rerun.json.gz` is only the output name of such a user rerun;
  it is not shipped in this folder, and no complete independent rerun has been performed so far.
- `dftstack/`: verbatim copies of the frozen research modules (pipeline, experts, feature blocks, DFT
  feature builder, frozen configuration). Their docstrings refer to the analysis plan and design logs, which are
  in the research repository (https://github.com/bhaskargurram-ai/matbench-composition-stack).
- `requirements.txt`: the same pinned versions as `info.json`.

## Reproduce
```
pip install -r requirements.txt   # matbench==0.6 needs --no-deps on Python 3.12, see info.json notes
python run_submission.py
```
CPU-only. Data are downloaded on first use into `./data/cache/`; expert predictions are cached in
`./rerun_cache/` so that an interrupted run resumes.
