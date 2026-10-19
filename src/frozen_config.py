"""Frozen per-task configuration for the final official-fold evaluation (see ANALYSIS_PLAN.md).
Chosen on inner CV of fold 0's training rows only (results/design/, logs/design_*.log)."""
SEED = 0       # seed for inner KFold splits and all stochastic experts
N_INNER = 5    # inner cross-fitting folds J

_REG = {"extra_features": True, "fold_matched": True, "stack_loss": "l1", "lgbm_l1_expert": True}
_CLF = {"extra_features": True, "fold_matched": True}

CFG = {
    "matbench_steels": {"extra_features": True, "fold_matched": True, "stack_loss": "l1"},  # n<1000: lgbm_all is already L1
    "matbench_expt_gap": dict(_REG),
    "matbench_expt_is_metal": dict(_CLF),
    "matbench_glass": dict(_CLF),
}
