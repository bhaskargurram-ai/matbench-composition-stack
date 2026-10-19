"""Frozen configuration for the final official-fold evaluation (ANALYSIS_PLAN.md, Amendment 1).

Design choices are made INSIDE each official training fold T_k by `src.run_final.select_design`:
every candidate below (the exact configurations explored during the fold-0 design phase, see
results/design/DESIGN_LOG.md) is cross-fitted on T_k, and the one with the best cross-fitted OOF
score of the global simplex stack is kept (ties -> the earlier, simpler candidate). The correction
hyperparameters are then selected inside T_k as before, and the fold-matched arm is always computed.
"""
SEED = 0       # seed for inner KFold splits and all stochastic experts
N_INNER = 5    # inner cross-fitting folds J

# Candidate design configurations (order = tie-break preference, simplest first).
CANDIDATES = {
    "regression": [
        {"name": "A_base_l2", "extra_features": False, "stack_loss": "l2"},                          # design v3
        {"name": "B_base_l1", "extra_features": False, "stack_loss": "l1", "lgbm_l1_expert": True},  # design v4
        {"name": "D_extra_l1", "extra_features": True, "stack_loss": "l1"},                          # design v6 (steels)
        {"name": "C_extra_l1_lgbml1", "extra_features": True, "stack_loss": "l1", "lgbm_l1_expert": True},  # v5/v6 (gap)
    ],
    "classification": [
        {"name": "A_base", "extra_features": False},   # design v3
        {"name": "C_extra", "extra_features": True},   # design v6
    ],
}
ALWAYS = {"fold_matched": True}   # the fold-matched arm is computed for the selected configuration

# Legacy single configuration used by the first (superseded, never scored) final run; kept for the record.
_REG = {"extra_features": True, "fold_matched": True, "stack_loss": "l1", "lgbm_l1_expert": True}
_CLF = {"extra_features": True, "fold_matched": True}
CFG = {
    "matbench_steels": {"extra_features": True, "fold_matched": True, "stack_loss": "l1"},
    "matbench_expt_gap": dict(_REG),
    "matbench_expt_is_metal": dict(_CLF),
    "matbench_glass": dict(_CLF),
}
