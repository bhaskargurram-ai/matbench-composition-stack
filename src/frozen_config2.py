"""Amendment-2 configuration (DRAFT until the Amendment-2 freeze; see ANALYSIS_PLAN.md Section 10)."""
SEED = 0
N_INNER = 5
BLOCKS2 = ("base", "extra", "dft", "ep_deml", "ep_matminer", "ep_megnet", "meredig", "miedema", "yang", "ape", "wen",
           "tmetal", "cbfv_oliynyk", "cbfv_mat2vec")
_COMMON = {"n_trials": 25}
CFG2 = {
    "matbench_steels": dict(_COMMON),
    "matbench_expt_gap": dict(_COMMON),
    "matbench_expt_is_metal": dict(_COMMON),
    "matbench_glass": dict(_COMMON),
}
