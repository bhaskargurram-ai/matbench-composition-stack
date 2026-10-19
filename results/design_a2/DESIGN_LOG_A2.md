# Amendment-2 design log (fold 0's TRAINING rows only)

Quick 5-fold CV of a default LightGBM on fold-0 training rows. Values are MAE for regression and probability ROC-AUC for classification; logs are in the session.

| Task | Base + extra | + DFT block | + DFT + new blocks |
|---|---|---|---|
| expt_gap | 0.3735 | 0.3015 | 0.3016 |
| expt_is_metal | 0.9810 | 0.9878 | 0.9894 |

**Glass** (no DFT block):

| Features | ROC-AUC |
|---|---|
| base + extra | 0.9430 |
| + alloy blocks | 0.9467 |
| + all new blocks | 0.9496 |

**Steels.** Feature blocks changed little (MAE 88.5–91.3). Extra trees on element fractions only reached 79.6–80.0 MAE, against 83.8–85.0 on F1.

**expt_gap hurdle model** (5-fold CV, LightGBM, F2-like features):

| Model | MAE |
|---|---|
| Plain L1 | 0.3054 |
| Hurdle, median rule | 0.2526 |
| Hurdle, expectation | 0.2514 |

**Full pipeline, one pseudo-outer fold** (train 80% / dev 20% of fold-0 training rows):

| Design round | Task | Stack | Final (stack_local) | Other |
|---|---|---|---|---|
| a2v3 | expt_gap | 0.2530 MAE | 0.2530 | |
| a2v3 | expt_is_metal | ROC-AUC 0.9915 | | balanced accuracy 0.9568 |
| a2v3 | glass | ROC-AUC 0.9531 | | balanced accuracy 0.8772 |
| a2v3 | steels | 83.38 MAE | 95.02 | the ungated correction hurt |
| a2v4 (correction gated) | steels | | 83.38 | |

**CrabNet.** Dev MAE 0.419 on expt_gap, at about 23 min per fit. Dropped.

**MODNet.** A single fit took more than 40 min. Dropped.
