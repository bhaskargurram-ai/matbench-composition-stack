# Design log (inner CV on fold 0's training rows only; 5 pseudo-folds, seed 2026)

Every configuration tried is listed. Values are means over pseudo-folds (MAE for steels/expt_gap; probability ROC-AUC for is_metal/glass). v1/v2 were debugging runs on steels (KRR/MLP target scaling fixed after v1).

| tag | task | config | status | best_single | mean_ensemble | stack | stack_local | stack_local_magpie | stack_local_frac | stack_fm | single_local |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | steels | None | complete | 85.5049 | 112.8071 | 86.5044 | 85.3291 | 89.5487 | 85.3291 | - | 85.0097 |
| v2 | steels | None | complete | 85.5049 | 115.5176 | 86.8727 | 85.3929 | 91.5492 | 85.3929 | - | 85.0097 |
| v3 | expt_gap | None | complete | 0.4201 | 5.6107 | 0.4164 | 0.4131 | 0.4164 | 0.4131 | - | 0.4161 |
| v3 | expt_is_metal | None | complete | 0.9708 | 0.9713 | 0.9737 | 0.9737 | 0.9737 | 0.9737 | - | 0.9708 |
| v3 | glass | None | complete | 0.9442 | 0.9411 | 0.9481 | 0.9481 | 0.9481 | 0.9481 | - | 0.9442 |
| v3 | steels | None | complete | 85.5049 | 109.4187 | 87.8810 | 87.4297 | 92.3427 | 87.4297 | - | 85.0097 |
| v4 | expt_gap | {'lgbm_l1_expert': True, 'stack_loss': 'l1'} | complete | 0.4240 | 0.4552 | 0.4129 | 0.4076 | 0.4099 | 0.4094 | - | 0.4169 |
| v4 | steels | {'lgbm_l1_expert': True, 'stack_loss': 'l1'} | complete | 85.5049 | 106.3171 | 86.7854 | 87.8015 | 91.2918 | 87.8015 | - | 85.0097 |
| v5 | expt_gap | {'lgbm_l1_expert': True, 'stack_loss': 'l1', 'extra_features': True} | complete | 0.3414 | 0.4088 | 0.3433 | 0.3394 | 0.3414 | 0.3394 | - | 0.3379 |
| v6 | expt_gap | {'fold_matched': True, 'extra_features': True, 'stack_loss': 'l1', 'lgbm_l1_expert': True} | partial (1/5 pseudo-folds; stopped at freeze) | 0.3623 | 0.4305 | 0.3638 | 0.3615 | 0.3610 | 0.3615 | 0.3604 | 0.3604 |
| v6 | expt_is_metal | {'fold_matched': True, 'extra_features': True} | partial (4/5 pseudo-folds; stopped at freeze) | 0.9792 | 0.9792 | 0.9807 | 0.9807 | 0.9807 | 0.9807 | 0.9805 | 0.9793 |
| v6 | glass | {'fold_matched': True, 'extra_features': True} | partial (3/5 pseudo-folds; stopped at freeze) | 0.9447 | 0.9381 | 0.9476 | 0.9476 | 0.9476 | 0.9476 | 0.9469 | 0.9447 |
| v6 | steels | {'fold_matched': True, 'extra_features': True, 'stack_loss': 'l1'} | complete | 86.7901 | 116.5308 | 87.4671 | 86.8297 | 91.1912 | 86.8297 | 87.5766 | 86.9557 |
