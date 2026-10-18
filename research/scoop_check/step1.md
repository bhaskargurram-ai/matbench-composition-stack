# Step 1 - Decompose the novelty
Timestamp: 2026-10-18T18:24:21-04:00

Research problem: composition-based materials property prediction on small/medium data (Matbench v0.1 composition tasks: steels, expt_gap, expt_is_metal, glass).
Claimed novelty: cross-fitted heterogeneous stacking with a novelty-damped local out-of-fold (OOF) residual correction in composition space.

- **Problem framing** - Regression (steels yield strength, experimental band gap) and binary classification (metal/non-metal, glass-forming ability) from chemical formula only; official Matbench 5-fold splits; MAE / ROC-AUC; all selection by nested CV inside each training fold.
- **Core mechanism** - (i) several diverse experts (boosted trees and RF on Magpie, kernel model on element fractions, MLP, ...) cross-fitted inside each training fold to give OOF predictions; (ii) a simplex-constrained (non-negative, sum-to-one) global stack fitted on those OOF predictions; (iii) a local correction that adds a kernel-weighted average of neighbouring training points' OOF stack residuals, multiplied by a damping factor that shrinks the correction toward zero when the query is far (novel) from the training compositions; strength and neighbourhood size chosen by inner CV.
- **Key insight** - A global convex stack leaves residual structure that is local in composition space (chemical families share systematic errors); OOF residuals estimate that structure without leakage, and damping by novelty avoids extrapolating corrections to unfamiliar chemistries.
- **Application domain** - Inorganic materials, composition-only, small/medium experimental datasets (312-5,680 rows); CPU-only.
