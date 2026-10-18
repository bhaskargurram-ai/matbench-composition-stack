# Step 7 - Delta and verdict
Timestamp: 2026-10-18T18:24:21-04:00

**Delta.** Unlike MODNet and the stacked band-gap ensemble of Kauwe et al., which fit one global model or meta-learner and treat its errors as exchangeable across composition space under a single split, the proposed work adds a cross-fitted, novelty-damped local OOF residual correction in composition space on top of a simplex-constrained heterogeneous stack, and evaluates it on the official Matbench folds with nested CV only. The measurable benefit is a reduction in MAE (or a gain in ROC-AUC) relative to the same global stack, isolated by ablation.

**Verdict: Level 3 — Medium Overlap.**

Justification: no retrieved or recalled paper combines all four axes. The mechanism itself is the owner's own SIG-Net (match on mechanism and insight, different domain). Its closest external relative is Shmalo's "Neural Means and Kernel Corrections for Operator Learning" (a convex stack followed by a global kernel-ridge residual correction, without novelty damping, outside materials). On the materials side, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet" and "Extracting Knowledge from DFT: Experimental Band Gap Predictions Through Ensemble Learning" share the framing and/or domain but not the local correction. The contribution is therefore a transfer-and-test study, not a new learning principle. The write-up must:
1. cite SIG-Net as the origin;
2. cite stacking (Wolpert, Breiman, Super Learner) and regression kriging as the mechanism's ancestors;
3. claim novelty only for the novelty-damped local OOF correction in composition space evaluated honestly on Matbench.

Resolved risk (update 2026-10-18): "Machine learning-driven property prediction for materials in data-scarcity scenarios: Ensemble of experts approach" (Miccio, Comput. Mater. Sci. 2025) was first unreadable (paywall). The owner then supplied the PDF and it was read. It is transfer learning from neural experts pre-trained on related polymer properties (SMILES inputs), with no stacking, OOF residuals, local correction or Matbench, so it is Level 4 — Low Overlap. It does not change the verdict, but it should be cited in related work (same target journal).
