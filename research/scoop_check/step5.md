# Step 5 - Full-paper deep dive
Timestamp: 2026-10-18T18:24:21-04:00

PDFs were fetched with `scripts/fetch_paper.sh` into `papers/` (git-ignored, not redistributed).

### [20] Shmalo, Neural Means and Kernel Corrections for Operator Learning
- Fetched: Zenodo record 10.5281/zenodo.23329780 (CC-BY). Read: abstract, Sec. 3.3, Sec. 4.8, Sec. 5 opening.
- Problem framing (verified): PDE / radiative-transfer operator-learning benchmarks (structural mechanics, OCO-2), relative L2 error, 10 seeds.
- Core mechanism (verified): six neural members combined by a convex stack; "The stacked mean is then corrected by kernel ridge regression of its residuals" with a Matern-5/2 kernel on standardised inputs, (s, lambda) chosen on a validation split; optional second correction on deep features (Sec. 3.3). The correction is a global KRR fit, not a novelty-damped neighbour average. Residual labels are training-row residuals, with an OOF-vs-in-sample sensitivity check (Sec. 4.8).
- Key insight (verified): member residuals are strongly aligned (normalised alignments 0.82-0.98), which limits gains from convex averaging; the residual kernel correction gives "a small, repeatable improvement to the stacked predictor" (Sec. 5).
- Application domain (verified): scientific-computing surrogates; no materials data.
- Venue (verified): Zenodo preprint (2026).
- Assumptions & scope: low-dimensional inputs, large n (1,250-20,000), dense design.
- Closest-passage evidence: Sec. 3.3, "The stacked mean is then corrected by kernel ridge regression of its residuals."
- Refined overlap: framing differ; mechanism partial (stack + kernel residual correction, but no novelty damping and no composition kernel); insight partial; domain differ -> 1 axis.

### [74] Kauwe, Welker, Sparks, Experimental band gap predictions through ensemble learning
- Fetched: NSF PAR accepted manuscript (par.nsf.gov/servlets/purl/10248525). Read: abstract, Sec. 2.2, Fig. 2 caption.
- Problem framing (verified): experimental band gaps of non-metals from formulas, random train/test split, RMSE.
- Core mechanism (verified): "a stacked ensemble learning system that combines machine learning from DFT calculations to improve learning on experimental data"; first-layer models (including a DFT-trained network) feed a meta-learner (Sec. 2.2). There is no local residual correction.
- Key insight (verified): models of DFT trends are useful features for experimental targets.
- Application domain (verified): experimental band gaps from formulas (the Zhuo et al. data family that matbench_expt_gap derives from).
- Venue (verified): Integrating Materials and Manufacturing Innovation 9, 213-220 (2020).
- Refined overlap: framing partial (same property, but a random split rather than Matbench); mechanism partial (stacking only); insight differ; domain match -> scored 2 (worst case).

### [10] MODNet (npj Comput. Mater. 2021)
- Fetched: nature.com PDF. Mechanism: feature selection by normalised mutual information plus a joint-learning MLP; evaluated on Matbench. No heterogeneous stacking and no residual correction.
- Refined overlap: framing match, domain match, mechanism differ, insight differ -> 2 axes.

### [28] Chang, Wang, Ertekin, Mixture of experts (npj Comput. Mater. 2022)
- Fetched: nature.com PDF. Mechanism: a gating network over frozen pretrained (CGCNN) feature extractors; 19 regression tasks; random 70/15/15 splits (Methods). Structure-based; no OOF residual correction.
- Refined overlap: framing partial, mechanism partial (combining experts), insight differ, domain partial -> 1 axis.

### [31] Miccio, Ensemble of experts (Comput. Mater. Sci. 258, 2025, doi:10.1016/j.commatsci.2025.114092)
- First fetch FAILED (Elsevier paywall; no OA copy; no abstract in any index; Firecrawl out of credits). The owner later supplied the PDF (`papers/Miccio2025_CMS_ensemble_of_experts.pdf`, git-ignored). Text extracted with pymupdf; abstract, introduction and Sec. 2.3 read (update 2026-10-18).
- Problem framing (verified): data-scarce prediction of polymer/molecular glass transition temperature (Tg), Tg of binary mixtures, and the Flory-Huggins chi of polymer-solvent systems.
- Core mechanism (verified): "an ensemble of experts (EE) approach ... using expert models previously trained on datasets of different, but physically meaningful properties" (abstract). Neural experts are pre-trained on related properties (e.g. logP) from tokenized SMILES, and their learned representations/outputs feed a model for the data-scarce target, compared against standard ANNs (Sec. 2.3). This is transfer learning; the text contains no stacking, OOF/cross-fitting, residual correction, kriging, nearest-neighbour correction or Matbench (keyword counts: stack 0, out-of-fold 0, residual 0, kriging 0, Matbench 0).
- Key insight (verified): knowledge from physically related property models compensates for scarce target data.
- Application domain (verified): polymers and molecular glass formers (SMILES); not inorganic compositions.
- Refined overlap: framing partial (data scarcity), mechanism differ, insight differ, domain differ -> 0 axes (Level 5); counted conservatively as at most 1 axis -> **Level 4 — Low Overlap**. Risk resolved. Cite it in related work (same journal; "ensemble of experts" under data scarcity), and do not use the phrase "ensemble of experts" for our method without distinguishing it.

### [17] Zhao et al., Interpretable stacking for HEA properties (Front. Mater. 2025)
- Fetched: frontiersin PDF. Mechanism: RF/GB/XGBoost base learners, an SVR meta-learner, k-fold OOF predictions for the meta-learner, and SHAP; a single HEA dataset with a random split. No local correction and no benchmark folds.
- Refined overlap: framing differ, mechanism partial (stacking), insight differ, domain partial (composition-based alloys) -> 1 axis.

### [R5] SIG-Net (owner)
- Not opened: it is outside this project folder, and the instructions say not to touch other projects. From CLAUDE.md: the same two-stage mechanism, confirmed in pre-registered replication (p = 0.016), in a different domain (JCIM, chemistry). Mechanism match and insight match; framing and domain differ -> 2 axes. It must be cited as the origin of the method.
