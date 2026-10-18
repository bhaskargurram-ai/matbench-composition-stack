# Scoop-check report

## 1. Verdict

**Level 3 — Medium Overlap**

## 2. Delta

**Delta.** Unlike MODNet and the stacked band-gap ensemble of Kauwe et al., which fit one global model or meta-learner and treat its errors as exchangeable across composition space under a single split, the proposed work adds a cross-fitted, novelty-damped local OOF residual correction in composition space on top of a simplex-constrained heterogeneous stack, and evaluates it on the official Matbench folds with nested CV only. The measurable benefit is a reduction in MAE (or a gain in ROC-AUC) relative to the same global stack, isolated by ablation.

Justification: no retrieved or recalled paper combines all four axes. The mechanism itself is the owner's own SIG-Net (match on mechanism and insight, different domain). Its closest external relative is Shmalo's "Neural Means and Kernel Corrections for Operator Learning" (a convex stack followed by a global kernel-ridge residual correction, without novelty damping, outside materials). On the materials side, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet" and "Extracting Knowledge from DFT: Experimental Band Gap Predictions Through Ensemble Learning" share the framing and/or domain but not the local correction. The contribution is therefore a transfer-and-test study, not a new learning principle. The write-up must:
1. cite SIG-Net as the origin;
2. cite stacking (Wolpert, Breiman, Super Learner) and regression kriging as the mechanism's ancestors;
3. claim novelty only for the novelty-damped local OOF correction in composition space evaluated honestly on Matbench.

Resolved risk (update 2026-10-18): "Machine learning-driven property prediction for materials in data-scarcity scenarios: Ensemble of experts approach" (Miccio, Comput. Mater. Sci. 2025) was first unreadable (paywall). The owner then supplied the PDF and it was read. It is transfer learning from neural experts pre-trained on related polymer properties (SMILES inputs), with no stacking, OOF residuals, local correction or Matbench, so it is Level 4 — Low Overlap. It does not change the verdict, but it should be cited in related work (same target journal).

## 3. Decomposed claim

- **Problem framing** - Regression (steels yield strength, experimental band gap) and binary classification (metal/non-metal, glass-forming ability) from chemical formula only; official Matbench 5-fold splits; MAE / ROC-AUC; all selection by nested CV inside each training fold.
- **Core mechanism** - (i) several diverse experts (boosted trees and RF on Magpie, kernel model on element fractions, MLP, ...) cross-fitted inside each training fold to give OOF predictions; (ii) a simplex-constrained (non-negative, sum-to-one) global stack fitted on those OOF predictions; (iii) a local correction that adds a kernel-weighted average of neighbouring training points' OOF stack residuals, multiplied by a damping factor that shrinks the correction toward zero when the query is far (novel) from the training compositions; strength and neighbourhood size chosen by inner CV.
- **Key insight** - A global convex stack leaves residual structure that is local in composition space (chemical families share systematic errors); OOF residuals estimate that structure without leakage, and damping by novelty avoids extrapolating corrections to unfamiliar chemistries.
- **Application domain** - Inorganic materials, composition-only, small/medium experimental datasets (312-5,680 rows); CPU-only.

## 4. Structured papers


---
**[1] Stacking Ensemble Learning with Genetic Algorithm Optimization for Multi-Property Prediction and Inverse Design of BCC-Type V-Based Hydrogen Storage Alloys**
- Title: Stacking Ensemble Learning with Genetic Algorithm Optimization for Multi-Property Prediction and Inverse Design of BCC-Type V-Based Hydrogen Storage Alloys
- Date: 2026-07
- Problem framing: Multi-property regression for V-based BCC hydrogen-storage alloys, random CV
- Core mechanism: Stacking ensemble + genetic-algorithm inverse design
- Key insight: Stacking beats single learners on a small alloy set
- Application domain: Hydrogen-storage alloys (composition)
- Overlap score: 1
- Source: doi:10.3390/coatings16070794

---
**[2] Physical encoding improves OOD performance in deep learning materials property prediction**
- Title: Physical encoding improves OOD performance in deep learning materials property prediction
- Date: 2024-07
- Problem framing: OOD materials property prediction
- Core mechanism: Physical (element-property) encodings for DL models
- Key insight: Physical encodings help extrapolation
- Application domain: Inorganic materials, composition/structure
- Overlap score: 1
- Source: doi:10.1016/j.commatsci.2024.113603

---
**[3] Interpretable ensemble learning for materials property prediction with classical interatomic potentials**
- Title: Interpretable ensemble learning for materials property prediction with classical interatomic potentials
- Date: 2023-07
- Problem framing: Property regression from classical-potential features
- Core mechanism: Ensemble learning (bagging/boosting) on interatomic-potential descriptors
- Key insight: Cheap physics features + ensembles are interpretable
- Application domain: Carbon crystals
- Overlap score: 0
- Source: doi:10.1038/s41524-024-01468-3

---
**[4] Pre-Activation based Representation Learning to Enhance Predictive Analytics on Small Materials Data**
- Title: Pre-Activation based Representation Learning to Enhance Predictive Analytics on Small Materials Data
- Date: 2023-06
- Problem framing: Small materials datasets
- Core mechanism: Pre-activation residual network representation learning
- Key insight: Better deep representations for small data
- Application domain: Inorganic materials (composition)
- Overlap score: 1
- Source: doi:10.1109/IJCNN54540.2023.10191086

---
**[5] In context learning foundation models for materials property prediction with small datasets**
- Title: In context learning foundation models for materials property prediction with small datasets
- Date: 2025-12
- Problem framing: Small-data materials property prediction
- Core mechanism: In-context-learning foundation model
- Key insight: ICL transfers across small tasks
- Application domain: Materials, incl. composition tasks
- Overlap score: 1
- Source: doi:10.1038/s41524-026-02089-8

---
**[6] Data Mining Methods for Biomechanical Property Prediction of Biomedical Materials Based on Optimized Feature Dimensionality Reduction**
- Title: Data Mining Methods for Biomechanical Property Prediction of Biomedical Materials Based on Optimized Feature Dimensionality Reduction
- Date: 2025-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Proceedings of the 2025 6th International Conference on Comp
- Overlap score: 0
- Source: doi:10.1145/3795154.3795181

---
**[7] Atomistic graph networks for experimental materials property prediction**
- Title: Atomistic graph networks for experimental materials property prediction
- Date: 2021-03
- Problem framing: Experimental materials property prediction
- Core mechanism: Atomistic graph networks with transfer
- Key insight: Transfer from DFT to experiment
- Application domain: Inorganic materials
- Overlap score: 1
- Source: arXiv:2103.13795

---
**[8] Interpretable Ensemble Learning for Materials Property Prediction with Classical Interatomic Potentials: Carbon as an Example**
- Title: Interpretable Ensemble Learning for Materials Property Prediction with Classical Interatomic Potentials: Carbon as an Example
- Date: 2023-07
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv (Cornell University)
- Overlap score: 0
- Source: doi:10.48550/arxiv.2308.10818

---
**[9] A strategy to apply machine learning to small datasets in materials science**
- Title: A strategy to apply machine learning to small datasets in materials science
- Date: 2018-05
- Problem framing: Small materials datasets
- Core mechanism: Crude estimation of property as feature + ML
- Key insight: Domain knowledge features help small data
- Application domain: Materials (composition)
- Overlap score: 1
- Source: doi:10.1038/s41524-018-0081-z

---
**[10] Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet**
- Title: Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet
- Date: 2021-06
- Problem framing: Matbench and small datasets
- Core mechanism: Feature selection (NMI) + joint-learning MLP (MODNet)
- Key insight: Relevant-feature selection reduces small-data overfitting
- Application domain: Matbench composition and structure tasks
- Overlap score: 2
- Source: doi:10.1038/s41524-021-00552-2

---
**[11] AtomSets as a hierarchical transfer learning framework for small and large materials datasets**
- Title: AtomSets as a hierarchical transfer learning framework for small and large materials datasets
- Date: 2021-10
- Problem framing: Small and large materials datasets
- Core mechanism: AtomSets transfer learning (pretrained MEGNet features + set model)
- Key insight: Pretrained features help small sets
- Application domain: Matbench tasks
- Overlap score: 2
- Source: doi:10.1038/s41524-021-00639-w

---
**[12] Materials Property Prediction with Uncertainty Quantification: A Benchmark Study**
- Title: Materials Property Prediction with Uncertainty Quantification: A Benchmark Study
- Date: 2022-11
- Problem framing: UQ for materials property prediction
- Core mechanism: Benchmark of UQ methods (ensembles, MC dropout, etc.)
- Key insight: UQ quality varies by method
- Application domain: Matbench-like datasets
- Overlap score: 1
- Source: doi:10.1063/5.0133528

---
**[13] Accurate, interpretable predictions of materials properties within transformer language models**
- Title: Accurate, interpretable predictions of materials properties within transformer language models
- Date: 2023-08
- Problem framing: Property prediction accuracy vs interpretability
- Core mechanism: Transformer language model on text descriptions
- Key insight: Text representations can be accurate and interpretable
- Application domain: Inorganic materials
- Overlap score: 0
- Source: doi:10.1016/j.patter.2023.100803

---
**[14] MPpredictor: An Artificial Intelligence-Driven Web Tool for Composition-Based Material Property Prediction**
- Title: MPpredictor: An Artificial Intelligence-Driven Web Tool for Composition-Based Material Property Prediction
- Date: 2023-03
- Problem framing: Composition-based property prediction web tool
- Core mechanism: Deep/ML models on composition
- Key insight: Tooling
- Application domain: Inorganic compositions
- Overlap score: 1
- Source: doi:10.1021/acs.jcim.3c00307

---
**[15] A Model Ensemble Approach Enables Data-Driven Property Prediction for Chemically Deconstructable Thermosets in the Low-Data Regime**
- Title: A Model Ensemble Approach Enables Data-Driven Property Prediction for Chemically Deconstructable Thermosets in the Low-Data Regime
- Date: 2023-09
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: ACS Central Science
- Overlap score: 0
- Source: doi:10.1021/acscentsci.3c00502

---
**[16] Enhancing material property prediction with ensemble deep graph convolutional networks**
- Title: Enhancing material property prediction with ensemble deep graph convolutional networks
- Date: 2024-07
- Problem framing: Materials property prediction
- Core mechanism: Ensembles of deep graph convolutional networks
- Key insight: Ensembling GNNs improves accuracy
- Application domain: Structure-based (Matbench/MP)
- Overlap score: 1
- Source: doi:10.48550/arXiv.2407.18847

---
**[17] An interpretable stacking ensemble model for high-entropy alloy mechanical property prediction**
- Title: An interpretable stacking ensemble model for high-entropy alloy mechanical property prediction
- Date: 2025-06
- Problem framing: HEA mechanical properties, random split
- Core mechanism: Two-level stacking (RF/GB/XGB base, SVR meta) + SHAP
- Key insight: Stacking outperforms single learners
- Application domain: High-entropy alloys (composition)
- Overlap score: 1
- Source: doi:10.3389/fmats.2025.1601874

---
**[18] Machine learning prediction of flexural strength of high‐entropy carbides via feature‐engineered ensemble modeling**
- Title: Machine learning prediction of flexural strength of high‐entropy carbides via feature‐engineered ensemble modeling
- Date: 2025-12
- Problem framing: HEC flexural strength
- Core mechanism: Feature-engineered ensemble models
- Key insight: Feature engineering + ensembles
- Application domain: High-entropy carbides
- Overlap score: 1
- Source: doi:10.1111/jace.70444

---
**[19] Physics-guided deep learning framework for optical spectroscopy-based materials property prediction**
- Title: Physics-guided deep learning framework for optical spectroscopy-based materials property prediction
- Date: 2026-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Conference on Advanced Materials and Equipment
- Overlap score: 0
- Source: doi:10.1117/12.3107551

---
**[20] Neural Means and Kernel Corrections for Operator Learning**
- Title: Neural Means and Kernel Corrections for Operator Learning
- Date: 2026-09
- Problem framing: Operator learning benchmarks
- Core mechanism: Six-member convex stack + Matern kernel-ridge regression of stack residuals
- Key insight: Residual alignment limits averaging; kernel residual correction gives small repeatable gains
- Application domain: PDE surrogates / radiative transfer
- Overlap score: 1
- Source: doi:10.5281/zenodo.23329779

---
**[21] MD-HIT: Machine learning for materials property prediction with dataset redundancy control**
- Title: MD-HIT: Machine learning for materials property prediction with dataset redundancy control
- Date: 2023-07
- Problem framing: Benchmark bias from redundant materials
- Core mechanism: Redundancy-controlled splits (MD-HIT)
- Key insight: Random splits overestimate accuracy
- Application domain: Materials property datasets (composition and structure)
- Overlap score: 1
- Source: arXiv:2307.04351

---
**[22] Machine Learning and Data Analytics for Design and Manufacturing of High-Entropy Materials Exhibiting Mechanical or Fatigue Properties of Interest**
- Title: Machine Learning and Data Analytics for Design and Manufacturing of High-Entropy Materials Exhibiting Mechanical or Fatigue Properties of Interest
- Date: 2020-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2012.07583

---
**[23] MACHINE LEARNING ACCELERATED PROPERTY PREDICTION AND SCREENING OF STABLE LITHIUM-BASED BATTERY MATERIALS**
- Title: MACHINE LEARNING ACCELERATED PROPERTY PREDICTION AND SCREENING OF STABLE LITHIUM-BASED BATTERY MATERIALS
- Date: 2026-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Journal of Science Natural Science
- Overlap score: 0
- Source: doi:10.18173/2354-1059.2026-0005

---
**[24] Metallurgically Interpretable Multimodal Machine Learning for Mechanical Property Prediction in 9% Cr Steels: Understanding Composition–Microstructure–Property Relationships**
- Title: Metallurgically Interpretable Multimodal Machine Learning for Mechanical Property Prediction in 9% Cr Steels: Understanding Composition–Microstructure–Property Relationships
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.21203/rs.3.rs-10863282/v1

---
**[25] Compositionally restricted attention-based network for materials property predictions**
- Title: Compositionally restricted attention-based network for materials property predictions
- Date: 2021-05
- Problem framing: Composition-based property prediction
- Core mechanism: CrabNet attention over element embeddings
- Key insight: Attention captures element context
- Application domain: Matbench composition tasks
- Overlap score: 2
- Source: doi:10.1038/s41524-021-00545-1

---
**[26] Machine learning-based accelerated property prediction of two-phase materials using microstructural descriptors and finite element analysis**
- Title: Machine learning-based accelerated property prediction of two-phase materials using microstructural descriptors and finite element analysis
- Date: 2021-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.1016/J.COMMATSCI.2021.110328

---
**[27] Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study**
- Title: Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study
- Date: 2024-01
- Problem framing: OOD materials property prediction benchmark
- Core mechanism: Benchmark of GNNs under OOD splits
- Key insight: OOD performance much worse than ID
- Application domain: Structure tasks
- Overlap score: 1
- Source: doi:10.1038/s41524-024-01316-4

---
**[28] Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework**
- Title: Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework
- Date: 2022-11
- Problem framing: Data-scarce materials properties
- Core mechanism: Mixture of experts over pretrained extractors with learned gating
- Key insight: Combine many source models adaptively
- Application domain: 19 materials regression tasks incl. experimental
- Overlap score: 2
- Source: doi:10.1038/s41524-022-00929-x

---
**[29] Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions**
- Title: Accelerating materials property prediction via a hybrid Transformer Graph framework that leverages four body interactions
- Date: 2025-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: npj Computational Materials
- Overlap score: 0
- Source: doi:10.1038/s41524-024-01472-7

---
**[30] Improved hardness prediction for reduced-activation high-entropy alloys by incorporating symbolic regression and domain adaptation on small datasets**
- Title: Improved hardness prediction for reduced-activation high-entropy alloys by incorporating symbolic regression and domain adaptation on small datasets
- Date: 2025-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Journal of Materials Informatics
- Overlap score: 0
- Source: doi:10.20517/jmi.2024.71

---
**[31] Machine learning-driven property prediction for materials in data-scarcity scenarios: Ensemble of experts approach**
- Title: Machine learning-driven property prediction for materials in data-scarcity scenarios: Ensemble of experts approach
- Date: 2025-08
- Problem framing: Data-scarce materials property prediction
- Core mechanism: 'Ensemble of experts' (abstract not available from any connector)
- Key insight: unknown (abstract unavailable)
- Application domain: Materials (Comput. Mater. Sci. 2025)
- Overlap score: 2
- Source: doi:10.1016/j.commatsci.2025.114092

---
**[32] Accelerating Elastic Property Prediction in Fe-C Alloys through Coupling of Molecular Dynamics and Machine Learning**
- Title: Accelerating Elastic Property Prediction in Fe-C Alloys through Coupling of Molecular Dynamics and Machine Learning
- Date: 2024-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials
- Overlap score: 0
- Source: doi:10.3390/ma17030601

---
**[33] Ensemble Machine Learning Model for Software Defect Prediction**
- Title: Ensemble Machine Learning Model for Software Defect Prediction
- Date: 2021-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Advances in Machine Learning &amp; Artificial Intelligence
- Overlap score: 0
- Source: doi:10.33140/amlai.02.01.03

---
**[34] Compositionally-Restricted Attention-Based Network for Materials Property Prediction**
- Title: Compositionally-Restricted Attention-Based Network for Materials Property Prediction
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.26434/chemrxiv.11869026.v2

---
**[35] Random Forest Ensemble Model for Reservoir Fluid Property Prediction**
- Title: Random Forest Ensemble Model for Reservoir Fluid Property Prediction
- Date: 2022-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: SPE Nigeria Annual International Conference and Exhibition
- Overlap score: 0
- Source: doi:10.2118/212044-ms

---
**[36] What Information is Necessary and Sufficient to Predict Materials Properties using Machine Learning?**
- Title: What Information is Necessary and Sufficient to Predict Materials Properties using Machine Learning?
- Date: 2022-06
- Problem framing: What information predicts properties
- Core mechanism: Composition-only vs structure models
- Key insight: Composition alone often suffices
- Application domain: Inorganic materials
- Overlap score: 1
- Source: doi:10.48550/arxiv.2206.04968

---
**[37] Accelerating the design of resorbable magnesium alloys: a machine learning approach to property prediction**
- Title: Accelerating the design of resorbable magnesium alloys: a machine learning approach to property prediction
- Date: 2026-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials &amp; Design
- Overlap score: 0
- Source: doi:10.1016/j.matdes.2026.116060

---
**[38] Dynamic embedding representation for graph neural networks to enhance materials property prediction with limited datasets**
- Title: Dynamic embedding representation for graph neural networks to enhance materials property prediction with limited datasets
- Date: 2026-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials Research Express
- Overlap score: 0
- Source: doi:10.1088/2053-1591/ae5148

---
**[39] MatUQ: a benchmark for uncertainty-aware out-of-distribution materials property prediction with graph neural networks**
- Title: MatUQ: a benchmark for uncertainty-aware out-of-distribution materials property prediction with graph neural networks
- Date: 2026-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: npj Computational Materials
- Overlap score: 0
- Source: doi:10.1038/s41524-026-02272-x

---
**[40] Multi-Source Domain Transfer Learning for Accurate Property Prediction in Two-Dimensional Materials**
- Title: Multi-Source Domain Transfer Learning for Accurate Property Prediction in Two-Dimensional Materials
- Date: 2026-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: arXiv:2605.24455

---
**[41] Mathematical Invariant-Enabled Topological Neural Networks for Molecular and Materials Property Prediction**
- Title: Mathematical Invariant-Enabled Topological Neural Networks for Molecular and Materials Property Prediction
- Date: 2026-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: arXiv:2610.07712

---
**[42] Attention-based composition learning for thermodynamic and electronic property prediction in inorganic materials**
- Title: Attention-based composition learning for thermodynamic and electronic property prediction in inorganic materials
- Date: 2026-08
- Problem framing: Formation energy and band gap
- Core mechanism: Attention-based composition learning
- Key insight: Attention over composition
- Application domain: Inorganic materials
- Overlap score: 1
- Source: doi:10.1016/j.nxmate.2026.103265

---
**[43] DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction**
- Title: DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction
- Date: 2026-08
- Problem framing: Low-data property prediction
- Core mechanism: Distillation + self-supervised pretraining, structure-agnostic
- Key insight: Pretraining helps low data
- Application domain: Inorganic composition tasks
- Overlap score: 1
- Source: doi:10.48550/arxiv.2609.00059

---
**[44] Large language model-enabled automated data extraction for concrete materials informatics**
- Title: Large language model-enabled automated data extraction for concrete materials informatics
- Date: 2026-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1038/s41524-026-02304-6

---
**[45] Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints**
- Title: Comparative Evaluation of Machine Learning Approaches for Minority-Class Financial Distress Prediction Under Class Imbalance Constraints
- Date: 2026-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2605.14067

---
**[46] Improving Novelty Detection using the Reconstructions of Nearest Neighbours**
- Title: Improving Novelty Detection using the Reconstructions of Nearest Neighbours
- Date: 2021-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1016/j.array.2022.100182

---
**[47] Deep Residual Networks with Synthetic Minority Over-Sampling Technique Edited Nearest Neighbours (SMOTE-ENN) for ECG Monitoring for Arrhythmia Classification**
- Title: Deep Residual Networks with Synthetic Minority Over-Sampling Technique Edited Nearest Neighbours (SMOTE-ENN) for ECG Monitoring for Arrhythmia Classification
- Date: 2026-07
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Pertanika Journal of Science and Technology
- Overlap score: 0
- Source: doi:10.47836/pjst.34.s1.01

---
**[48] A polymer dataset for accelerated property prediction and design**
- Title: A polymer dataset for accelerated property prediction and design
- Date: 2016-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Scientific Data
- Overlap score: 0
- Source: doi:10.1038/sdata.2016.12

---
**[49] MaterialsAtlas.org: a materials informatics web app platform for materials discovery and survey of state-of-the-art**
- Title: MaterialsAtlas.org: a materials informatics web app platform for materials discovery and survey of state-of-the-art
- Date: 2022-04
- Problem framing: Materials informatics web platform
- Core mechanism: Survey of property models
- Key insight: Tooling
- Application domain: Materials
- Overlap score: 0
- Source: doi:10.1038/s41524-022-00750-6

---
**[50] LLM4Mat-bench: benchmarking large language models for materials property prediction**
- Title: LLM4Mat-bench: benchmarking large language models for materials property prediction
- Date: 2024-10
- Problem framing: LLM benchmark for property prediction
- Core mechanism: LLM fine-tuning/prompting
- Key insight: LLMs lag specialised models
- Application domain: Materials incl. composition
- Overlap score: 1
- Source: doi:10.1088/2632-2153/add3bb

---
**[51] Realistic material property prediction using domain adaptation based machine learning**
- Title: Realistic material property prediction using domain adaptation based machine learning
- Date: 2023-12
- Problem framing: Realistic OOD property prediction
- Core mechanism: Domain adaptation
- Key insight: Target-domain information improves OOD
- Application domain: Inorganic materials
- Overlap score: 1
- Source: doi:10.1039/d3dd00162h

---
**[52] Revolutionizing inverse design of ionic liquids through the multi-property prediction of over 300,000 novel variants using ensemble deep learning**
- Title: Revolutionizing inverse design of ionic liquids through the multi-property prediction of over 300,000 novel variants using ensemble deep learning
- Date: 2024-06
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials science & engineering. R, Reports
- Overlap score: 0
- Source: doi:10.1016/j.mser.2024.100798

---
**[53] Deep Learning-Based Prediction of Material Properties Using Chemical Compositions and Diffraction Patterns as Experimentally Accessible Inputs**
- Title: Deep Learning-Based Prediction of Material Properties Using Chemical Compositions and Diffraction Patterns as Experimentally Accessible Inputs
- Date: 2021-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: The Journal of Physical Chemistry Letters
- Overlap score: 0
- Source: doi:10.1021/acs.jpclett.1c02305

---
**[54] Chemical Composition Data‐Driven Machine‐Learning Prediction for Phase Stability and Materials Properties of Inorganic Crystalline Solids**
- Title: Chemical Composition Data‐Driven Machine‐Learning Prediction for Phase Stability and Materials Properties of Inorganic Crystalline Solids
- Date: 2022-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: physica status solidi (b)
- Overlap score: 0
- Source: doi:10.1002/pssb.202100525

---
**[55] Composition Design and Property Prediction for AlCoCrCuFeNi High-Entropy Alloy Based on Machine Learning**
- Title: Composition Design and Property Prediction for AlCoCrCuFeNi High-Entropy Alloy Based on Machine Learning
- Date: 2025-06
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Metals
- Overlap score: 0
- Source: doi:10.3390/met15070733

---
**[56] Research on Property Prediction of Materials Based on Machine Learning**
- Title: Research on Property Prediction of Materials Based on Machine Learning
- Date: 2020-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Conference on Advanced Electronic Materials, C
- Overlap score: 0
- Source: doi:10.1109/aemcse50948.2020.00017

---
**[57] Outlier-Based Domain of Applicability Identification for Materials Property Prediction Models**
- Title: Outlier-Based Domain of Applicability Identification for Materials Property Prediction Models
- Date: 
- Problem framing: Domain of applicability
- Core mechanism: Outlier-based DoA detection
- Key insight: Novelty/distance flags unreliable predictions
- Application domain: Materials property models
- Overlap score: 1
- Source: doi:10.26434/chemrxiv-2023-pmrfw-v2

---
**[58] Enhancing composition-based materials property prediction by cross-modal knowledge transfer**
- Title: Enhancing composition-based materials property prediction by cross-modal knowledge transfer
- Date: 2025-11
- Problem framing: Composition-based prediction
- Core mechanism: Cross-modal knowledge transfer from crystal GNNs to composition models
- Key insight: Structure knowledge improves composition models
- Application domain: Matbench composition tasks
- Overlap score: 2
- Source: doi:10.1038/s41598-026-53182-3

---
**[59] Measuring the impact of materials composition on MXene property prediction using explainable AI**
- Title: Measuring the impact of materials composition on MXene property prediction using explainable AI
- Date: 2025-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials Today Energy
- Overlap score: 0
- Source: doi:10.1016/j.mtener.2025.102136

---
**[60] Orbital Graph Convolutional Neural Network for Material Property Prediction**
- Title: Orbital Graph Convolutional Neural Network for Material Property Prediction
- Date: 2020-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1103/PhysRevMaterials.4.093801

---
**[61] Inverse Design of Composite Metal Oxide Optical Materials based on Deep Transfer Learning**
- Title: Inverse Design of Composite Metal Oxide Optical Materials based on Deep Transfer Learning
- Date: 2020-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1016/j.commatsci.2020.110166

---
**[62] MT-CGCNN: Integrating Crystal Graph Convolutional Neural Network with Multitask Learning for Material Property Prediction**
- Title: MT-CGCNN: Integrating Crystal Graph Convolutional Neural Network with Multitask Learning for Material Property Prediction
- Date: 2018-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:1811.05660

---
**[63] Applications of Machine Learning in Polymer Materials: Property Prediction, Material Design, and Systematic Processes**
- Title: Applications of Machine Learning in Polymer Materials: Property Prediction, Material Design, and Systematic Processes
- Date: 2025-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2510.26100

---
**[64] Deep Metric Learning and Image Classification with Nearest Neighbour Gaussian Kernels**
- Title: Deep Metric Learning and Image Classification with Nearest Neighbour Gaussian Kernels
- Date: 2017-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:1705.09780

---
**[65] A unified weighting framework for evaluating nearest neighbour classification**
- Title: A unified weighting framework for evaluating nearest neighbour classification
- Date: 2023-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2311.16872

---
**[66] Evaluation of the impact of the indiscernibility relation on the fuzzy-rough nearest neighbours algorithm**
- Title: Evaluation of the impact of the indiscernibility relation on the fuzzy-rough nearest neighbours algorithm
- Date: 2022-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2211.14134

---
**[67] Accelerating materials property prediction via a hybrid structure-composition transformer-graph framework that leverages four body interactions**
- Title: Accelerating materials property prediction via a hybrid structure-composition transformer-graph framework that leverages four body interactions
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.26226/m.64c26778632e9539aa87d934

---
**[68] Interpretable Ensemble Learning for Anticancer Peptide Prediction via Physicochemical Property Integration**
- Title: Interpretable Ensemble Learning for Anticancer Peptide Prediction via Physicochemical Property Integration
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.2139/ssrn.6088529

---
**[69] Machine Learning-Driven Materials Design and Performance Prediction in Organic Solar Cells Emphasizing Ensemble Learning Models**
- Title: Machine Learning-Driven Materials Design and Performance Prediction in Organic Solar Cells Emphasizing Ensemble Learning Models
- Date: 2026
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Computers, Materials &amp; Continua
- Overlap score: 0
- Source: doi:10.32604/cmc.2026.080623

---
**[70] Concrete strength prediction: Exploring an ensemble machine learning approach**
- Title: Concrete strength prediction: Exploring an ensemble machine learning approach
- Date: 2026
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Materials Research Proceedings Emerging Research in Material
- Overlap score: 0
- Source: doi:10.21741/9781644903933-2

---
**[71] Machine learning approaches for the prediction of materials properties**
- Title: Machine learning approaches for the prediction of materials properties
- Date: 2020-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: APL Materials
- Overlap score: 0
- Source: doi:10.1063/5.0018384

---
**[72] High-performance concrete strength prediction based on ensemble learning**
- Title: High-performance concrete strength prediction based on ensemble learning
- Date: 2022-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Construction and Building Materials
- Overlap score: 0
- Source: doi:10.1016/j.conbuildmat.2022.126694

---
**[73] AFLOW-ML: A RESTful API for machine-learning predictions of materials properties**
- Title: AFLOW-ML: A RESTful API for machine-learning predictions of materials properties
- Date: 2018-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Computational Materials Science
- Overlap score: 0
- Source: doi:10.1016/j.commatsci.2018.03.075

---
**[74] Extracting Knowledge from DFT: Experimental Band Gap Predictions Through Ensemble Learning**
- Title: Extracting Knowledge from DFT: Experimental Band Gap Predictions Through Ensemble Learning
- Date: 2020-07
- Problem framing: Experimental band gaps from composition, random split
- Core mechanism: Stacked ensemble: DFT-trained models' predictions + experimental models -> meta-learner
- Key insight: DFT trends transfer to experimental gaps
- Application domain: Experimental band gap (same source data family as matbench_expt_gap)
- Overlap score: 2
- Source: doi:10.1007/s40192-020-00178-0

---
**[75] Multi‐class classification using kernel density estimation on
                    <i>K</i>
                    ‐nearest neighbours**
- Title: Multi‐class classification using kernel density estimation on
                    <i>K</i>
                    ‐nearest neighbours
- Date: 2016-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Electronics Letters
- Overlap score: 0
- Source: doi:10.1049/el.2015.4437

---
**[76] Learning from mistakes: Sampling strategies to efficiently train machine learning models for material property prediction**
- Title: Learning from mistakes: Sampling strategies to efficiently train machine learning models for material property prediction
- Date: 2023-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Computational Materials Science
- Overlap score: 0
- Source: doi:10.1016/j.commatsci.2023.112167

---
**[77] Prediction of Euroleague Games based on Supervised Classification Algorithm k-Nearest Neighbours**
- Title: Prediction of Euroleague Games based on Supervised Classification Algorithm k-Nearest Neighbours
- Date: 2018
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Proceedings of the 6th International Congress on Sport Scien
- Overlap score: 0
- Source: doi:10.5220/0006893502030207

---
**[78] Integration of the <i>k</i>-nearest neighbours and patch-based features for PolSAR image classification by using a two-branch residual network**
- Title: Integration of the <i>k</i>-nearest neighbours and patch-based features for PolSAR image classification by using a two-branch residual network
- Date: 2021-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Remote Sensing Letters
- Overlap score: 0
- Source: doi:10.1080/2150704x.2021.1966121

---
**[79] A Critical Review of Tribological Property Prediction of Thermally Sprayed Coatings Using Machine Learning Approaches**
- Title: A Critical Review of Tribological Property Prediction of Thermally Sprayed Coatings Using Machine Learning Approaches
- Date: 2025-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Journal of Bio- and Tribo-Corrosion
- Overlap score: 0
- Source: doi:10.1007/s40735-025-01082-w

---
**[80] A quantum residual attention neural network for high-precision material property prediction**
- Title: A quantum residual attention neural network for high-precision material property prediction
- Date: 2025-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Quantum Information Processing
- Overlap score: 0
- Source: doi:10.1007/s11128-025-04670-4

---
**[81] Machine learning modeling of melt-spinning for yarn property prediction**
- Title: Machine learning modeling of melt-spinning for yarn property prediction
- Date: 2025-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Results in Materials
- Overlap score: 0
- Source: doi:10.1016/j.rinma.2025.100776

---
**[82] Accelerating ensemble uncertainty estimates in supervised materials property regression models**
- Title: Accelerating ensemble uncertainty estimates in supervised materials property regression models
- Date: 2024-11
- Problem framing: Ensemble UQ in materials regression
- Core mechanism: Faster ensemble uncertainty estimates
- Key insight: Cheaper ensembles
- Application domain: Materials regression
- Overlap score: 1
- Source: doi:10.1016/j.commatsci.2024.113494

---
**[83] TemporalVAE for Temporal Prediction on Small Datasets v1**
- Title: TemporalVAE for Temporal Prediction on Small Datasets v1
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.17504/protocols.io.eq2ly47yplx9/v1

---
**[84] Solar Irradiance Prediction in Bangladesh: An Ensemble Machine-Learning FrameworkSolar Irradiance Prediction in Bangladesh: An Ensemble Machine-Learning Framework**
- Title: Solar Irradiance Prediction in Bangladesh: An Ensemble Machine-Learning FrameworkSolar Irradiance Prediction in Bangladesh: An Ensemble Machine-Learning Framework
- Date: 2025-09
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Journal of Research Publication and Reviews
- Overlap score: 0
- Source: doi:10.55248/gengpi.6.0925.3412

---
**[85] Machine Learning Based Prediction and Classification of Properties of Engineered Composite Materials**
- Title: Machine Learning Based Prediction and Classification of Properties of Engineered Composite Materials
- Date: 2020-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: https://openalex.org/W3136564151

---
**[86] ROBUST ESTIMATION METHODS FOR k-NEAREST NEIGHBOURS ENSEMBLE REGRESSION MODEL**
- Title: ROBUST ESTIMATION METHODS FOR k-NEAREST NEIGHBOURS ENSEMBLE REGRESSION MODEL
- Date: 2025-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Zenodo (CERN European Organization for Nuclear Research)
- Overlap score: 0
- Source: doi:10.5281/zenodo.18052841

---
**[87] Efficient Estimation of Material Property Curves and Surfaces via Active Learning**
- Title: Efficient Estimation of Material Property Curves and Surfaces via Active Learning
- Date: 2020-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1103/PhysRevMaterials.5.013802

---
**[88] Leveraging Composition-Based Material Descriptors for Machine Learning Optimization**
- Title: Leveraging Composition-Based Material Descriptors for Machine Learning Optimization
- Date: 2023-04
- Problem framing: Classifier optimisation with composition descriptors
- Core mechanism: Feature selection for composition-based classifiers
- Key insight: Minimal descriptor sets
- Application domain: Inorganic materials
- Overlap score: 1
- Source: doi:10.1016/j.mtcomm.2023.106579

---
**[89] Sub-seasonal forecasting with a large ensemble of deep-learning weather prediction models**
- Title: Sub-seasonal forecasting with a large ensemble of deep-learning weather prediction models
- Date: 2021-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1029/2021MS002502

---
**[90] Meta-Learning Probabilistic Inference For Prediction**
- Title: Meta-Learning Probabilistic Inference For Prediction
- Date: 2018-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:1805.09921

---
**[91] From Non-Paying to Premium: Predicting User Conversion in Video Games with Ensemble Learning**
- Title: From Non-Paying to Premium: Predicting User Conversion in Video Games with Ensemble Learning
- Date: 2019-06
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1145/3337722.3341855

---
**[92] RNA Secondary Structure Prediction By Learning Unrolled Algorithms**
- Title: RNA Secondary Structure Prediction By Learning Unrolled Algorithms
- Date: 2020-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2002.05810

---
**[93] k-Nearest Neighbour Classifiers: 2nd Edition (with Python examples)**
- Title: k-Nearest Neighbour Classifiers: 2nd Edition (with Python examples)
- Date: 2020-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1145/3459665

---
**[94] Bags of Projected Nearest Neighbours: Competitors to Random Forests?**
- Title: Bags of Projected Nearest Neighbours: Competitors to Random Forests?
- Date: 2025-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2503.09651

---
**[95] Using Enriched Category Theory to Construct the Nearest Neighbour Classification Algorithm**
- Title: Using Enriched Category Theory to Construct the Nearest Neighbour Classification Algorithm
- Date: 2023-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2312.16529

---
**[96] Minimax rates for cost-sensitive learning on manifolds with approximate nearest neighbours**
- Title: Minimax rates for cost-sensitive learning on manifolds with approximate nearest neighbours
- Date: 2018-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:1803.00310

---
**[97] Prediction of Heart Disease Using Ensemble Machine Learning Techniques**
- Title: Prediction of Heart Disease Using Ensemble Machine Learning Techniques
- Date: 2026-09
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Journal of Intelligent Data and Machine Learni
- Overlap score: 0
- Source: doi:10.55640/ijidml-v03i09-01

---
**[98] Econometric-Regularized Ensemble Learning Framework for Multi-Factor Gold Price Prediction**
- Title: Econometric-Regularized Ensemble Learning Framework for Multi-Factor Gold Price Prediction
- Date: 2025
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Proceedings of the 3rd International Conference on Data Anal
- Overlap score: 0
- Source: doi:10.5220/0014751200004818

---
**[99] Temporal Ensembling-based Deep k-Nearest Neighbours for Learning with Noisy Labels**
- Title: Temporal Ensembling-based Deep k-Nearest Neighbours for Learning with Noisy Labels
- Date: 2023
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: ESANN 2023 proceesdings
- Overlap score: 0
- Source: doi:10.14428/esann/2023.es2023-144

---
**[100] Better utilization of materials’ compositions for predicting their properties: Material composition visualization network**
- Title: Better utilization of materials’ compositions for predicting their properties: Material composition visualization network
- Date: 2022-11
- Problem framing: Composition-based property prediction
- Core mechanism: Composition visualisation network
- Key insight: Image-like composition encoding
- Application domain: Inorganic materials
- Overlap score: 1
- Source: doi:10.1016/j.engappai.2022.105539

---
**[101] KernelKnn: Kernel k Nearest Neighbors**
- Title: KernelKnn: Kernel k Nearest Neighbors
- Date: 2016-07
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.32614/cran.package.kernelknn

---
**[102] Comparison of music genre classification using Nearest Centroid Classifier and k-Nearest Neighbours**
- Title: Comparison of music genre classification using Nearest Centroid Classifier and k-Nearest Neighbours
- Date: 2016-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: 2016 International Conference on Information Management and 
- Overlap score: 0
- Source: doi:10.1109/icimtech.2016.7930314

---
**[103] k-nächste Nachbarn (k nearest neighbours)**
- Title: k-nächste Nachbarn (k nearest neighbours)
- Date: 2020-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Springer eBooks
- Overlap score: 0
- Source: doi:10.1007/978-3-658-30153-8_8

---
**[104] tpospisi/NNKCDE 0.3**
- Title: tpospisi/NNKCDE 0.3
- Date: 2019-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Zenodo (CERN European Organization for Nuclear Research)
- Overlap score: 0
- Source: doi:10.5281/zenodo.3364857

---
**[105] NNKCDE: Nearest Neighbor Kernel Conditional Density Estimation**
- Title: NNKCDE: Nearest Neighbor Kernel Conditional Density Estimation
- Date: 2020-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Astrophysics Source Code Library
- Overlap score: 0
- Source: https://openalex.org/W3164413928

---
**[106] K-Nearest Neighbours based diagnosis of hyperglycemia**
- Title: K-Nearest Neighbours based diagnosis of hyperglycemia
- Date: 2017-12
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Journal of Trend in Scientific Research and De
- Overlap score: 0
- Source: doi:10.31142/ijtsrd7046

---
**[107] Improving Alloy Mechanical Property Prediction by Addressing Data Homogeneity through Noise Augmentation**
- Title: Improving Alloy Mechanical Property Prediction by Addressing Data Homogeneity through Noise Augmentation
- Date: 2026-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Journal of materials engineering and performance (Print)
- Overlap score: 0
- Source: doi:10.1007/s11665-026-14020-9

---
**[108] KERNEL AND NEAREST-NEIGHBOR ESTIMATION OF A**
- Title: KERNEL AND NEAREST-NEIGHBOR ESTIMATION OF A
- Date: 2016-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: https://openalex.org/W2438681483

---
**[109] Fast Computation of Leave-One-Out Cross-Validation for $k$-NN Regression**
- Title: Fast Computation of Leave-One-Out Cross-Validation for $k$-NN Regression
- Date: 2024-05
- Problem framing: k-NN regression model selection
- Core mechanism: Fast LOOCV for k-NN regression
- Key insight: Closed-form LOOCV
- Application domain: General ML
- Overlap score: 0
- Source: doi:10.48550/arxiv.2405.04919

---
**[110] Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery**
- Title: Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery
- Date: 2025-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1002/mgea.70059

---
**[111] The ECMWF Ensemble Prediction System: Looking Back (more than) 25 Years and Projecting Forward 25 Years**
- Title: The ECMWF Ensemble Prediction System: Looking Back (more than) 25 Years and Projecting Forward 25 Years
- Date: 2018-03
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1002/qj.3383

---
**[112] A Quadtree, a Steiner Spanner, and Approximate Nearest Neighbours in Hyperbolic Space**
- Title: A Quadtree, a Steiner Spanner, and Approximate Nearest Neighbours in Hyperbolic Space
- Date: 2023-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.20382/jocg.v16i2a5

---
**[113] A Soft Range Limited K-Nearest Neighbours Algorithm for Indoor Localization Enhancement**
- Title: A Soft Range Limited K-Nearest Neighbours Algorithm for Indoor Localization Enhancement
- Date: 2019-08
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1109/JSEN.2018.2874453

---
**[114] Nearest-Neighbour-Induced Isolation Similarity and its Impact on Density-Based Clustering**
- Title: Nearest-Neighbour-Induced Isolation Similarity and its Impact on Density-Based Clustering
- Date: 2019-06
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1609/aaai.v33i01.33014755

---
**[115] Improving the Effectiveness and Efficiency of Stochastic Neighbour Embedding with Isolation Kernel**
- Title: Improving the Effectiveness and Efficiency of Stochastic Neighbour Embedding with Isolation Kernel
- Date: 2019-06
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1613/jair.1.12904

---
**[116] Cross-shell excitations in doubly magic 132Sn and its nearest neighbours**
- Title: Cross-shell excitations in doubly magic 132Sn and its nearest neighbours
- Date: 2020-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1016/j.nuclphysa.2021.122262

---
**[117] Interactions beyond nearest neighbours and rigidity of discrete energies: a compactness result and an application to dimension reduction**
- Title: Interactions beyond nearest neighbours and rigidity of discrete energies: a compactness result and an application to dimension reduction
- Date: 2016-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:1601.05968

---
**[118] Beyond Pick-and-Place: Tackling Robotic Stacking of Diverse Shapes**
- Title: Beyond Pick-and-Place: Tackling Robotic Stacking of Diverse Shapes
- Date: 2021-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2110.06192

---
**[119] GeoVersa: Design-Based Residual-Correction Forests for Digital Soil Mapping**
- Title: GeoVersa: Design-Based Residual-Correction Forests for Digital Soil Mapping
- Date: 2026-07
- Problem framing: Digital soil mapping
- Core mechanism: Design-based residual-correction forests
- Key insight: Correct ML residuals spatially
- Application domain: Soil science
- Overlap score: 1
- Source: doi:10.32614/cran.package.geoversa

---
**[120] Nearest Neighbours and XAI Based Approach for Soft Labelling**
- Title: Nearest Neighbours and XAI Based Approach for Soft Labelling
- Date: 2023
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Proceedings of the 15th International Conference on Agents a
- Overlap score: 0
- Source: doi:10.5220/0011619800003393

---
**[121] Application of K-Nearest Neighbours in Regression Modelling for Solving Transportation Problems**
- Title: Application of K-Nearest Neighbours in Regression Modelling for Solving Transportation Problems
- Date: 2023-07
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: International Journal of Science and Research (IJSR)
- Overlap score: 0
- Source: doi:10.21275/mr23713200616

---
**[122] BlockingPy: approximate nearest neighbours for blocking of records for entity resolution**
- Title: BlockingPy: approximate nearest neighbours for blocking of records for entity resolution
- Date: 
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.2139/ssrn.5644279

---
**[123] Nearest Neighbours**
- Title: Nearest Neighbours
- Date: 2019-11
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Machine Learning and Big Data with kdb+/q
- Overlap score: 0
- Source: doi:10.1002/9781119404729.ch21

---
**[124] Water Quality Classification using an Out-of-Fold CatBoost-LSTM Stacking Framework**
- Title: Water Quality Classification using an Out-of-Fold CatBoost-LSTM Stacking Framework
- Date: 2026-03
- Problem framing: Water quality classification
- Core mechanism: Out-of-fold CatBoost-LSTM stacking
- Key insight: OOF stacking avoids leakage
- Application domain: Environmental data
- Overlap score: 1
- Source: doi:10.1109/icfact66887.2026.11518014

---
**[125] Neural Nearest Neighbors Networks**
- Title: Neural Nearest Neighbors Networks
- Date: 2018-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv (Cornell University)
- Overlap score: 0
- Source: doi:10.48550/arxiv.1810.12575

---
**[126] Optimal choice of $k$ for $k$-nearest neighbor regression**
- Title: Optimal choice of $k$ for $k$-nearest neighbor regression
- Date: 2019-09
- Problem framing: k-NN regression theory
- Core mechanism: Choice of k by LOOCV
- Key insight: Optimal k rates
- Application domain: General ML
- Overlap score: 0
- Source: doi:10.48550/arxiv.1909.05495

---
**[127] Percolation in Spatial Networks**
- Title: Percolation in Spatial Networks
- Date: 2022-07
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: doi:10.1017/9781009168076

---
**[128] Near-Orthogonality Regularization in Kernel Methods.**
- Title: Near-Orthogonality Regularization in Kernel Methods.
- Date: 2017-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: Uncertainty in Artificial Intelligence
- Overlap score: 0
- Source: https://openalex.org/W2771200462

---
**[129] Developing Novel Descriptors to Predict Physical Properties of Inorganic Compounds from Compositional Formula**
- Title: Developing Novel Descriptors to Predict Physical Properties of Inorganic Compounds from Compositional Formula
- Date: 2018-01
- Problem framing: Composition-based property descriptors
- Core mechanism: New compositional descriptors
- Key insight: Descriptor design
- Application domain: Inorganic compounds
- Overlap score: 1
- Source: doi:10.2751/jcac.19.7

---
**[130] K-NEAREST NEIGHBORS SPARSE OUTLIER REMOVAL ALGORITHM BASED ON FREQUENCY HISTOGRAM**
- Title: K-NEAREST NEIGHBORS SPARSE OUTLIER REMOVAL ALGORITHM BASED ON FREQUENCY HISTOGRAM
- Date: 2016-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: other field
- Overlap score: 0
- Source: https://openalex.org/W2745694635

---
**[131] Error minimizing algorithms for nearest eighbor classifiers**
- Title: Error minimizing algorithms for nearest eighbor classifiers
- Date: 2024-02
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: OSTI OAI (U.S. Department of Energy Office of Scientific and
- Overlap score: 0
- Source: https://openalex.org/W4298041117

---
**[132] Resilience of the physicochemical properties of graphene-based materials for applications in harsh radiation environments**
- Title: Resilience of the physicochemical properties of graphene-based materials for applications in harsh radiation environments
- Date: 2026-05
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: doi:10.1016/j.diamond.2026.113821

---
**[133] Generating Datasets of 3D Garments with Sewing Patterns**
- Title: Generating Datasets of 3D Garments with Sewing Patterns
- Date: 2021-09
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2109.05633

---
**[134] Probing magnetic ordering in air stable iron-rich van der Waals minerals**
- Title: Probing magnetic ordering in air stable iron-rich van der Waals minerals
- Date: 2023-04
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2304.06533

---
**[135] Big Data Science in Nanoporous Materials**
- Title: Big Data Science in Nanoporous Materials
- Date: 2023-01
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: AI‐Guided Design and Property Prediction for Zeolites and Na
- Overlap score: 0
- Source: doi:10.1002/9781119819783.ch12

---
**[136] Few-shot Molecular Property Prediction: A Survey**
- Title: Few-shot Molecular Property Prediction: A Survey
- Date: 2025-10
- Problem framing: off-topic for this claim (see title)
- Core mechanism: off-topic for this claim (see title)
- Key insight: off-topic for this claim (see title)
- Application domain: arXiv
- Overlap score: 0
- Source: arXiv:2510.08900

---
**[137] Ensembles of Graph and Physics-Informed Machine Learning for Scientific Modeling in Materials Science: A Review**
- Title: Ensembles of Graph and Physics-Informed Machine Learning for Scientific Modeling in Materials Science: A Review
- Date: 2025-08
- Problem framing: Review: graph + physics ensembles
- Core mechanism: Review
- Key insight: Review
- Application domain: Materials science
- Overlap score: 1
- Source: doi:10.1007/s11831-025-10325-5

---
**[R1] About regression-kriging: From equations to case studies (Hengl, Heuvelink, Rossiter)**
- Title: About regression-kriging: From equations to case studies (Hengl, Heuvelink, Rossiter)
- Date: 2007
- Problem framing: Regression kriging for spatial prediction
- Core mechanism: Trend model + kriging of its residuals in geographic space
- Key insight: Residuals are spatially autocorrelated, so kriging them adds information
- Application domain: Geostatistics / soil mapping
- Overlap score: 1
- Source: model-recall; DOI 10.1016/j.cageo.2007.05.001 (verify) (Source: model-recall)

---
**[R2] Stacked generalization (Wolpert)**
- Title: Stacked generalization (Wolpert)
- Date: 1992
- Problem framing: General supervised learning
- Core mechanism: Level-1 learner on cross-validated level-0 predictions
- Key insight: CV predictions estimate generalisation biases
- Application domain: General ML
- Overlap score: 1
- Source: model-recall; Neural Networks 5(2) (verify) (Source: model-recall)

---
**[R3] Stacked regressions (Breiman)**
- Title: Stacked regressions (Breiman)
- Date: 1996
- Problem framing: Regression
- Core mechanism: Non-negative least-squares combination of CV predictions
- Key insight: Non-negativity constraints stabilise stacking
- Application domain: General ML
- Overlap score: 1
- Source: model-recall; Machine Learning 24 (verify) (Source: model-recall)

---
**[R4] Super Learner (van der Laan, Polley, Hubbard)**
- Title: Super Learner (van der Laan, Polley, Hubbard)
- Date: 2007
- Problem framing: Prediction with oracle guarantees
- Core mechanism: Cross-validated convex combination of a library of learners
- Key insight: CV-selected combination is asymptotically as good as the oracle
- Application domain: Biostatistics / general
- Overlap score: 1
- Source: model-recall; Stat. Appl. Genet. Mol. Biol. 6 (verify) (Source: model-recall)

---
**[R5] SIG-Net (owner's own first paper, submitted to JCIM 2026-10-18)**
- Title: SIG-Net (owner's own first paper, submitted to JCIM 2026-10-18)
- Date: 2026-10
- Problem framing: Molecular/chemical property prediction (per CLAUDE.md; the paper itself was not opened, other project folders are out of scope)
- Core mechanism: Cross-fitted stacking of diverse experts + novelty-damped local OOF residual correction
- Key insight: Local OOF residual structure that the global stack misses; damped where the query is novel
- Application domain: Chemistry (JCIM)
- Overlap score: 2
- Source: project context (CLAUDE.md); github.com/bhaskargurram-ai/SIG-Net (Source: model-recall)

## 5. Comparison result


- **Proposed work**
  - Title: —
  - Date: —
  - Source: —
  - Problem framing: composition-only regression and classification on the four Matbench v0.1 composition tasks, official 5-fold splits, nested CV for every choice
  - Core mechanism: cross-fitted heterogeneous experts -> simplex-constrained stack on OOF predictions -> novelty-damped kernel-weighted local OOF residual correction in composition space
  - Key insight: the global stack leaves local, family-specific residual structure that OOF residuals can estimate without leakage; damping by novelty prevents mis-correction of unfamiliar chemistries
  - Application domain: inorganic materials, small/medium experimental datasets, CPU
- **Prior work A** (SIG-Net, owner)
  - Title: SIG-Net
  - Date: 2026-10
  - Source: github.com/bhaskargurram-ai/SIG-Net (CLAUDE.md)
  - Problem framing: chemistry property prediction (JCIM); differs
  - Core mechanism: same two-stage mechanism; match
  - Key insight: same; match
  - Application domain: molecules/chemistry; differs
  - Axes matching: 2 -> **Level 3 — Medium Overlap** (two axes differ)
- **Prior work B** (MODNet)
  - Title: Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet
  - Date: 2021-06
  - Source: doi:10.1038/s41524-021-00552-2
  - Problem framing: Matbench, small data; match
  - Core mechanism: NMI feature selection + joint-learning MLP; differs
  - Key insight: relevance-redundancy feature selection for small data; differs
  - Application domain: inorganic materials, Matbench; match
  - Axes matching: 2 -> **Level 3 — Medium Overlap** (two axes differ)
- **Prior work C** (Kauwe et al.)
  - Title: Extracting Knowledge from DFT: Experimental Band Gap Predictions Through Ensemble Learning
  - Date: 2020
  - Source: doi:10.1007/s40192-020-00178-0
  - Problem framing: experimental band gap, random split; partial
  - Core mechanism: stacked ensemble with DFT-model features; partial (no local correction)
  - Key insight: DFT knowledge transfer; differs
  - Application domain: composition-based experimental band gaps; match
  - Axes matching: 2 (worst case, counting the partial framing) -> **Level 3 — Medium Overlap** (two axes differ)
- **Prior work D** (Shmalo)
  - Title: Neural Means and Kernel Corrections for Operator Learning
  - Date: 2026
  - Source: doi:10.5281/zenodo.23329780
  - Problem framing: operator learning; differs
  - Core mechanism: convex stack + global KRR residual correction; partial -> counted as match (worst case)
  - Key insight: residual alignment limits averaging; differs
  - Application domain: PDE surrogates; differs
  - Axes matching: 1 -> **Level 4 — Low Overlap** (three axes differ)
- **Prior work E** (Chang et al., mixture of experts): 1 axis -> Level 4 — Low Overlap
- **Prior work F** (Zhao et al., HEA stacking): 1 axis -> Level 4 — Low Overlap
- **Prior work G** (Miccio, ensemble of experts, Comput. Mater. Sci. 2025; read after the owner supplied the PDF): pre-trained neural experts on related polymer properties from SMILES (transfer learning); at most 1 axis (partial framing) -> Level 4 — Low Overlap
- Recalled classics (Wolpert; Breiman; Super Learner; regression kriging): each matches only on mechanism (one component) -> Level 4 — Low Overlap

Worst case (minimum level): **Level 3 — Medium Overlap**.
