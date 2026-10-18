# Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability

paper_id: dblp:journals/corr/abs-2509-03547
tier: H
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Machine learning has revolutionized materials science, accelerating material discovery and 
property optimization across various domains1–3. The two prominent approaches in this field 
are feature-based and graph-neural-network (GNN) models, each with distinct advantages and 
limitations4,5. Feature-based models rely on predefined descriptors such as elemental 
properties, geometric features, and electronic structure information. They are highly 
interpretable and effective with small datasets, offering insights into structure-property 
relationships6,7. These models adapt well to custom tasks in experimental settings, such as 
nanocrystal research8, catalysis9, and organic photovoltaics10. In contrast, GNN models 
represent materials as graphs, capturing structural information through message passing and 
learning deep representations with simple atomic descriptors. This often results in more 
accurate predictions for complex materials, but requires greater computational resources and 
data for training11,12. GNNs are particularly effective in the large-scale screening of materials 
and for constructing interatomic potentials owing to their efficient computation and local 
information aggregation,13 however they lack interpretability.  
Boosting the accuracy of feature-based models to make them competitive on larger datasets 
usually implies employing neural network models and relying on extensive suites, such as 
MatMiner7, to produce meaningful features. This process is particularly time-consuming for 
sophisticated descriptors like the Orbital Field Matrix (OFM)14 and the Smooth Overlap of 
Atomic Positions (SOAP)15. A novel strategy to boost these feature-based models involves 
leveraging the rich latent-space representations learned by GNN models pretrained on vast 
datasets. Even though neural networks are universal function approximators, easing their 
burden through well-aligned feature transformations can improve generalization, reduce 
training time, and stabilize convergence16,17.  


In this work, we address these challenges by proposing a hybrid approach that combines 
traditional chemically intuitive descriptors with latent features obtained from a diverse set of 
pretrained models. We incorporate features from both structure-based (MEGNet, coGN)18,19 
and composition-based (ROOST)20 GNNs, as well as from ORB21, a powerful equivariant 
Machine Learning Interatomic Potential (MLIP). To avoid the featurization bottleneck of 
traditional descriptors, we also leverage GNNs to generate fast, latent-space approximations of 
MatMiner (ℓ-MM) and Orbital Field Matrix (ℓ-OFM) features. Finally, we augment this feature 
set with new descriptors derived via symbolic regression. This multifaceted strategy aims to 
create a more robust, accurate, and versatile featurizer that capitalizes on the distinct strengths 
of each approach to be useful for a wider range of dataset sizes. 
To simplify the generation of all those features, a package was developed named MatterVial 
standing for MATerials feaTuRe Extraction Via Interpretable Artificial Learning, which, 
besides producing all latent-space features from the GNN models, aids in obtaining the 
interpretable chemical descriptors that correlate to these high-level features. This is achieved 
through techniques such as SHapley Additive exPlanations (SHAP) analysis in surrogate 
models and symbolic regression via Sure Independence Screening and Sparsifying Operator 
(SISSO) to obtain an approximate formula from the most important features. Our results 
demonstrate an overall improvement in all analyzed datasets compared with the baseline 
MatMiner featurizer. In addition, it surpassed the performance of the individual GNN models 
in several cases, indicating that the combination of traditional and latent-space features leads 
to a more robust generalization.  
This work is situated within a recent methodological trend that repurposes GNNs not as end-
to-end predictors, but as powerful and data-efficient feature generators for a variety of 
downstream tasks22–26. Our approach bridges feature-based and graph-based methods, 
leveraging their strengths to develop more versatile and task-agnostic machine learning models 


in materials science. By enhancing the accuracy, efficiency, and interpretability of property 
prediction, this framework facilitates the integration of both experimental and simulated data. 
Moreover, it aligns with the growing demand for explainable AI27,28, which is essential for the 
advancement of self-driving laboratories in materials discovery and optimization29. 


Fig. 1 | Overview of the methodology for leveraging latent-space features from GNN models 
with MatterVial. On the left (I), the generation and deployment of descriptor-oriented GNN 
models are illustrated. At the center (II), task-oriented GNN models—either pretrained or 
trained on the fly for adjacent variants—are shown, with feature extraction possible from 
activation, pooling, or multi-layer perceptron (MLP) layers on the model architecture. On the 
right (III), formulas from symbolic regression from SISSO are also implemented, leveraging 
traditional physiochemical descriptors available in MatMiner as a base.  
Results and discussion 
We evaluate our approach using the full MatBench v0.1 benchmark30 with MODNet4, which 
is the state-of-the-art feature-based model in materials science12. We adopt the same MatMiner 
featurization as that used in MODNet for MatBench in the original publication4,31. These can 
be complemented by three categories of MatterVial features, as illustrated in Fig. 1: 
I. 
Latent-space features from descriptor-oriented GNNs: Conventional material 
descriptors are transformed into latent representations using an autoencoder trained on 
Materials Project (MP) data. These descriptors include the widely used MatMiner 


features (ℓ-MM) and the features from the Orbital Field Matrix featurizer (ℓ-OFM). A 
GNN was then trained to replicate these latent features directly from the input 
structures. This method achieves a computational efficiency similar to that of GNNs 
and still preserves interpretability via decoding. 
II. 
Latent-space features from task-oriented GNNs: These features are extracted 
directly from the intermediate layers of pretrained GNN models that have been 
developed for various tasks. Specifically, we incorporate MEGNet models from the 
Materials Virtual Lab (MVL) that were pretrained for the prediction of elastic constants, 
band gap, and formation energy, as well as for the metal-insulator classification. We 
also consider composition-based ROOST models for the band gap and formation 
energy. In addition, we include the internal layers of ORB-v3, a state-of-the-art 
equivariant MLIP trained to reproduce energies and forces. This group capitalizes on 
the strengths of GNN architectures in capturing complex structural representations, 
aiming to enhance predictive performance on larger datasets. 
III. 
Symbolically-Derived Feature Combinations: Here, we use the MatMiner features 
as a basis to generate new compound features. Through symbolic regression with 
SISSO, we identify several combinations of pairs of features (rung one) that exhibit 
enhanced correlations with the target properties of interest in materials science. These 
derived formulas are then incorporated as new features. 
Since the features obtained from task-oriented GNNs are high-level and not directly 
interpretable as traditional descriptors, we develop a method to decompose them into 
interpretable descriptors, which is integrated in the Interpreter module in MatterVial. In 
addition, features from descriptor-oriented GNNs can be decoded in their interpretable 
counterparts. Equally, the third group of augmented features via symbolic regression can have 
their formulas retrieved by name. Comprehensive implementation details for each category and 


for the Interpreter module are available in the Methods section and in the Supplementary 
Information. 
MatBench validation of MatterVial features 
Table 1 presents the performance of MODNet using MatMiner augmented with MatterVial 
descriptors (MODNet@MM+MV) and MODNet using only MatterVial descriptors 
(MODNet@MV) relative to the baseline model using only MatMiner features 
(MODNet@MM) in the 13 MatBench tasks. The results show that blending both latent-space 
representations from task-oriented and descriptor-oriented GNNs with symbolically derived 
features consistently reduces prediction errors across this diverse array of property prediction 
tasks.

## Method

based models have traditionally outperformed GNNs. Specifically, our models set new 
performance records for four tasks previously led by MODNet@MM and now achieve a 
leading performance in metallicity classification from experimental data. Notably, the glass-
forming ability task alone did not result in substantial improvements. We highlight that for 
smaller composition-based datasets, MatMiner featurization is sufficiently fast to make 
MODNet@MM+MV computationally effective. For larger datasets, in which traditional 
featurization is very time consuming, our MODNet@MV models significantly bridge the gap 
between feature-based and graph-based models, even outperforming state-of-the-art (SOTA) 
models in predicting properties such as elastic constants, band gap, metallicity, and formation 
energy. This success demonstrates that our approach effectively addresses the common 
shortcomings of both feature- and graph-based models. Note, however, that some of the larger 
MatBench tasks can no longer be considered truly independent test sets for models exposed to 
vast amounts of similar ab initio data during pretraining. 




Table 1 | Performance comparison of three MODNet variants against the best multi-
purpose MatBench model on each task in the MatBench v0.1 benchmark. Metrics are 
reported as mean absolute error (MAE) for regression and area under the receiver-
operator curve (AUROC) for classification. MODNet@MM uses only MatMiner 
features; MODNet@MM+MV augments these features with MatterVial descriptors; and 
MODNet@MV uses only MatterVial features essentially substituting MatMiner features 
by ℓ-MM. For each task, the MatterVial feature group that yields the best result is shown. 
Scores in bold identify the overall best model per task, and shaded tasks are those in 
which MODNet was already the best model.   
MatBench task 
𝑛 
MODNet@ 
MM 
(baseline) 
MODNet@ 
MM+MV 
(% error 
reduction*) 
MODNet@ 
MV 
(% error 
reduction*) 
MatBench 
record 
(model) 
Best 
MatterVial 
groups** 
Steels yield strength 
(MPa) 32 
312
87.76
𝟖𝟓. 𝟏𝟐(𝟑. 𝟎%) 
120.95
(-37.8%)
MODNet
ROOST
Eexfol. (meV/atom)33 
636
33.19
29.19 (12.1%)
28.86 (13.0%) 
MODNet
ORB, MVL, ℓ-
OFM, ℓ-MM
argmax(PhDOS) (cm-1) 34 
1,265
34.27
𝟑𝟎. 𝟎𝟖(12.2%)
30.58 (10.8%) 
𝟐𝟖. 𝟕𝟔
(MegNet)
MVL, ORB, ℓ-
OFM, ROOST, 
SISSO
Exp. band gap (eV) 35 
4,604
0.333
𝟎. 𝟐𝟗𝟎(𝟏𝟐. 𝟗%) 
0.351 (-5.5%)
MODNet
ROOST, SISSO
Refractive index 36,37 
4,764
0.271
0.235 (13.3%)
0.234 (13.7%) 
MODNet
ORB, ℓ-OFM, 
MVL, ℓ-MM
Exp. metallicity (eV) 35  
4,921
0.916
𝟎. 𝟗𝟕𝟔(𝟕𝟏. 𝟒%) 
0.898 (-59.3%)
0.921
(AMMExpress)
ROOST
Glass-forming ability 
38,39 
5,680
0.936
(𝟎. 𝟗𝟔𝟎)†
0.937 (1.6%)
0.904 (-50.0%)
MODNet
ROOST
Logarithmic Gvrh 
(log10GPa) 40 
10,987
0.073
𝟎. 𝟎𝟑𝟐(𝟓𝟓. 𝟓%) 
0.033 (54.8%)
0.067 (coGN)
MVL, ORB, ℓ-
MM, ROOST, 
SISSO
Logarithmic Kvrh 
(log10GPa) 40  
10,987
0.056
𝟎. 𝟎𝟐𝟕(𝟒𝟗. 𝟔%) 
0.028 (50.1%)
0.049 (coNGN)
MVL, ORB, ℓ-
MM, ℓ-OFM,  
ROOST, SISSO
Perovskite ΔHform 
(eV/unitcell)41 
18,928
0.0908
0.0386 (57.5%)
0.0389 (57.3%) 
𝟎. 𝟎𝟐𝟔𝟗(coGN) 
ORB, MVL, ℓ-
OFM, ℓ-MM, 
ROOST, SISSO
Band gap (eV) 37 
106,113
0.2199
0.137 (37.6%)
0.137 (37.8%) 
0.156 (coGN)
MVL, ORB,
ROOST, SISSO
Metallicity 37 
106,113
0.904
𝟎. 𝟗𝟕𝟖(𝟕𝟕. 𝟏%) 
0.976 (75.0%) 
0.9520
(CGCNN)
ORB, MVL, ℓ-
OFM, ROOST
Ef (eV/atom) 37 
132,752
0.0448
0.0147 (67.2%)
0.0138 (69.2%) 
0.0170 (coGN)
MVL, ORB, ℓ-
OFM, ℓ-MM
* % error reduction =
ெ஺ா್ೌ
ೞ೐೗೔೙೐ିெ஺ா೘೚೏೐೗
ெ஺ா್ೌ
ೞ೐೗೔೙೐
× 100% (regression)   or    
(ଵି஺௎ோை஼್ೌ
ೞ೐೗೔೙೐)ି(ଵି஺௎ோை஼೘೚೏೐೗)
(ଵି஺௎ோை஼್ೌ
ೞ೐೗೔೙೐)
× 100% (classification) 
** Ordered by importance, MVL, ORB and ROOST refer to the task-oriented GNN features, respectively those from MVL MEGNet 
models for structures, the MLIP Orb-v3 and pretrained ROOST models for compositions. ℓ-MM and ℓ-OFM refer to the descriptor-
oriented GNN features, ℓ-MM when included, substitutes the MatMiner features for faster generation. SISSO refers to the group of 
features derived from MM features via symbolic regression. 
† As we were unable to replicate the reported 0.960 AUROC for glass formability using MODNet, we present our best MODNet@MM
