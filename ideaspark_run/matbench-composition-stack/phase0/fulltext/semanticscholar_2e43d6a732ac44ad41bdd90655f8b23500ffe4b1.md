# Enhancing composition-based materials property prediction by cross-modal knowledge transfer

paper_id: semanticscholar:2e43d6a732ac44ad41bdd90655f8b23500ffe4b1
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro



## Method

chemical language
model
masked language
modeling
crystal structure
predictor
crystal graph
neural network
target
property
chemical formula
chemical formula
generated crystal
structure
multimodal
foundational
model
multitask
regression
(A)
(B)
Figure 1: A conceptual scheme of cross-modal knowledge transfer formulations. (A) Implicit knowledge
transfer pipeline includes two pretraining phases: masked language modeling (on chemical symbols and stoichiometric
coefficients) and multitask regression on multimodal embeddings produced within the foundational model (MultiMat).
(B) Explicit knowledge transfer pipeline involves crystal structure prediction in a high-throughput manner and
materials property prediction using structure-aware model.
the development of universal representations that capture domain knowledge inherently. A pioneering model
of this kind, ElemNet[16], was based on a 17-layered fully-connected architecture, whereas the input vectors
consisted of element fractions. Its successors[17, 18, 19, 20] have been advanced by incorporating pretrained
element embeddings[21] and attention mechanisms[22, 23] into training. Diverse pretraining strategies,
including self-supervised learning, fingerprint learning, and multimodal learning, have been applied to improve
the efficacy of the Representation Learning from Stoichiometry (Roost) framework[17] on downstream tasks.
One more step in developing accurate structure-based models has been made by Na[24], who proposed the
cross-modal transfer learning approach for modeling experimental properties.
The aforementioned deep learning approaches incorporate chemical composition as a set of elemental or
atom-wise attributes, whereas the advent of chemical language models (CLMs) reframes the composition-based
property prediction as a sequence modeling task. The models originally trained via masked language modeling
(MLM) on materials science abstracts[25] and crystal text descriptions[26] have been finetuned and evaluated
in reproducing diverse quantities[27]. Here, we take a next step in developing CLMs aimed at composition-
based property prediction. In contrast to the previously mentioned MatBERT[25] and LLM-Prop[26] models,
multiple modalities beyond natural language were incorporated into pretraining (Figure 1A). After the MLM
stage, embeddings from CLMs were aligned to those of the foundation model recently presented under the
framework of multimodal learning for materials[8] (MultiMat). We utilized the crystal structure encoder that
was contrastively pretrained on four materials modalities: crystal structure, density of electronic states, charge
density, and textual description. This approach can be characterized as implicit cross-modal Knowledge
Transfer (imKT) since the considered property predictors, CLMs, directly operate on modality of interest,
i.e., chemical composition. Alternatively, materials property prediction was explicitly transferred (exKT)
from the compositional to structural domain with the large language model—CrystaLLM[28]—served as a
crystal structure predictor, followed by GNN finetuned on the generated crystals (Figure 1B). The rest of the
paper is devoted to the demonstration of how cross-modal knowledge transfer affects predictive performance,
taking into account the current state-of-the-art (SOTA) algorithms. In addition, explainability analysis of
CLM outputs is presented, considering high-order token interactions.
Table 1 presents a comparison of our best-performing models with SOTA algorithms available in the literature;
a comprehensive picture of how accurately all models perform is provided in Tables S1–S5. Considering 20
tasks from the JARVIS-DFT dataset (LLM4Mat-Bench[27]), substantial improvement (from 4.5 to 39.6%) in
mean absolute error (MAE) was achieved in 18 cases; on average, the MAE decreased by 15.7%. Knowledge
transfer failed to reduce MAE for two tasks: energy above the convex hull and maximal piezoelectric strain
coefficient. Prediction of the former property is expectedly complicated by introducing stability-aware weights
2

Enhancing composition-based materials property prediction
A Preprint
Table 1: Predictive performance comparison of transfer knowledge models with existing architec-
tures. Three separated groups of tasks correspond to JARVIS-DFT, SNUMAT (both are the LLM4Mat-Bench
datasets), and MatBench benchmarks. Mean absolute error (MAE) values are provided for the models
demonstrating state-of-the-art (SOTA) performance.
SOTA existing models
SOTA presented models
predictive task
architecture
MAE ↓
performance
boost ↑
MAE ↓
architecture
FEPA
MatBERT-109M
0.126
+8.8%
0.11488 ± 0.00018 imKT@ModernBERT
Band gap (OPT)
MatBERT-109M
0.235
+15.5%
0.1985 ± 0.0019
imKT@BERT
Total energy
MatBERT-109M
0.194
+39.6%
0.1172 ± 0.0005
imKT@ModernBERT
Ehull
MatBERT-109M
0.096
–
0.1031 ± 0.0009
imKT@RoFormer
Band gap (MBJ)
MatBERT-109M
0.491
+23.2%
0.3773 ± 0.0030
imKT@ModernBERT
Kv
MatBERT-109M
18.498
+11.6%
16.35 ± 0.24
imKT@ModernBERT
Gv
MatBERT-109M
14.241
+10.4%
12.76 ± 0.05
imKT@ModernBERT
SLME
MatBERT-109M
5.851
+16.1%
4.911 ± 0.010
imKT@ModernBERT
Spillage
MatBERT-109M
0.409
+15.4%
0.3462 ± 0.0029
imKT@ModernBERT
ϵx (OPT)
MatBERT-109M
32.661
+25.5%
24.32 ± 0.06
imKT@ModernBERT
ϵ
Gemma2-9b-it:5S
28.228
+5.8%
26.6 ± 0.4
imKT@RoFormer
Max. piezo. (dij)
Gemma2-9b-it:5S
7.973
–
9.67 ± 0.10
imKT@ModernBERT
Max. piezo. (eij)
LLM-Prop-35M
0.156
+4.5%
0.1490 ± 0.0026
imKT@BERT
Max. EFG
MatBERT-109M
26.621
+12.5%
23.30 ± 0.11
imKT@ModernBERT
Exfoliation energy
MatBERT-109M
37.445
+21.2%
29.5 ± 1.4
imKT@RoFormer
avg. me
MatBERT-109M
0.103
+18.7%
0.0837 ± 0.0010
imKT@ModernBERT
n-Seebeck
MatBERT-109M
58.342
+16.7%
48.6 ± 0.5
imKT@ModernBERT
n-PF
MatBERT-109M
528.070
+6.5%
493.7 ± 1.7
imKT@ModernBERT
p-Seebeck
MatBERT-109M
61.085
+17.8%
50.22 ± 0.06
imKT@ModernBERT
p-PF
LLM-Prop-35M
544.737
+12.2%
478.5 ± 1.4
imKT@ModernBERT
Band gap GGA
MatBERT-109M
0.461
+19.9%
0.3694 ± 0.0009
imKT@ModernBERT
Band gap HSE
MatBERT-109M
0.553
+21.5%
0.4341 ± 0.0027
imKT@ModernBERT
Band gap GGA optical MatBERT-109M
0.701
+10.3%
0.629 ± 0.004
imKT@ModernBERT
Band gap HSE optical
MatBERT-109M
0.749
+9.1%
0.6811 ± 0.0015
imKT@ModernBERT
Castelli perovskites
AtomSets
0.082 ± 0.001
–
0.149 ± 0.010
imKT@RoFormer
Refractive index
Roost-SSL
0.3122 ± 0.0808
–
0.35 ± 0.09
imKT@RoFormer
log10(shear modulus)
CrabNet
0.092
+4.8%
0.0876 ± 0.0020
imKT@ModernBERT
log10(bulk modulus)
CrabNet
0.068
+1.6%
0.0669 ± 0.0031
imKT@ModernBERT
Experimental band gap
CrabNet
0.338
+7.7%
0.312 ± 0.022
imKT@RoFormer
MP formation energy
CrabNet
77
–
78.9 ± 1.7
imKT@ModernBERT
MP band gap
Finder
0.231
–
0.253 ± 0.004
imKT@RoFormer
Phonon peak
Roost-SSL
46.05 ± 4.22
–
54 ± 4
imKT@RoFormer
in the MLM loss function (see details in Supporting Information, Methods section), so CLMs pretrained on
multimodal embeddings surpassed models trained via the two-stage procedure (Table S6). By applying the
imKT approach, we also significantly improved performance in predicting four band-gap-related tasks from
the SNUMAT dataset (LLM4Mat-Bench); on average, the MAE decreased by 15.2%.
LLM4Mat-Bench is the largest benchmark for evaluating performance of CLMs in predicting properties of
crystalline materials, lacking the assessment of composition-based predictors based on other neural network
architectures. To fill this gap, we utilized another well-established suite of predictive tasks, MatBench[29]
(Tables 1, S4, S5). The imKT models achieved SOTA results on only three out of eight considered tasks;
there is no dominating architecture that provides the best performance on most tasks, as opposed to the
LLM4Mat-Bench datasets. An integrative comparison across datasets was done by calculating the weighted
average of MAD:MAE ratio, where MAD stands for the mean absolute deviation. The best value of 8.25
was achieved by CrabNet, whereas the top-two value of 8.11 corresponds to imKT@ModernBERT (Table
S5). Excellent performance of this imKT model relates to advances in the BERT architecture (ModernBERT
3

Enhancing composition-based materials property prediction
A Preprint
GaAsO3
cubic perovskite
G = 81.38 GPa
MnTa2Os
Heusler
G = 80.06 GPa
(A)
(B)
(C)
Figure 2: Explainability analysis of chemical language model predicting shear modulus. (A) Element-wise
importance scores computed as averaged SHAPley Interaction Quantification (SHAP-IQ) values. Most influential (B)
two- and (C) three-token combinations, according to the averaged SHAP-IQ values. Two crystal structures from the
JARVIS-DFT dataset are depicted to outline most common structural prototypes; the corresponding DFT-computed
shear moduli are provided as well.
outperforms BERT and RoFormer, Tables S1–S5) and to the cross-modal knowledge transfer: multimodal
learning as a pretraining task yields accuracy on par with the two-step procedure (Tables S6–S10). To
sum up, the imKT technique ensures the highest in-class, i.e., across CLMs, performance (according to
LLM4Mat-Bench results), and approaching accuracy of the best-performing architecture (i.e., CrabNet), as it
follows from the MatBench results.
Analyzing Tables S1–S5 reveals that the imKT models surpass the exKT models across all tasks, lowering
MAE by 7.4% on average. We assume that including structural information in the prediction pipeline does not
improve accuracy because of two reasons. First, DFT-computed benchmarks contain a substantial fraction of
metastable compounds, which are not a target of CSP. For instance, 37% of the JARVIS-DFT compounds
have an energy above the convex hull exceeding 0.1 eV/atom, and 6% are highly unstable (Ehull > 1.0
eV/atom). Second, the applied CSP approach has limited ability to output the most stable polymorphs:
considering the JARVIS-DFT compounds with Ehull < 0.01 eV/atom, only a minor portion (11.7%) of the
generated crystal structures suites the original ones, in accordance with the structure matching procedure
implemented in the pymatgen[30]. Whereas the former issue is inherent to the task, the latter can be addressed
4

Enhancing composition-based materials property prediction
A Preprint
by further developing high-throughput CSP algorithms. It should be also noted that training and inference
of GPT-like and GNN models (used for generating crystal structures and reproducing target quantities,
respectively) may result in substantial computational demands, which are another reason to prioritize CLMs
as composition-based property predictors. Nevertheless, we consider exKT as a promising technique, taking
into account its current performance: CLMs presented in LLM4Mat-Bench were outperformed in 22 out of 24
cases, summarizing the JARVIS-DFT and SNUMAT tasks (Tables S1–S3).
Neural networks for materials property prediction are typically utilized as “black-boxes”, missing understanding
of their internal machinery. A few exceptions among composition-based models are transformers with
attention attributes[18, 20], which are debatably related to feature importances[31]. In this context, we
applied a post hoc explainability technique that includes any-order feature interactions to gain insights into
structure-property relationships, going beyond element-wise contributions. Specifically, the outputs of the
imKT@ModernBERT model for predicting shear modulus (JARVIS-DFT dataset, LLM4Mat-Bench) were
processed using the SHAPley Interaction Quantification[32] (SHAP-IQ) approach. As a starting point for
analysis, the averaged elemental contributions are provided in Figure 2A. Expectedly, the highest impact
on the shear modulus value is exerted by the presence of some platinum group metals and non-metals
forming (ultra)hard materials, e.g., borides, carbides, and silicides. Considering the most influential two-token
combinations (Figure 2B), we conclude that oxygen-containing groups reduce the target quantity. In contrast,
boron-containing sequences increase the shear modulus value in most cases, though further insights can
hardly be gained. Three-token combinations are most informative features, which extend our understanding
beyond a compositional aspect (Figure 2C). In particular, the “1 O 3” sequence relates to the cubic perovskite
prototype (CaTiO3, E21 Strukturbericht designation, Pm¯3m space group) in 38% of cases, as it follows
from analysis of the corresponding JARVIS-DFT crystal structures. The Heusler structure (AlCu2Mn, L21
Strukturbericht designation, Fm¯3m space group) was identified for a noticeable part of “2...1...1” and
“1...2...1” sequences (41% and 35% of cases, respectively). However, there are no preferable prototypes for
other three-token sequences. These findings represent an early example of interpreting chemical compositions
through the crystallographic prototype analysis and advanced game-theoretic approach (i.e., SHAP-IQ),
which is a promising combination for enhancing CLM explainability.
The presented approach—cross-modal knowledge transfer—is proved to be an effective route for enhancing
efficacy of composition-based materials property prediction. Its modular structure (in both formulations)
allows replacing distinct components.
In the case of implicit transfer, we expect further enhancement
owing to the development of multimodal representation learning and CLMs. On the other hand, explicit
transfer can be advanced by resolving issues related to generating unplausible crystal structures; introducing
multimodal foundational models as property predictors operating on hypothetical structures is another avenue
for empowering the exKT pipeline. Finally, advanced ensembling techniques, e.g., mixture of experts, were
shown to be a powerful auxiliary tool for increasing accuracy of heterogenous property predictors[33]; two
formulations of cross-modal knowledge transfer can be unified within this approach.
2
Conflicts of interest
There are no conflicts of interest to declare.
3
Funding
This work was supported by the Ministry of Economic Development of the Russian Federation in accordance
with the subsidy agreement (agreement identifier 000000C313925P4H0002; grant No 139-15-2025-012).
4
Acknowledgments
We are grateful to Viggo Moro and other developers of the MultiMat framework for sharing model checkpoint,
which was used for pretraining chemical language models.
5
Data and software availability
Source code for preprocessing data and training models can be found at https://github.com/korolewadim/
multimat-modernbert. The ModernBERT-base model pretrained via a two-step procedure is available at
https://huggingface.co/korolewadim/multimat-modernbert.
5

Enhancing composition-based materials property prediction
A Preprint
References
[1] Alex Zunger. Inverse design in search of materials with target functionalities. Nature Reviews Chemistry,
2(4):0121, 2018.
[2] Keith T Butler, Daniel W Davies, Hugh Cartwright, Olexandr Isayev, and Aron Walsh. Machine learning
for molecular and materials science. Nature, 559(7715):547–555, 2018.
[3] Patrick Reiser, Marlen Neubert, André Eberhard, Luca Torresi, Chen Zhou, Chen Shao, Houssam Metni,
Clint van Hoesel, Henrik Schopmans, Timo Sommer, et al. Graph neural networks for materials science
and chemistry. Communications Materials, 3(1):93, 2022.
[4] Tian Xie and Jeffrey C Grossman. Crystal graph convolutional neural networks for an accurate and
interpretable prediction of material properties. Physical review letters, 120(14):145301, 2018.
[5] Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong. Graph networks as a universal
machine learning framework for molecules and crystals. Chemistry of Materials, 31(9):3564–3572, 2019.
[6] Kamal Choudhary and Brian DeCost. Atomistic line graph neural network for improved materials
property predictions. npj Computational Materials, 7(1):185, 2021.
[7] Àlex Solé, Albert Mosella-Montoro, Joan Cardona, Silvia Gómez-Coca, Daniel Aravena, Eliseo Ruiz,
and Javier Ruiz-Hidalgo. A cartesian encoding graph neural network for crystal structure property
prediction: application to thermal ellipsoid estimation. Digital Discovery, 4(3):694–710, 2025.
[8] Viggo Moro, Charlotte Loh, Rumen Dangovski, Ali Ghorashi, Andrew Ma, Zhuo Chen, Samuel Kim,
Peter Y Lu, Thomas Christensen, and Marin Soljačić. Multimodal foundation models for material
property prediction and discovery. Newton, 1(1), 2025.
[9] Daniel W Davies, Keith T Butler, Adam J Jackson, Andrew Morris, Jarvist M Frost, Jonathan M
Skelton, and Aron Walsh. Computational screening of all stoichiometric inorganic materials. Chem,
1(4):617–627, 2016.
[10] Artem R Oganov, Chris J Pickard, Qiang Zhu, and Richard J Needs. Structure prediction drives materials
discovery. Nature Reviews Materials, 4(5):331–348, 2019.
[11] Mohammad Alghadeer, Nufida D Aisyah, Mahmoud Hezam, Saad M Alqahtani, Ahmer AB Baloch,
and Fahhad H Alharbi. Machine learning prediction of materials properties from chemical composition:
Status and prospects. Chemical Physics Reviews, 5(4), 2024.
[12] Logan Ward, Ankit Agrawal, Alok Choudhary, and Christopher Wolverton. A general-purpose machine
learning framework for predicting properties of inorganic materials.
npj Computational Materials,
2(1):1–7, 2016.
[13] Ya Zhuo, Aria Mansouri Tehrani, and Jakoah Brgoch. Predicting the band gaps of inorganic solids by
machine learning. The journal of physical chemistry letters, 9(7):1668–1673, 2018.
[14] Luca M Ghiringhelli, Jan Vybiral, Sergey V Levchenko, Claudia Draxl, and Matthias Scheffler. Big data
of materials science: critical role of the descriptor. Physical review letters, 114(10):105503, 2015.
[15] Runhai Ouyang, Stefano Curtarolo, Emre Ahmetcik, Matthias Scheffler, and Luca M Ghiringhelli. Sisso:
A compressed-sensing method for identifying the best low-dimensional descriptor in an immensity of
offered candidates. Physical Review Materials, 2(8):083802, 2018.
[16] Dipendra Jha, Logan Ward, Arindam Paul, Wei-keng Liao, Alok Choudhary, Chris Wolverton, and
Ankit Agrawal. Elemnet: Deep learning the chemistry of materials from only elemental composition.
Scientific reports, 8(1):17593, 2018.
[17] Rhys EA Goodall and Alpha A Lee. Predicting materials properties without crystal structure: deep
representation learning from stoichiometry. Nature communications, 11(1):6280, 2020.
[18] Anthony Yu-Tung Wang, Steven K Kauwe, Ryan J Murdock, and Taylor D Sparks. Compositionally
restricted attention-based network for materials property predictions. Npj Computational Materials,
7(1):77, 2021.
[19] Chi Chen and Shyue Ping Ong. Atomsets as a hierarchical transfer learning framework for small and
large materials datasets. npj Computational Materials, 7(1):173, 2021.
[20] Achintha Ihalage and Yang Hao.
Formula graph self-attention network for representation-domain
independent materials discovery. Advanced Science, 9(18):2200164, 2022.
6

Enhancing composition-based materials property prediction
A Preprint
[21] Vahe Tshitoyan, John Dagdelen, Leigh Weston, Alexander Dunn, Ziqin Rong, Olga Kononova, Kristin A
Persson, Gerbrand Ceder, and Anubhav Jain. Unsupervised word embeddings capture latent knowledge
from materials science literature. Nature, 571(7763):95–98, 2019.
[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing
systems, 30, 2017.
[23] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio.
Graph attention networks. In International Conference on Learning Representations, 2018.
[24] Gyoung S Na. Cross-modality material embedding loss for transferring knowledge between heterogeneous
material descriptors. npj Computational Materials, 11(1):235, 2025.
[25] Amalie Trewartha, Nicholas Walker, Haoyan Huo, Sanghoon Lee, Kevin Cruse, John Dagdelen, Alexander
Dunn, Kristin A Persson, Gerbrand Ceder, and Anubhav Jain. Quantifying the advantage of domain-
specific pre-training on named entity recognition ta
