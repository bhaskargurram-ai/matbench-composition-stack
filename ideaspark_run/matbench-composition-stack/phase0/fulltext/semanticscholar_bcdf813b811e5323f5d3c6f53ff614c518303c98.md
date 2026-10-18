# In context learning foundation models for materials property prediction with small datasets

paper_id: semanticscholar:bcdf813b811e5323f5d3c6f53ff614c518303c98
tier: T2
source_used: html_arxiv
warning: none

## Intro

The discovery and design of novel materials are fundamental for developing energy storage, semiconductor technology, catalysis, and numerous other industrial applications, critical to addressing global challenges and fostering transformative progress in modern society.
A central bottleneck in the material discovery process is accurate and fast prediction of fundamental materials properties such as ion conductivity, thermal conductivity, formation energy, band gap, dielectric constants, elastic moduli, etc. Such material property prediction models have been widely used in screening new functional materials from known materials repositories
[
1
]
, inverse materials design
[
2
]
, and guiding experimental explorations
[
3
]
.
While quantum-mechanical simulations such as density functional theory (DFT)
may achieve accurate predictions for some materials properties, they are usually computationally expensive and cannot be scaled to the millions of candidate compounds required for high-throughput virtual screening. This has motivated the rapid development of machine learning (ML) methods for materials property prediction as a complementary approach to accelerate materials discovery.
In the past decade, machine-learning-based materials property prediction models
have achieved significant progress
[
4
]
and are increasingly integrated into materials discovery pipelines
[
5
,
6
]
. These include composition-based property prediction algorithms based on the widely used Magpie (2016) features
[
7
]
, recent
attention-based neural network Roost (2020)
[
8
]
and CrabNet (2021)
[
9
]
, neural network based ModNet (2021)
[
10
]
, and large-language-model-based Darwin (2023)
[
11
]
, which learn to quickly predict material properties directly from the composition, making them suitable for high-throughput screening candidate materials without structural information.
Compared to composition-based predictors, structure-based material property prediction models have demonstrated higher performance and stronger generalization across diverse property datasets, especially those graph neural networks (GNN) based models
[
12
]
from early CGCNN
[
13
]
to more recent ALIGNN
[
14
]
, DeeperGATGNN
[
15
]
, M3GNet
[
16
]
, coGN, coNGN
[
17
]
, and CHGNet
[
18
]
. These models incorporate atomic connectivity and geometric features to extract better structural and physicochemical features from the structures for accurate property prediction. A series of architectural innovations in neural networks have been introduced to model multi-body interactions and bond angles
[
16
,
14
,
19
,
20
]
. To deal with periodicity
[
21
]
and transformations such as translation, rotation, and reflection using equivariant neural networks
[
20
,
22
,
23
,
21
]
, Transformers
[
24
,
25
]
, graph transformers
[
20
,
21
,
26
]
, transfer learning
[
25
,
27
]
, active learning
[
28
]
, self-supervised learning and the pretraining+finetuning approaches
[
29
]
, and multi-modal neural networks
[
30
,
31
]
, have been proposed to improve material property prediction.
Despite substantial research progress, the challenge of accurately predicting material properties from small datasets remains unsolved. This is evident from the stagnation of prediction performance on the Matbench
[
4
]
leaderboards since 2023, where most mean absolute error (MAE) improvements of state-of-the-art (SOTA) algorithms over ALIGNN (2021)
[
14
]
are below 10%. In a separate study, the PMCGNN (2025)
[
32
]
model, which combines global and local modules, reported an MAE of 0.038 GPa for bulk moduli prediction (a 5.00% reduction compared to PotNet
[
33
]
, 2023) and 0.063 GPa for shear moduli prediction (a 3.08% reduction compared to PotNet, 2023). In another study of 2025
[
25
]
, their sophisticated CrysCo algorithm achieves similar performance (with <3.5% improvement) for six out of eight datasets compared to ALIGNN (2021). Their transfer learning model CrysCoT also only achieved marginal improvement (<3%) compared to CrysCo as it is easy to overfit the target datasets during the finetuning stage. Similar marginal improvements can also be found in the multi-modal model
[
30
]
.
Most materials property prediction tasks with high-demand industrial applications are inherently defined over small datasets such as ionic conductivity
[
34
]
, thermal conductivity
[
35
]
, elastic moduli
[
4
]
, piezoelectric tensors
[
36
]
, and dielectric properties
[
4
]
, usually containing a few hundreds to a few hundreds of samples. The complexity of the structure-property relationship coupled with such small datasets and GNNs’ high expressive power naturally leads to overfitting: the model learns the training examples including its noises instead of learning general rules. This weakness is further compounded by inherent material dataset redundancy
[
37
,
38
,
39
]
and the challenge for out-of-distribution predictions needed for discovering novel outlier/exceptional materials
[
40
,
41
]
Recently, the broader field of machine learning has been transformed
by the emergence of foundation models (FMs)
[
42
,
43
,
44
,
45
]
.
Unlike task-specific architectures, FMs are trained on massive and heterogeneous datasets using scalable architectures and objectives,
producing general-purpose representations that can be transferred across diverse downstream tasks. This paradigm has reshaped diverse fields such as natural language processing
[
46
]
, computer vision
[
47
]
, medical imaging
[
48
]
, and chemistry
[
49
]
. Foundation models can be used for addressing the persistent ML challenges such as data scarcity and poor generalization
[
50
]
. In the field of materials science
[
51
]
, large-language model (LLM) based foundation models have been applied to diverse materials discovery tasks such as data extraction, property prediction, and synthesis planning
[
51
]
. However, it remains a challenge how to link such LLM models to highly specialized GNN models for materials property prediction. FMs trained with supervised pretraining over large-scale structure-formation energy data (1.58M) have recently become popular in machine learning potentials models such as MACE, SevenNet, eSEN as shown in the matbench-discovery leaderboard
[
52
]
. Extract universal features from such models for generic property prediction is nontrivial as effective transfer depends on appropriate alignment between pretraining and target tasks. So far, both self-supervised learning based
[
53
]
and multi-modality FMs
[
30
]
have yet to demonstrate significant performance improvement for small-dataset material property prediction.
These findings establish in context learning FMs
[
54
,
55
]
as a versatile and generalizable paradigm for scientific property prediction, highlighting both their opportunities and current limitations, and laying the groundwork for FM-enabled, plug-and-play pipelines to accelerate materials discovery.
In summary, this work introduces a unified in-context learning foundation model (ICL-FM) framework for materials property prediction that bridges composition-based and structure-aware representations. By coupling the pretrained TabPFN transformer with both handcrafted Magpie descriptors and graph-based embeddings from ALIGNN and CGCNN, we demonstrate that foundation models can achieve state-of-the-art accuracy while providing interpretable insights into material-property relationships. Comprehensive evaluations across the MatBench benchmark suite and a standalone thermal conductivity dataset confirm the model’s robustness, generalization, and physical consistency. This study establishes ICL-FM as a versatile and data-efficient paradigm for advancing foundation-model applications in materials informatics.

## Method

Magpie and Extended Compositional Features
For composition-based tasks, raw chemical formulas were converted into numerical feature vectors using the standard Magpie featurization pipeline as implemented in the Matminer package. The MagpieEX feature set is implemented in house using Python and Pymatgen Package and is provided with open-source code. The detailed definitions of the four feature sets (Magpie, MagpieEX, Magpie+Struct1, and Magpie+Struct2)
are provided in the ”Results” section.
Baseline Models
To contextualize the performance of the proposed method, we compare against a diverse set of established baseline models for materials property prediction, spanning both structure-based and composition-based learning paradigms. These include graph neural networks that operate directly on crystal structures as well as attention-based models that rely solely on chemical composition. A brief description of each baseline model and its key architectural characteristics is provided in Supplementary Note S2.
Structural Features and Graph Embeddings
In Table
3
, each embedding corresponds to a specific stage/layer of the ALIGNN network architecture, illustrated schematically in Supplementary Figure S1.
In Table
4
, three ALIGNN–FM integration strategies were evaluated:
GNN Input + FM
,
Last-Layer + FM
, and
Feature-Fusion + FM
.
Figure
7
provides a schematic overview of the representations listed in Table
4
.
Figure 7:
Schematic illustration of the three ALIGNN–FM integration strategies used in Table 4 of the main text.
(a)
GNN Input + FM:
raw structural tensors containing atomic indices and neighbor distances.
(b)
Last-Layer
foundation model trained on ALIGNN readout embeddings.
(c)
Feature-Fusion + FM:
multi-layer concatenation of ALIGNN embeddings before FM adaptation.
These configurations evaluate how different structural hierarchies contribute to foundation-model learning.
Adapted and modified from Gupta
et al.
[
67
]
.
The
GNN Input
representation was generated as follows. Raw crystal structures were converted into fixed-size tabular tensors inspired by the ALIGNN framework. Each structure was encoded as a tensor of shape
[
N
,
1
+
2
​
M
]
[N,1+2M]
, where
N
N
is the maximum number of atoms per unit cell (set to 16) and
M
M
is the maximum number of nearest neighbors per atom (set to 12). Each row begins with the atomic number, followed by the indices of its
M
M
nearest neighbors (within an 8 Å cutoff) and their corresponding pairwise distances. Structures with fewer than
N
N
atoms or atoms with fewer than
M
M
neighbors were zero-padded with
-1
, ensuring a consistent input dimension for all samples.
This encoding preserves atomic connectivity and distance relationships while producing a uniform feature structure suitable for ICL-FMs.
Training and Evaluation Protocol
Following the official
Matbench
benchmarking pipeline, we ensured fair and reproducible evaluation across all tasks.
The benchmark provides standardized datasets and fixed train/test splits designed for consistent comparison of materials property prediction models.
We downloaded all tasks using the
matbench
Python package (
matbench.client.MatbenchBenchmark
) and strictly adhered to the predefined splits to avoid data leakage and ensure comparability with prior work.
Each benchmark task was trained independently on its corresponding training set, and model performance was evaluated using the official Matbench scoring functions.
All regression tasks were assessed using Mean Absolute Errors (MAEs) and, where applicable, Root Mean Squared Error (RMSE).
This standardized protocol establishes a consistent and transparent basis for comparing our ICL-FM framework
with other models reported in the Matbench literature.
The same 5-fold cross-validation protocol was applied to the standalone thermal conductivity dataset to ensure consistency with the MatBench evaluation scheme.
To enable fair comparison among different models, we fixed the random seeds used for data splitting, ensuring identical training and test partitions across all experiments.
This standardized approach provides a consistent basis for comparing the ICL-FM framework with baseline and state-of-the-art models across both composition- and structure-based tasks.
SHAP Analysis
Feature importance was analyzed using the SHapley Additive exPlanations (SHAP) framework to quantify the contribution of individual features and hierarchical groups to model predictions.
For each trained model, we computed mean absolute SHAP values using the official SHAP Python package, aggregated across all test folds. This analysis was performed for both composition-based and structure-based models, including hierarchical attribution of atomic-, edge-, and angle-level features in the ALIGNN–FM configuration.
All visualizations were generated using the
summary_plot
and
bar_plot
functions from the SHAP library.
t-SNE Representation Mapping
To visualize the learned feature spaces, we applied t-distributed stochastic neighbor embedding (t-SNE) to raw features and latent embeddings from both the foundation model (FM) and baseline models.
The t-SNE projections were computed using the
scikit-learn
implementation with a perplexity of 30, learning rate of 200, and 5,000 iterations.
Each data point represents a material sample and is color-coded by its measured or predicted property value, providing an interpretable two-dimensional view of how the FM reorganizes compositional and structural representations in relation to the target property.
Implementation Details & Computational Environment
All experiments were implemented in Python using model-specific environments for reproducibility.
PyTorch Geometric
was used for GNNs (CGCNN, ALIGNN),
TensorFlow
for coNGN, and custom transformer utilities for TabPFN.
Each model’s code, preprocessing, and evaluation scripts are organized in separate GitHub folders with accompanying environment files (
requirements.txt
or
environment.yml
).
TabPFN version
Since TabPFN is an evolving foundation model, all experiments involving TabPFN were conducted using
TabPFN v2.2.1
, rather than the original 2022 release, to ensure consistency across evaluations.
TabPFN performs inference within seconds due to its single-pass evaluation mechanism, whereas GNN-based models such as ALIGNN typically require several hours of training on larger datasets.
All experiments were conducted on a workstation equipped with an NVIDIA RTX 4090 GPU (24 GB VRAM), an Intel i7–8700K CPU, and 32 GB of system memory.
