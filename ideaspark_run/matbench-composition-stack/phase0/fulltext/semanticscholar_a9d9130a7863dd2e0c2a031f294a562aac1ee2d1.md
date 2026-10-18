# A comprehensive benchmark of active learning strategies with AutoML for small-sample regression in materials science

paper_id: semanticscholar:a9d9130a7863dd2e0c2a031f294a562aac1ee2d1
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

The high cost and difficulty of acquiring labeled data in materials science often limits the scale of data-driven modeling efforts. Experimental synthesis and characterization often require expert knowledge, expensive equipment, and time-consuming procedures, making it critical to develop data-efficient learning strategies. Integrating Automated Machine Learning (AutoML) with active learning enables the construction of robust material-property prediction models while substantially reducing the volume of labeled data required. This benchmark study aims to evaluate various active learning (AL) strategies within AutoML in materials science regression tasks. The performance of each strategy in terms of model accuracy and data efficiency is analyzed. The 9 datasets used are derived from materials formulation design, which are typically small due to high data acquisition costs. 17 active-learning strategies, together with a Random-Sampling baseline, are systematically evaluated and compared for their effectiveness. Early in the acquisition process, uncertainty-driven (LCMD, Tree-based-R) and diversity-hybrid (RD-GS) strategies clearly outperform geometry-only heuristics (GSx, EGAL) and baseline, selecting more informative samples and improving model accuracy. As the labeled set grows, the gap narrows and all 17 methods converge, indicating diminishing returns from AL under AutoML.

## Method


