# Physics-Informed Hybrid Machine Learning Model for Carbonation Depth Prediction in Concrete through Residual Correction and Variogram Analysis of Response Surfaces

paper_id: semanticscholar:67274a738894a915857a99a8cffd8a90e6ec534a
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro


 Carbonation-induced deterioration of reinforced concrete is a major durability concern, as it reduces pore solution alkalinity and accelerates reinforcement corrosion. Conventional service life models often oversimplify the combined effects of material and environmental factors, limiting their predictive reliability. This study presents a hybrid residual correction framework that integrates a physics-based carbonation model with a stacked ensemble of machine learning algorithms: gradient boosted regression trees (GBRT), support vector regression (SVR), and Gaussian process regression (GPR), combined through an XGBoost metamodel. Unlike conventional stacking, the metamodel is trained on residuals between physical model predictions and experimental measurements. This enables systematic correction of mechanistic biases while retaining physical interpretability. A second contribution is the application of variogram analysis of response surfaces (VARS) for variance-based global sensitivity analysis, which quantifies feature influence using geostatistical indicators (sill, nugget, and range), offering insights beyond standard feature importance methods. The model was deployed as an app within a MATLAB-based user interface (UI) to promote practical use, enabling service life prediction from minimal, easily measurable input parameters. The framework was validated against accelerated carbonation experiments and long-term natural exposure data from the literature. The results demonstrate that the residual-based metamodel reproduced observed carbonation depths with higher accuracy than individual base learners or the physical model.

## Method


