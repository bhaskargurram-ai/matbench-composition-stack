
The unfiltered set (all 164 records, with abstracts) is in `research/paper_search/paper_search_results.json`.

**Degraded grounding (honest record):** DBLP returned 0 (every query failed with a JSON decode error from the API); OpenReview returned 0 (no `.env` credentials; anonymous client, and these are not ML-venue topics); Semantic Scholar returned only 10 records because the keyless tier hit HTTP 429 on 6 of 7 queries. arXiv, OpenAlex and Crossref ran normally. The installed `search_papers.py` has no `--json` flag although SKILL.md asks for one, so `research/run_paper_search.py` calls the skill's own programmatic API (`search_papers`, `dedup`, `rank`) and writes the JSON.

### Model Knowledge (recalled, not retrieved; every entry must be verified before citation)

| # | Title | Year | Venue | Notes |
|---|---|---|---|---|
| [M1](https://scholar.google.com/scholar?q=A+general-purpose+machine+learning+framework+for+predicting+properties+of+inorganic+materials) | A general-purpose machine learning framework for predicting properties of inorganic materials (Ward et al.) | 2016 | npj Comput. Mater. | Magpie descriptors; outside the 2020 window but the featurisation used here |
| [M2](https://scholar.google.com/scholar?q=Is+domain+knowledge+necessary+for+machine+learning+materials+properties) | Is domain knowledge necessary for machine learning materials properties? (Murdock et al.) | 2020 | IMMI | element-fraction vs. Magpie baselines (uncertain — verify) |
| [M3](https://scholar.google.com/scholar?q=Predicting+the+band+gaps+of+inorganic+solids+by+machine+learning+Zhuo) | Predicting the band gaps of inorganic solids by machine learning (Zhuo, Mansouri Tehrani, Brgoch) | 2018 | J. Phys. Chem. Lett. | source of matbench_expt_gap / expt_is_metal data |
| [M4](https://scholar.google.com/scholar?q=Stacked+generalization+Wolpert) | Stacked generalization (Wolpert) | 1992 | Neural Networks | stacking origin |
| [M5](https://scholar.google.com/scholar?q=Stacked+regressions+Breiman) | Stacked regressions (Breiman) | 1996 | Machine Learning | non-negative stacking weights |
| [M6](https://scholar.google.com/scholar?q=Super+learner+van+der+Laan) | Super Learner (van der Laan, Polley, Hubbard) | 2007 | Stat. Appl. Genet. Mol. Biol. | cross-fitted convex stacking |
| [M7](https://scholar.google.com/scholar?q=Formula+graph+self-attention+network+representation-domain+independent+materials+discovery) | Formula graph self-attention network for representation-domain independent materials discovery (Ihalage, Hao) | 2022 | Adv. Sci. | composition GNN on Matbench (uncertain — verify) |

### Summary

1. **Overview.** Seven queries (Matbench composition ML; MODNet; CrabNet; Roost; stacking ensembles for composition; residual kriging / local correction; Darwin), 2020–2026, 10 per source per query; 164 unique records, 74 judged relevant. The corpus covers the Matbench composition leaderboard models, composition-based stacking in alloy/property niches, and regression-kriging residual correction from geostatistics.
2. **Trends.** Benchmark-defining work clusters in 2020–2021 (Matbench [21], Roost [48], CrabNet [3], MODNet [12, 4]); 2022–2024 adds attention/transformer variants [6, 8, 2] and interpretable deep models [14]; 2024–2026 is dominated by application-specific stacking ensembles for alloys [15, 32, 56] and LLM/foundation-model baselines [55, 42]. Residual kriging lives almost entirely in soil/geoscience [27, 30, 50, 54, 85]. Venues: npj Comput. Mater., Patterns, Mater. Des./Acta-type journals, Comput. Mater. Sci.
3. **Key themes.**
   - *Composition-only leaderboard models* — MODNet feature selection + joint learning, CrabNet attention, Roost message passing [12, 3, 48].
   - *Benchmark protocol and bias* — Matbench/Automatminer, MODNet bias-imbalance study, leakage/composition-redundancy [21, 4, 33].
   - *Stacking/ensembles in materials* — mostly single-dataset, random-split, alloy hardness/strength or band gap [15, 51, 56, 32, 22].
   - *Residual correction / regression kriging* — ML trend + kriged residuals in geographic space [27, 50, 54, 30].
   - *Small-data and extrapolation* — [26, 53, 42, 31].
4. **Keywords frequency (titles of kept papers).**

| Keyword | Count |
|---|---|
| property prediction | 30 |
| machine learning | 26 |
| materials | 40 |
| ensemble / stacking | 14 |
| kriging / residual | 11 |

5. **Most cited (in this set).**

| Rank | Title | Year | Citations |
|---|---|---|---|
| 1 | Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm | 2020 | 405 |
| 2 | Predicting materials properties without crystal structure: deep representation learning from stoichiometry | 2020 | 391 |
| 3 | Compositionally restricted attention-based network for materials property predictions | 2020 | 277 |
| 4 | Machine learning approaches for the prediction of materials properties | 2020 | 255 |
| 5 | Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet | 2021 | 120 |

6. **Most cited by first author (in this set; author names from the JSON records).**

| Rank | Author | Papers in set | Total citations |
|---|---|---|---|
| 1 | Alexander Dunn | 2 | 419 |
| 2 | Rhys E. A. Goodall | 2 | 393 |
| 3 | Anthony Yu-Tung Wang | 3 | 316 |
| 4 | Siwar Chibani | 1 | 255 |
| 5 | Pierre-Paul De Breuck | 2 | 157 |

7. **Recommendations for reading.**
   1. Matbench / Automatminer [21] — the protocol and leaderboard this paper must respect.
   2. MODNet [12] and its bias-imbalance study [4] — current best on steels/glass; the CPU-feasible baseline to beat.
   3. Roost [48] and CrabNet [3] — the learned composition representations that could join the expert pool.
   4. Regression kriging vs. RF [50] — the closest methodological analogue of local OOF residual correction (in geographic space).
   5. Optimism bias / composition redundancy [33] — directly relevant to why a local correction in composition space can look good under random splits.
