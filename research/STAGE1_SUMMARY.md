# Stage 1 summary: ideation and prior art (ResearchStudio-Idea)

## Grounding
- paper-search and scoop-check used the skills' own scripts.
  - Run 1 had no `.env`; run 2 used full connectors once the owner provided `.env`, loaded only by the skills' own loaders.
  - DBLP never worked (the API returned non-JSON). OpenReview connected but returned 0 records (rate limit, and the topic is off-venue).
  - Semantic Scholar, arXiv, OpenAlex and Crossref all worked in run 2.
- idea-spark Phase 0 ran with `lit_grounding_mode = real`:
  - 132 records, 74 after the relevance partition, plus 7 connector-verified host references;
  - full text for 7 of 12 deep-read papers.
- Files: `research/paper_search/allinone.md`, `research/scoop_check/scoop_check_report.md`, `ideaspark_run/matbench-composition-stack/`.

## Closest prior work
| Work | Relation | Overlap |
|---|---|---|
| SIG-Net (owner, JCIM submission 2026) | Same two-stage mechanism, chemistry domain. It is the origin of the method and must be cited. | mechanism + insight |
| MODNet (De Breuck et al., npj Comput. Mater. 2021) and MatterVial (Gouvêa et al., arXiv 2509.03547, 2025) | Matbench composition leaders. Each is one global model, with no residual stage. | framing + domain |
| Kauwe, Welker, Sparks (IMMI 2020) | Stacked ensemble for experimental band gaps. No local correction. | domain + part of the mechanism |
| Shmalo (Zenodo 2026) | Convex stack + global kernel-ridge residual correction, in operator learning. | part of the mechanism |
| Miccio (Comput. Mater. Sci. 2025) | "Ensemble of experts" = transfer learning for polymers. | low; cite in related work |
| Ancestors | Stacking (Wolpert 1992; Breiman 1996; Super Learner 2007), regression kriging (Hengl et al. 2007), Δ-learning. | each covers one component |

## Scoop-check verdict
**Level 3 — Medium Overlap.**

No paper combines all four axes, but every component has ancestors. The contribution is therefore a transfer-and-test study on the official Matbench folds, not a new learning principle. The manuscript must claim exactly that.

## Idea card (idea-spark)
**Candidate:** "Fold-Matched Local Residual Correction for Cross-Fitted Composition Stacks on Small Matbench Tasks". Source: `ideaspark_run/matbench-composition-stack/phase3_revise/final_candidate.json`.

**Core idea.** The naive correction adds a neighbour's out-of-fold residual to the prediction of the full-refit stack. That stack was trained on the neighbour, so it has already partly absorbed the neighbour's error (the "absorption term" M_q). The naive correction therefore double-counts, most of all for near-duplicate neighbours, which is not where novelty damping acts.

**Fix.** Transfer `r_j + F^(-m(j))(x_q) − F^full(x_q)`. For the damping/strength selection, build deployment-matched training instances with pair-excluded refits `F^(-m,-m')`.

**Review trail.**
- Coherence gate: no blocking findings; the A+M identity was executed.
- Critique: revise. The revision patch passed the post-revision review with a consistency patch.

**Run outcome: `needs_work`, no rendered Phase 4 cards.** After the review passed, the navigator required "confirmed retrieval terms". But its own instruction to the reviewer was to report empty `confirmed_terms`, so the gate failed (`novelty_reaudit`), and the term-refresh collision could not obtain terms.
- I ran the collision retrieval manually with signature/alias terms (240 hits, `phase3_revise/collision/`).
- A title scan found no fold-matched / pair-excluded residual-correction method.
- This is recorded as a skill-flow inconsistency, not as a validated card. The candidate above is the last reviewed (post-revision pass) version.

## Final design decision: followed the stated direction, plus one pre-registered departure arm
**Followed.**
- Cross-fitted heterogeneous experts → simplex (non-negative, sum-to-one) stack → novelty-damped local OOF residual correction.
- Kernel-weighted neighbour residuals with a pseudo-count κ that shrinks the correction to zero for novel queries.
- Strength α, neighbourhood k, bandwidth h, κ and the composition space are all selected inside each training fold. α = 0 (no correction) is always allowed.
- This is the primary arm, because the paper's goal is to test whether the method confirmed in SIG-Net transfers.

**Departure (adopted from the idea card, as a pre-registered secondary arm `stack_fm`).** The fold-matched transfer with pair-excluded refits.

Why it is secondary, not a replacement:
- It is the idea card's main proposal, and the coherence gate confirmed the double-counting algebra.
- But it is unvalidated. The design-phase inner-CV numbers (fold 0 only) did not show it to be consistently better than the naive correction.

**Simplifications versus the card** (resource-driven, recorded honestly):
- no ElMD distance (we use standardised-Magpie Euclidean and element-fraction L1 instead);
- no TabPFN or MODNet experts (CPU cost and dependencies);
- no separate expert-set selection (the simplex weights select);
- λ(s) uses the family α·S/(S+κ), where S is the kernel mass, selected by training-row loss over all manufactured instances rather than by grouped CV over the inner folds.

**Other design changes, made on fold-0 inner CV only.**
- 21 extra label-free matminer descriptors.
- An L1 (MAE-matched) stacking objective for regression.
- An inner-CV balanced-accuracy threshold for hard-label classification metrics, because the board's "rocauc" values are label-based.
