# Step 2 - Search and deduplicate
Timestamp: 2026-10-18T18:24:21-04:00

Queries sent to paper-search (one multi-query call through the skill's API; 2016-2026; 15 per source per query; full connectors after the owner added `.env`):
- Query 1 - Original-Problem: `composition-based materials property prediction small datasets`
- Query 2 - Broad-Domain: `materials property prediction ensemble learning`
- Query 3 - Method-Signature: `stacking out-of-fold residual correction nearest neighbours kernel`

Per-source hits: arxiv=45, dblp=0, open_alex=45, openreview=0, semantic_scholar=30, crossref=45 -> 137 unique after dedup by DOI / arXiv id / normalised title (28 duplicates merged). DBLP returned non-JSON on every query (connector failure, same as in paper-search); OpenReview returned 0 (logged-in client; rate-limited, and the topic is off-venue). Raw JSON: `search_results.json`; stderr: `search_stderr.txt`.

Recalled papers added (Source: model-recall; not present in the search results; verify before citation):
- About regression-kriging: From equations to case studies (Hengl, Heuvelink, Rossiter, 2007)
- Stacked generalization (Wolpert, 1992)
- Stacked regressions (Breiman, 1996)
- Super Learner (van der Laan, Polley, Hubbard, 2007)
- SIG-Net (owner's first paper, JCIM submission 2026-10-18) - the origin of the mechanism being transferred (project context, CLAUDE.md).
