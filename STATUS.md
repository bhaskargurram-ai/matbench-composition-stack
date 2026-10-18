# STATUS

**Current stage:** Stage 1 (ResearchStudio-Idea): paper-search and scoop-check are done; idea-spark is next.

## Done
- Experiment dependencies are installed in `.venv`: scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, numpy 2.5.4, pandas 2.3.3, matminer 0.10.1, pymatgen.
  - `matbench==0.6` was installed with `--no-deps`, because its pins (scikit-learn 1.0, monty 2021) cannot build on Python 3.12.
- **Data:** all 4 composition tasks are cached with their official folds (`src/data.py` → `data/cache/`, git-ignored).
  - Magpie (132) + element-fraction (103) features are computed. They are label-free.
  - Fold test sizes: steels 63/63/62/62/62; expt_gap 921×4/920; expt_is_metal 985/984×4; glass 1136×5.
- **paper-search:** `research/paper_search/allinone.md`.
  - Run 1 was degraded. Run 2 used full connectors after the owner added `.env`.
- **scoop-check:** `research/scoop_check/` (step1–7.md, `scoop_check_report.md`). Verdict: **Level 3 — Medium Overlap**.

## Grounding / connector record (honest)
| Connector | Run 1 (no `.env`) | Run 2 (with `.env`, loaded only by the skills' own loaders) |
|---|---|---|
| Semantic Scholar | 10 (HTTP 429) | OK |
| arXiv, OpenAlex, Crossref | OK | OK |
| OpenReview | 0 | 0 (logged-in client, but hit the API's 3-request rate limit; off-venue topic) |
| DBLP | 0 (API returned non-JSON on every query) | 0 (same error) |

- `search_papers.py` has no `--json` flag, although SKILL.md asks for one. `research/run_paper_search.py` therefore calls the skill's own programmatic API.
- Firecrawl credits are exhausted, so it was not used.
- **One candidate could not be read:** Miccio 2025, Comput. Mater. Sci. (paywalled; no abstract in any index). The owner should check it with library access.

## Key numbers
None yet (no experiments run).

## Problems / blockers
- None blocking.

## Next step
Run `/idea-spark` in `ideaspark_run/matbench-composition-stack/` (navigator `next` loop).
