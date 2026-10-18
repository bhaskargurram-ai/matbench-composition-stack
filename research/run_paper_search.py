"""Thin wrapper around the ResearchStudio paper-search skill.

The installed search_papers.py CLI has no --json flag (SKILL.md asks for one),
so this calls the skill's own programmatic API (references/programmatic_api.md)
and writes raw per-source results + the deduped/ranked list (with abstracts) to JSON.
"""
import argparse, json, sys, pathlib
SKILL = pathlib.Path(__file__).resolve().parents[1] / ".claude" / "skills" / "paper-search" / "scripts"
sys.path.insert(0, str(SKILL))
from search_papers import search_papers  # noqa: E402
from postprocess import dedup, rank  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True, help="pipe-separated")
ap.add_argument("--start-year", type=int, required=True)
ap.add_argument("--end-year", type=int, required=True)
ap.add_argument("--max-papers", type=int, default=10)
ap.add_argument("--sources", nargs="*", default=None)
ap.add_argument("--json", required=True)
a = ap.parse_args()
qs = [q.strip() for q in a.queries.split("|") if q.strip()]
raw = search_papers(qs, a.start_year, a.end_year, a.max_papers, sources=a.sources, parallel=True)
hits = {k: len(v) for k, v in raw.items()}
merged = dedup(raw)
n_total = sum(hits.values())
ranked, _ = rank(merged, qs)
out = {"queries": qs, "start_year": a.start_year, "end_year": a.end_year,
       "per_source_hits": hits, "n_raw": n_total, "n_unique": len(ranked),
       "ranked": ranked, "raw": raw}
pathlib.Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("per-source hits:", ", ".join(f"{k}={v}" for k, v in hits.items()),
      f"· {len(ranked)} unique ({n_total-len(ranked)} duplicates merged)")
for i, p in enumerate(ranked, 1):
    print(f"{i}\t{p.get('publication_date') or p.get('year')}\t{(p.get('venue') or '')[:40]}\t{p.get('citation_count')}\t{p['relevance_score']}\t{','.join(p['found_in'])}\t{'[survey] ' if p['is_survey'] else ''}{p.get('title')}\t{p.get('doi') or p.get('url')}")
