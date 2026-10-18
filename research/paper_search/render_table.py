import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
keep = set(int(x) for x in sys.argv[2].split(','))
ab = {'semantic_scholar':'SS','open_alex':'OpenAlex','arxiv':'arXiv','openreview':'OR','crossref':'Crossref','dblp':'DBLP'}
hits = d['per_source_hits']
n = d['n_unique']; nraw = d['n_raw']
print(f"per-source hits: {', '.join(f'{k}={v}' for k,v in hits.items())} · {n} unique ({nraw-n} duplicates merged) · {n-len(keep)} filtered as irrelevant\n")
print("| # | Title | Date | Venue | Citations | Score | Sources |\n|---|---|---|---|---|---|---|")
for i, p in enumerate(d['ranked'], 1):
    if i not in keep: continue
    url = p.get('url') or (('https://doi.org/'+p['doi']) if p.get('doi') else '')
    t = (('[survey] ' if p['is_survey'] else '') + (p.get('title') or '')).replace('|','/')
    v = (p.get('venue') or '').replace('|','/')[:50]
    print(f"| [{i}]({url}) | {t} | {(p.get('publication_date') or str(p.get('year') or ''))[:7]} | {v} | {p.get('citation_count') or 0} | {p['relevance_score']} | {', '.join(ab.get(s,s) for s in p['found_in'])} |")
