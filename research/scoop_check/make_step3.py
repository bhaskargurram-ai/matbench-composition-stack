import json, datetime
from triage_spec import T
d = json.load(open('search_results.json', encoding='utf-8'))
recalled = [
 ("About regression-kriging: From equations to case studies (Hengl, Heuvelink, Rossiter)", "2007", "Regression kriging for spatial prediction", "Trend model + kriging of its residuals in geographic space", "Residuals are spatially autocorrelated, so kriging them adds information", "Geostatistics / soil mapping", 1, "model-recall; DOI 10.1016/j.cageo.2007.05.001 (verify)"),
 ("Stacked generalization (Wolpert)", "1992", "General supervised learning", "Level-1 learner on cross-validated level-0 predictions", "CV predictions estimate generalisation biases", "General ML", 1, "model-recall; Neural Networks 5(2) (verify)"),
 ("Stacked regressions (Breiman)", "1996", "Regression", "Non-negative least-squares combination of CV predictions", "Non-negativity constraints stabilise stacking", "General ML", 1, "model-recall; Machine Learning 24 (verify)"),
 ("Super Learner (van der Laan, Polley, Hubbard)", "2007", "Prediction with oracle guarantees", "Cross-validated convex combination of a library of learners", "CV-selected combination is asymptotically as good as the oracle", "Biostatistics / general", 1, "model-recall; Stat. Appl. Genet. Mol. Biol. 6 (verify)"),
 ("SIG-Net (owner's own first paper, submitted to JCIM 2026-10-18)", "2026-10", "Molecular/chemical property prediction (per CLAUDE.md; the paper itself was not opened, other project folders are out of scope)", "Cross-fitted stacking of diverse experts + novelty-damped local OOF residual correction", "Local OOF residual structure that the global stack misses; damped where the query is novel", "Chemistry (JCIM)", 2, "project context (CLAUDE.md); github.com/bhaskargurram-ai/SIG-Net"),
]
out = [f"# Step 3 - Abstract-level triage\n\nTimestamp: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n{len(d['ranked'])} deduplicated search records + {len(recalled)} recalled records. Records not in the hand-triaged set are outside materials property prediction or are generic k-NN/ensemble papers from other fields; their fields are abbreviated and overlap is 0 unless noted. Abstracts are in `search_results.json`.\n"]
for i, p in enumerate(d['ranked'], 1):
    date = (p.get('publication_date') or str(p.get('year') or ''))[:7]
    src = p.get('doi') and ('doi:' + p['doi']) or p.get('arxiv_id') and ('arXiv:' + p['arxiv_id']) or p.get('url')
    if i in T:
        f, m, k, a, o = T[i]
    else:
        f = m = k = "off-topic for this claim (see title)"; a = (p.get('venue') or 'other field')[:60]; o = 0
    out.append(f"---\n**[{i}] {p['title']}**\n- Title: {p['title']}\n- Date: {date}\n- Problem framing: {f}\n- Core mechanism: {m}\n- Key insight: {k}\n- Application domain: {a}\n- Overlap score: {o}\n- Source: {src}\n")
for j, (t, dt, f, m, k, a, o, s) in enumerate(recalled, 1):
    out.append(f"---\n**[R{j}] {t}**\n- Title: {t}\n- Date: {dt}\n- Problem framing: {f}\n- Core mechanism: {m}\n- Key insight: {k}\n- Application domain: {a}\n- Overlap score: {o}\n- Source: {s} (Source: model-recall)\n")
open('step3.md', 'w', encoding='utf-8').write("\n".join(out))
