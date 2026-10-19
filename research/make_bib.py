"""Build manuscript/references.bib from VERIFIED metadata only.
Crossref-verified entries come from research/reference_verification.json (research/verify_refs.py).
The remaining entries were verified individually (source noted in each `note`-free comment below)."""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
V = json.load(open(ROOT / "research" / "reference_verification.json", encoding="utf-8"))


def esc(s):
    s = s.replace("&amp;", "&").replace("&", "\\&").replace("Δ", "$\\Delta$")
    return s


def authors(lst):
    return " and ".join(a.replace("Pierre-Paul De Breuck", "Pierre-Paul {De Breuck}") for a in lst)


out = []
for k, v in V.items():
    if not v.get("ok"):
        continue
    t = v.get("type")
    title = esc(re.sub(r"<[^>]+>", "", v["title"]))
    if t == "proceedings-article":
        out.append(f"@inproceedings{{{k},\n  author = {{{authors(v['authors'])}}},\n  title = {{{{{title}}}}},\n"
                   f"  booktitle = {{{esc(v['venue'])}}},\n  year = {{{v['year']}}},\n  pages = {{{v.get('page') or ''}}},\n  doi = {{{v['doi']}}}\n}}")
    else:
        out.append(f"@article{{{k},\n  author = {{{authors(v['authors'])}}},\n  title = {{{{{title}}}}},\n"
                   f"  journal = {{{esc(v['venue'])}}},\n  year = {{{v['year']}}},\n  volume = {{{v.get('volume') or ''}}},\n"
                   f"  pages = {{{v.get('page') or ''}}},\n  doi = {{{v['doi']}}}\n}}")

# Individually verified (2026-10-18):
# - Shmalo: Zenodo API record 23329780 (DataCite DOI, preprint, published 2026-09-07)
# - MatterVial: arXiv API 2509.03547 (published 2025-09-02; authors from the arXiv record)
# - scikit-learn: arXiv API 1201.0490 + JMLR 12 (2011) 2825-2830
# - LightGBM: proceedings.neurips.cc page (NeurIPS 30, 2017)
# - Matbench leaderboard: matbench.materialsproject.org (accessed 2026-10-18)
# - SIG-Net: the author's own repository, github.com/bhaskargurram-ai/SIG-Net (manuscript submitted)
out.append("""@misc{shmalo2026kernel,
  author = {Yitzchak Shmalo},
  title = {{Neural Means and Kernel Corrections for Operator Learning}},
  year = {2026},
  howpublished = {Zenodo preprint},
  doi = {10.5281/zenodo.23329780}
}""")
out.append("""@misc{gouvea2025mattervial,
  author = {Rog{\\'e}rio Almeida Gouv{\\^e}a and Pierre-Paul {De Breuck} and Tatiane Pretto and Gian-Marco Rignanese and Marcos Jos{\\'e} Leite Santos},
  title = {{Combining feature-based approaches with graph neural networks and symbolic regression for synergistic performance and interpretability}},
  year = {2025},
  howpublished = {arXiv:2509.03547}
}""")
out.append("""@article{pedregosa2011sklearn,
  author = {Fabian Pedregosa and Ga{\\"e}l Varoquaux and Alexandre Gramfort and Vincent Michel and Bertrand Thirion and Olivier Grisel and Mathieu Blondel and Peter Prettenhofer and Ron Weiss and Vincent Dubourg and Jake Vanderplas and Alexandre Passos and David Cournapeau and Matthieu Brucher and Matthieu Perrot and {\\'E}douard Duchesnay},
  title = {{Scikit-learn: Machine Learning in Python}},
  journal = {Journal of Machine Learning Research},
  year = {2011},
  volume = {12},
  pages = {2825--2830}
}""")
out.append("""@inproceedings{ke2017lightgbm,
  author = {Guolin Ke and Qi Meng and Thomas Finley and Taifeng Wang and Wei Chen and Weidong Ma and Qiwei Ye and Tie-Yan Liu},
  title = {{LightGBM: A Highly Efficient Gradient Boosting Decision Tree}},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year = {2017}
}""")
out.append("""@misc{matbenchboard,
  author = {{Matbench developers}},
  title = {{Matbench leaderboard (per-task leaderboards, matbench v0.1)}},
  howpublished = {\\url{https://matbench.materialsproject.org/}},
  note = {Accessed 18 October 2026},
  year = {2026}
}""")
out.append("""@misc{gurram2026signet,
  author = {Bhaskar Gurram},
  title = {{SIG-Net}: cross-fitted stacking with a novelty-damped local out-of-fold residual correction},
  howpublished = {Manuscript submitted; code at \\url{https://github.com/bhaskargurram-ai/SIG-Net}},
  year = {2026}
}""")
p = ROOT / "manuscript" / "references.bib"
p.write_text("\n\n".join(out) + "\n", encoding="utf-8")
print("wrote", p, len(out), "entries")
