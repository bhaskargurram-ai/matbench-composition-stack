"""Write manuscript/highlights.txt (Elsevier highlights: 3-5 bullets, <= 85 characters each, including spaces)
from the generated number macros, and check the abstract length (<= 200 words, macros expanded).
Usage: python -m src.make_highlights
"""
from __future__ import annotations
import re
import sys

from .data import ROOT

MS = ROOT / "manuscript"
MACRO_FILES = ["numbers_a2.tex", "numbers_ablation.tex"]


def macros():
    m = {}
    for f in MACRO_FILES:
        for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (MS / "generated" / f).read_text(encoding="utf-8")):
            m[name] = val
    return m


def expand(tex, m):
    tex = re.sub(r"\\(\w+)\{\}", lambda g: m.get(g.group(1), g.group(0)), tex)
    tex = re.sub(r"\\(\w+)(?![\w{])", lambda g: m.get(g.group(1), g.group(0)), tex)
    return tex.replace("~", " ").replace("--", "-")


def main():
    m = macros()
    bullets = [
        "Pre-registered, fully nested stacking on four Matbench composition tasks",
        f"Experimental band-gap MAE {m['AtwoFinalGap']} eV vs {m['AtwoLeaderGap']} eV, best merged Matbench entry",
        f"Without formula-matched PBE DFT features, band-gap MAE rises to {m['AblNoDftGap']} eV",
        "Hurdle experts for the zero-inflated band gap matter about as much as DFT features",
        "Local residual correction never survived nested validation; glass lags",
    ]
    bad = [(b, len(b)) for b in bullets if len(b) > 85]
    if bad or not 3 <= len(bullets) <= 5:
        sys.exit(f"highlight too long: {bad}")
    (MS / "highlights.txt").write_text("Highlights\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n", encoding="utf-8")
    for b in bullets:
        print(f"{len(b):3d}  {b}")
    ab = (MS / "sections" / "v3_abstract.tex").read_text(encoding="utf-8")
    ab = re.sub(r"\\(begin|end)\{abstract\}", "", ab)
    n = len(expand(ab, m).split())
    print("abstract words:", n)
    if n > 200:
        sys.exit("abstract longer than 200 words")


if __name__ == "__main__":
    main()
