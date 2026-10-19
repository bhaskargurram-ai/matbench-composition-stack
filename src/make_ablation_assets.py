"""Manuscript macros/tables for the post-hoc ablation, generated from results/ablation_summary.json
(written once by src.evaluate_ablation). Nothing here is typed by hand; re-running does not re-score.
Usage: python -m src.make_ablation_assets"""
from __future__ import annotations
import json
from .data import ROOT

GEN = ROOT / "manuscript" / "generated"
SHORT = {"matbench_steels": "steels", "matbench_expt_gap": "expt\\_gap", "matbench_expt_is_metal": "expt\\_is\\_metal",
         "matbench_glass": "glass"}
UNIT = {"matbench_steels": "MAE (MPa)", "matbench_expt_gap": "MAE (eV)", "matbench_expt_is_metal": "bal.\\ acc.",
        "matbench_glass": "bal.\\ acc."}
MACRO = {"matbench_steels": "Steels", "matbench_expt_gap": "Gap", "matbench_expt_is_metal": "Metal", "matbench_glass": "Glass"}
VM = {"nodft": "NoDft", "nohurdle": "NoHurdle"}
VNAME = {"nodft": "no DFT block", "nohurdle": "no hurdle experts"}
REG = {"matbench_steels", "matbench_expt_gap"}
ORDER = ["nodft:matbench_expt_gap", "nohurdle:matbench_expt_gap", "nodft:matbench_expt_is_metal",
         "nodft:matbench_steels", "nodft:matbench_glass"]


def f(x, t):
    return f"{x:.2f}" if t == "matbench_steels" else f"{x:.4f}"


def fd(x, t):
    return f"{x:+.2f}" if t == "matbench_steels" else f"{x:+.4f}"


def fp(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def verdict(t, d, p):
    good = d < 0 if t in REG else d > 0
    if p >= 0.05:
        return "on par"
    return "better" if good else "worse"


def main(src=None, gen=None):
    """src/gen override only for smoke tests on synthetic input."""
    global GEN
    GEN = gen or (ROOT / "manuscript" / "generated")
    S = json.load(open(src or (ROOT / "results" / "ablation_summary.json")))
    M, rows, frows = [], [], []
    m = lambda k, v: M.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    for key in ORDER:
        r = S[key]; t = r["task"]; v = r["variant"]; k = VM[v] + MACRO[t]
        worse = sum((x > 0) if t in REG else (x < 0) for x in r["per_fold_delta_vs_a2"])
        m(f"Abl{k}", f(r["variant_metric"], t)); m(f"Abl{k}Std", f(r["variant_std"], t))
        m(f"Abl{k}Delta", fd(r["delta_vs_a2"], t))
        m(f"Abl{k}Rel", f"{100 * r['delta_vs_a2'] / r['a2_metric']:+.1f}")
        m(f"Abl{k}CIlo", fd(r["ci_vs_a2"][0], t)); m(f"Abl{k}CIhi", fd(r["ci_vs_a2"][1], t))
        m(f"Abl{k}P", fp(r["p_vs_a2"])); m(f"Abl{k}FoldsWorse", str(worse))
        m(f"Abl{k}VsA2Verdict", verdict(t, r["delta_vs_a2"], r["p_vs_a2"]))
        m(f"Abl{k}VsLeader", fd(r["delta_vs_leader"], t))
        m(f"Abl{k}VsLeaderCIlo", fd(r["ci_vs_leader"][0], t)); m(f"Abl{k}VsLeaderCIhi", fd(r["ci_vs_leader"][1], t))
        m(f"Abl{k}VsLeaderP", fp(r["p_vs_leader"]))
        m(f"Abl{k}VsLeaderVerdict", verdict(t, r["delta_vs_leader"], r["p_vs_leader"]))
        m(f"Abl{k}CorrOn", str(r["corr_on_folds"]))
        if "prob_rocauc_variant" in r:
            m(f"Abl{k}AUC", f"{r['prob_rocauc_variant']:.4f}"); m(f"Abl{k}AtwoAUC", f"{r['prob_rocauc_a2']:.4f}")
        if t == "matbench_expt_gap":
            # share of the full pipeline's advantage over the leader that disappears without the component
            adv = r["leader_metric"] - r["a2_metric"]
            m(f"Abl{k}ShareOfLeaderGain", f"{100 * r['delta_vs_a2'] / adv:.0f}")
        rows.append(f"{SHORT[t]} & {UNIT[t]} & {VNAME[v]} & {f(r['a2_metric'], t)} $\\pm$ {f(r['a2_std'], t)} & "
                    f"{f(r['variant_metric'], t)} $\\pm$ {f(r['variant_std'], t)} & "
                    f"{fd(r['delta_vs_a2'], t)} [{fd(r['ci_vs_a2'][0], t)}, {fd(r['ci_vs_a2'][1], t)}] & {fp(r['p_vs_a2'])} & "
                    f"{worse}/5 & {r['leader']}: {f(r['leader_metric'], t)} & "
                    f"{fd(r['delta_vs_leader'], t)} [{fd(r['ci_vs_leader'][0], t)}, {fd(r['ci_vs_leader'][1], t)}] & "
                    f"{fp(r['p_vs_leader'])} \\\\")
        frows.append(f"{SHORT[t]} & {VNAME[v]} & " + " & ".join(
            f"{f(a, t)} / {f(b, t)}" for a, b in zip(r["a2_folds"], r["variant_folds"])) +
                     (f" & {r['prob_rocauc_a2']:.4f} / {r['prob_rocauc_variant']:.4f}" if "prob_rocauc_variant" in r else " & --") +
                     f" & {r['corr_on_folds']}/5 \\\\")
    (GEN / "numbers_ablation.tex").write_text("\n".join(M) + "\n", encoding="utf-8")
    tab = [r"\begin{tabular}{lllcccccccc}", r"\toprule",
           r" & & & \multicolumn{2}{c}{mean $\pm$ s.d.\ over 5 folds} & \multicolumn{3}{c}{variant $-$ full A2} & "
           r"\multicolumn{3}{c}{variant $-$ leader} \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
           r"Task & Metric & Variant & Full A2 & Variant & $\Delta$ [95\% CI] & $p_\mathrm{boot}$ & folds worse & "
           r"Leader & $\Delta$ [95\% CI] & $p_\mathrm{boot}$ \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_ablation.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
    ftab = [r"\begin{tabular}{llccccccc}", r"\toprule",
            r"Task & Variant & fold 0 & fold 1 & fold 2 & fold 3 & fold 4 & prob.\ ROC-AUC & corr.\ kept \\", r"\midrule"] + \
        frows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_ablation_folds.tex").write_text("\n".join(ftab) + "\n", encoding="utf-8")
    print("wrote", GEN / "numbers_ablation.tex", GEN / "table_ablation.tex", GEN / "table_ablation_folds.tex")


if __name__ == "__main__":
    main()
