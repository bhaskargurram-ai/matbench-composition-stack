"""Generate manuscript numbers/tables/figures for the Amendment-2 results (results/final_a2) and the
A1-vs-A2 comparison. Nothing here is edited by hand. Usage: python -m src.make_paper_assets2"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from .data import ROOT, TASKS

RES = ROOT / "results" / "final_a2"
GEN = ROOT / "manuscript" / "generated"
FIG = ROOT / "manuscript" / "figures"
SHORT = {"matbench_steels": "steels", "matbench_expt_gap": "expt\\_gap", "matbench_expt_is_metal": "expt\\_is\\_metal",
         "matbench_glass": "glass"}
MACRO = {"matbench_steels": "Steels", "matbench_expt_gap": "Gap", "matbench_expt_is_metal": "Metal", "matbench_glass": "Glass"}
REG = {"matbench_steels", "matbench_expt_gap"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def f(x, t):
    return f"{x:.2f}" if t == "matbench_steels" else f"{x:.4f}"


def fp(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    st = json.load(open(RES / "stats.json")); sm = json.load(open(RES / "summary.json"))
    per = pd.read_csv(RES / "metrics_per_fold.csv")
    lb = json.load(open(ROOT / "results" / "leaderboard_snapshot.json"))
    a1 = json.load(open(ROOT / "results" / "final" / "summary.json"))
    M = []
    m = lambda k, v: M.append(f"\\newcommand{{\\{k}}}{{{v}}}")

    # ---- main table: final vs leader (board semantics) + rank
    L = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"Task & Metric & Final pipeline & Leader (board) & $\Delta$ [95\% CI] & $p_\mathrm{Holm}$ & Rank \\", r"\midrule"]
    for t in TASKS:
        s = st[t]["primary_vs_leader"]; k = MACRO[t]
        ours_sd = float(np.std(s["ours_folds"], ddof=1))
        ent = lb[t]["entries"]
        if t in REG:
            rank = 1 + sum(e[1] < s["ours"] for e in ent)
        else:
            rank = 1 + sum(e[4] > s["ours"] for e in ent)
        good = (s["delta"] < 0) if t in REG else (s["delta"] > 0)
        verdict = ("beats leader" if good and s["p_holm"] < 0.05 else
                   "below leader" if (not good) and s["p_holm"] < 0.05 else "on par")
        L.append(f"{SHORT[t]} & {'MAE' if t in REG else 'bal.\\ acc.'} & {f(s['ours'], t)} $\\pm$ {f(ours_sd, t)} & "
                 f"{s['leader']}: {f(s['leader_value'], t)} & {s['delta']:+.4g} [{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}] & "
                 f"{fp(s['p_holm'])} & {rank} of {len(ent) + 1} \\\\")
        m(f"AtwoFinal{k}", f(s["ours"], t)); m(f"AtwoFinal{k}Std", f(ours_sd, t)); m(f"AtwoLeader{k}", f(s["leader_value"], t))
        m(f"AtwoLeaderName{k}", s["leader"]); m(f"AtwoDelta{k}", f"{s['delta']:+.4g}"); m(f"AtwoPHolm{k}", fp(s["p_holm"]))
        m(f"AtwoCIlo{k}", f"{s['ci95'][0]:+.3g}"); m(f"AtwoCIhi{k}", f"{s['ci95'][1]:+.3g}")
        m(f"AtwoRank{k}", str(rank)); m(f"AtwoNEntries{k}", str(len(ent) + 1)); m(f"AtwoVerdict{k}", verdict)
        m(f"AtwoRel{k}", f"{100 * s['delta'] / s['leader_value']:+.1f}")
        a = st[t]["a2_vs_a1"]; m(f"AtwoVsAone{k}", f"{a['delta']:+.4g}"); m(f"AtwoVsAoneP{k}", fp(a["p_boot"]))
        c = st[t]["stack_vs_best_single"]; m(f"AtwoStackVsBest{k}", f"{c['rel_change_pct']:+.2f}"); m(f"AtwoStackVsBestP{k}", fp(c["boot_p"]))
        c = st[t]["local_vs_stack"]; m(f"AtwoLocalVsStack{k}", f"{c['rel_change_pct']:+.2f}"); m(f"AtwoLocalVsStackP{k}", fp(c["boot_p"]))
        sels = st[t]["selections"]
        m(f"AtwoCorrOn{k}", str(sum(1 for x in sels if x["sel_stack_local"].get("alpha", 0) > 0)))
        if "prob_rocauc_ours" in st[t]:
            m(f"AtwoAUC{k}", f"{st[t]['prob_rocauc_ours']:.4f}")
            if "prob_rocauc_leader" in st[t]:
                m(f"AtwoLeaderAUC{k}", f"{st[t]['prob_rocauc_leader']:.4f}")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "a2_table_leader.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- arms table (native metrics) for A2
    arms = ["best_single", "mean_ensemble", "stack", "stack_local", "single_local"]
    lab = {"best_single": "Best single (inner CV)", "mean_ensemble": "Equal-weight mean", "stack": "Simplex stack",
           "stack_local": "Final: stack + gated local corr.", "single_local": "Best single + gated corr."}
    L = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Arm & " + " & ".join(SHORT[t] for t in TASKS) + r" \\",
         " & MAE (MPa) & MAE (eV) & ROC-AUC & ROC-AUC \\\\", r"\midrule"]
    for a in arms:
        cells = []
        for t in TASKS:
            met = "mae" if t in REG else "rocauc"
            x = sm[t].get(a, {}).get(met)
            cells.append(f"{f(x['mean'], t)} $\\pm$ {f(x['std'], t)}" if x else "--")
        L.append(lab[a] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    singles = sorted({a for t in TASKS for a in sm[t] if a.startswith("single:")})
    for a in singles:
        cells = []
        for t in TASKS:
            met = "mae" if t in REG else "rocauc"
            x = sm[t].get(a, {}).get(met)
            cells.append(f(x["mean"], t) if x else "--")
        L.append("Expert " + a.split(":", 1)[1].replace("_", "\\_") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "a2_table_arms.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- weights table
    exps = sorted({e for t in TASKS for x in st[t]["selections"] for e in x["stack_weights"]})
    L = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Expert & " + " & ".join(SHORT[t] for t in TASKS) + r" \\", r"\midrule"]
    for e in exps:
        cells = []
        for t in TASKS:
            ws = [x["stack_weights"].get(e) for x in st[t]["selections"]]
            cells.append(f"{np.mean([w for w in ws if w is not None]):.2f}" if any(w is not None for w in ws) else "--")
        L.append(e.replace("_", "\\_") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "a2_table_weights.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- A1 vs A2 table (board metric)
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Task & Amendment 1 (primary arm) & Amendment 2 (final) & $p_\mathrm{boot}$ \\", r"\midrule"]
    for t in TASKS:
        g = per[(per.task == t)]
        v1 = g[g.arm == "A1:stack_local"]["board_metric"].mean(); v2 = g[g.arm == "stack_local"]["board_metric"].mean()
        L.append(f"{SHORT[t]} & {f(v1, t)} & {f(v2, t)} & {fp(st[t]['a2_vs_a1']['p_boot'])} \\\\")
        m(f"AoneBoard{MACRO[t]}", f(v1, t))
    L += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "a2_table_a1a2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- timing
    tf = RES / "timing.jsonl"
    if tf.exists():
        tm = pd.read_json(tf, lines=True); tm = tm[tm.returncode == 0]
        for t in TASKS:
            sub = tm[tm.task == t]
            if len(sub):
                m(f"AtwoWallMin{MACRO[t]}", f"{sub.wall_s.median() / 60:.0f}")
        m("AtwoTotalJobHours", f"{tm.wall_s.sum() / 3600:.1f}")
    (GEN / "numbers_a2.tex").write_text("\n".join(M) + "\n", encoding="utf-8")

    # ---- figure: per-fold board metric, ours vs leader vs A1
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax, t in zip(axes, TASKS):
        g = per[per.task == t]
        lead = [a for a in g.arm.unique() if a.startswith("LEADER:")][0]
        for j, (a, l) in enumerate((("stack_local", "final (A2)"), (lead, lead.split(":", 1)[1]), ("A1:stack_local", "A1 primary"))):
            v = g[g.arm == a].sort_values("fold")["board_metric"].values
            ax.plot(np.arange(5) + (j - 1) * 0.12, v, "o", ms=4, color=COL[j], label=l, mec="#fcfcfb", mew=0.6)
        ax.set_title(SHORT[t].replace("\\", ""), fontsize=8); ax.set_xticks(range(5)); ax.set_xlabel("official fold")
        ax.set_ylabel("MAE" if t in REG else "balanced accuracy", fontsize=7)
        ax.legend(frameon=False, fontsize=5.5, loc="best")
    fig.tight_layout(); fig.savefig(FIG / "fig_a2_leader.pdf"); plt.close(fig)
    print("A2 assets written")


if __name__ == "__main__":
    main()
