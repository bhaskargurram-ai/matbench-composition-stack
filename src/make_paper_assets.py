"""Generate every number, table and figure used in the manuscript from result files.
Nothing in manuscript/generated or manuscript/figures is edited by hand.

Usage: python -m src.make_paper_assets [results_dir]
"""
from __future__ import annotations
import json, sys, pathlib
import numpy as np, pandas as pd

from .data import ROOT, TASKS

RES = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "final"
GEN = ROOT / "manuscript" / "generated"
FIG = ROOT / "manuscript" / "figures"
SHORT = {"matbench_steels": "steels", "matbench_expt_gap": "expt\\_gap", "matbench_expt_is_metal": "expt\\_is\\_metal",
         "matbench_glass": "glass"}
MACRO = {"matbench_steels": "Steels", "matbench_expt_gap": "Gap", "matbench_expt_is_metal": "Metal", "matbench_glass": "Glass"}
REG = {"matbench_steels", "matbench_expt_gap"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ARM_LABEL = {"best_single": "Best single (inner CV)", "mean_ensemble": "Equal-weight mean", "stack": "Simplex stack",
             "stack_local": "Stack + local corr. (naive)", "stack_local_magpie": "\\quad ablation: Magpie space only",
             "stack_local_frac": "\\quad ablation: fraction space only", "stack_fm": "Stack + fold-matched corr.",
             "single_local": "Best single + local corr."}
ARM_SHORT = {"best_single": "Best single", "stack": "Stack", "stack_local": "Stack + naive corr.",
             "stack_fm": "Stack + fold-matched corr."}


def fmt(x, task):
    return f"{x:.2f}" if task == "matbench_steels" else f"{x:.4f}"


def fmt_p(p):
    if p != p:
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def macro_name(s):
    return "".join(ch for ch in s.title() if ch.isalpha())


def main():
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    stats = json.load(open(RES / "stats.json"))
    summ = json.load(open(RES / "summary.json"))
    per = pd.read_csv(RES / "metrics_per_fold.csv")
    lb = json.load(open(ROOT / "results" / "leaderboard_snapshot.json"))
    macros = []

    def M(name, val):
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # ---------------- table: all arms
    arms = ["best_single", "mean_ensemble", "stack", "stack_local", "stack_local_magpie", "stack_local_frac",
            "stack_fm", "single_local"]
    lines = [r"\begin{tabular}{l" + "c" * len(TASKS) + "}", r"\toprule",
             "Arm & " + " & ".join(f"{SHORT[t]}" for t in TASKS) + r" \\",
             " & " + " & ".join("MAE" + (" (MPa)" if t == "matbench_steels" else " (eV)") if t in REG else "ROC-AUC"
                                for t in TASKS) + r" \\", r"\midrule"]
    for a in arms:
        cells = []
        for t in TASKS:
            m = "mae" if t in REG else "rocauc"
            if a in summ[t] and m in summ[t][a]:
                cells.append(f"{fmt(summ[t][a][m]['mean'], t)} $\\pm$ {fmt(summ[t][a][m]['std'], t)}")
            else:
                cells.append("--")
        lines.append(ARM_LABEL[a] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    singles = sorted({a for t in TASKS for a in summ[t] if a.startswith("single:")})
    for a in singles:
        cells = []
        for t in TASKS:
            m = "mae" if t in REG else "rocauc"
            cells.append(f"{fmt(summ[t][a][m]['mean'], t)}" if a in summ[t] else "--")
        lines.append("Expert " + a.split(":", 1)[1].replace("_", "\\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_arms.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- table: primary + key secondary tests
    lab = {"stack_fm": "fold-matched", "stack_local": "naive local", "stack": "stack", "best_single": "best single"}

    def row(t, c, label, holm=True):
        d = c["delta"]; lo, hi = c["boot_ci95"]
        return (f"{SHORT[t]} & {label} & {d:+.4g} ({c['rel_change_pct']:+.2f}\\%) & [{lo:+.3g}, {hi:+.3g}] & "
                f"{fmt_p(c['boot_p'])} & {fmt_p(c['boot_p_holm']) if holm else '--'} & {c['folds_improved']}/5 & "
                f"{fmt_p(c['nadeau_bengio_p'])} \\\\")
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Task & Comparison (A vs B) & $\Delta$ = A$-$B (rel.) & 95\% CI & $p_\mathrm{boot}$ & $p_\mathrm{Holm}$ & folds A better & $p_\mathrm{NB}$ \\",
             r"\midrule"]
    keep = [("stack_fm", "stack"), ("stack_fm", "stack_local"), ("stack", "best_single"), ("stack_local", "best_single")]
    for t in TASKS:
        lines.append(row(t, stats[t]["primary"], "naive local vs stack (primary)"))
        for c in stats[t]["secondary"]:
            if (c["arm"], c["comparator"]) in keep:
                lines.append(row(t, c, f"{lab[c['arm']]} vs {lab[c['comparator']]}", holm=False))
        if t != TASKS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- numbers for the text
    n_improved = 0
    for t in TASKS:
        p = stats[t]["primary"]; k = MACRO[t]
        good = (p["delta"] < 0) if t in REG else (p["delta"] > 0)
        n_improved += int(good and p["boot_p_holm"] < 0.05)
        M(f"Delta{k}", f"{p['delta']:+.4g}"); M(f"Rel{k}", f"{p['rel_change_pct']:+.2f}")
        M(f"PBoot{k}", fmt_p(p["boot_p"])); M(f"PHolm{k}", fmt_p(p["boot_p_holm"])); M(f"Folds{k}", f"{p['folds_improved']}")
        m = "mae" if t in REG else "rocauc"
        for a in ("stack", "stack_local", "stack_fm", "best_single", "mean_ensemble", "single_local"):
            M(f"{k}{macro_name(a)}", fmt(summ[t][a][m]["mean"], t))
            M(f"{k}{macro_name(a)}Std", fmt(summ[t][a][m]["std"], t))
            if t not in REG and "balanced_accuracy_innerthr" in summ[t][a]:
                M(f"{k}{macro_name(a)}BA", f"{summ[t][a]['balanced_accuracy_innerthr']['mean']:.4f}")
        for c in stats[t]["secondary"]:
            nm = f"{k}{macro_name(c['arm'])}Vs{macro_name(c['comparator'])}"
            M(nm + "Rel", f"{c['rel_change_pct']:+.2f}"); M(nm + "P", fmt_p(c["boot_p"]))
            M(nm + "Folds", f"{c['folds_improved']}")
        sels = stats[t]["selections"]
        M(f"{k}NaiveOn", str(sum(1 for s in sels if s["sel_stack_local"]["alpha"] > 0)))
        M(f"{k}FmOn", str(sum(1 for s in sels if (s.get("sel_stack_fm") or {}).get("alpha", 0) > 0)))
        M(f"{k}PermNaiveOn", str(sum(1 for s in sels if (s.get("perm_control_naive") or {}).get("alpha", 0) > 0)))
        M(f"{k}PermFmOn", str(sum(1 for s in sels if (s.get("perm_control_fm") or {}).get("alpha", 0) > 0)))
        f0 = per[(per.task == t) & (per.fold == 0)].set_index("arm")[m]
        M(f"{k}FoldZeroStack", fmt(f0["stack"], t)); M(f"{k}FoldZeroLocal", fmt(f0["stack_local"], t))
        M(f"{k}FoldZeroFm", fmt(f0["stack_fm"], t)); M(f"{k}FoldZeroBest", fmt(f0["best_single"], t))
    M("NTasksImproved", str(n_improved))

    # ---------------- selections table
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Task & naive corr. on & fold-matched corr. on & perm.\ control on (naive) & perm.\ control on (f.-m.) \\", r"\midrule"]
    for t in TASKS:
        sels = stats[t]["selections"]
        c = [sum(1 for s in sels if (s.get(key) or {}).get("alpha", 0) > 0)
             for key in ("sel_stack_local", "sel_stack_fm", "perm_control_naive", "perm_control_fm")]
        lines.append(f"{SHORT[t]} & " + " & ".join(f"{x}/5" for x in c) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_selections.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- stack weights table (mean over folds)
    experts = sorted({e for t in TASKS for s in stats[t]["selections"] for e in s["stack_weights"]})
    lines = [r"\begin{tabular}{l" + "c" * len(TASKS) + "}", r"\toprule", "Expert & " + " & ".join(SHORT[t] for t in TASKS) + r" \\", r"\midrule"]
    for e in experts:
        cells = []
        for t in TASKS:
            ws = [s["stack_weights"].get(e) for s in stats[t]["selections"]]
            cells.append(f"{np.mean(ws):.2f}" if all(w is not None for w in ws) else "--")
        lines.append(e.replace("_", "\\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_weights.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- leaderboard table
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Task & Metric & Ours: stack + naive corr. & Best board entry & Rank \\", r"\midrule"]
    for t in TASKS:
        ent = lb[t]["entries"]
        if t in REG:
            o = summ[t]["stack_local"]["mae"]; val, sd = o["mean"], o["std"]
            rank = 1 + sum(e[1] < val for e in ent)
            best = min(ent, key=lambda e: e[1])
            lines.append(f"{SHORT[t]} & MAE & {fmt(val, t)} $\\pm$ {fmt(sd, t)} & {best[0]}: {fmt(best[1], t)} $\\pm$ {fmt(best[2], t)} & {rank} of {len(ent) + 1} \\\\")
        else:
            o = summ[t]["stack_local"]["balanced_accuracy_innerthr"]; val, sd = o["mean"], o["std"]
            rank = 1 + sum(e[4] > val for e in ent)
            best = max(ent, key=lambda e: e[4])
            lines.append(f"{SHORT[t]} & balanced acc. & {val:.4f} $\\pm$ {sd:.4f} & {best[0]}: {best[4]:.4f} & {rank} of {len(ent) + 1} \\\\")
            M(f"{MACRO[t]}AUCprob", f"{summ[t]['stack_local']['rocauc']['mean']:.4f}")
        M(f"{MACRO[t]}Rank", f"{rank}"); M(f"{MACRO[t]}NEntries", f"{len(ent) + 1}")
        M(f"{MACRO[t]}BoardBest", f"{best[0]}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_leaderboard.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- calibration table (classification)
    lines = [r"\begin{tabular}{llcccc}", r"\toprule", r"Task & Arm & Brier & log loss & ECE & bal.\ acc. \\", r"\midrule"]
    for t in TASKS:
        if t in REG:
            continue
        for a in ("best_single", "stack", "stack_local", "stack_fm"):
            s = summ[t][a]
            ba = s.get("balanced_accuracy_innerthr", {}).get("mean", float("nan"))
            lines.append(f"{SHORT[t]} & {ARM_SHORT[a]} & {s['brier']['mean']:.4f} & {s['logloss']['mean']:.4f} & "
                         f"{s['ece']['mean']:.4f} & {ba:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_calibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- timing
    tf = RES / "timing.jsonl"
    if tf.exists():
        tm = pd.read_json(tf, lines=True)
        tm = tm[tm.returncode == 0]
        for t in TASKS:
            sub = tm[tm.task == t]
            if len(sub):
                M(f"{MACRO[t]}WallMin", f"{sub.wall_s.mean() / 60:.0f}")
        M("TotalCPUhWall", f"{tm.wall_s.sum() / 3600:.1f}")
    (GEN / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5})
    comp = (("stack_local", "naive local corr."), ("stack_fm", "fold-matched corr."), ("best_single", "best single"))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, t in zip(axes, TASKS):
        m = "mae" if t in REG else "rocauc"
        sub = per[per.task == t].pivot(index="fold", columns="arm", values=m)
        for j, (a, l) in enumerate(comp):
            rel = 100 * (sub[a] - sub["stack"]) / sub["stack"]
            ax.plot(sub.index + (j - 1) * 0.15, rel, "o", ms=4, color=COL[j], label=l, mec="#fcfcfb", mew=0.6)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(SHORT[t].replace("\\", ""), fontsize=8)
        ax.set_xlabel("official fold"); ax.set_xticks(range(5))
        ax.set_ylabel(("MAE" if t in REG else "ROC-AUC") + " change vs stack (%)", fontsize=7)
    axes[0].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(FIG / "fig_perfold.pdf"); plt.close(fig)
    sf = RES / "support_strata.csv"
    if sf.exists():
        st = pd.read_csv(sf)
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
        for ax, t in zip(axes, TASKS):
            sub = st[st.task == t].groupby(["arm", "bin"])["value"].mean().unstack(0)
            for j, (a, l) in enumerate(comp):
                if a in sub:
                    ax.plot(sub.index, 100 * (sub[a] - sub["stack"]) / sub["stack"], "o-", ms=4, lw=1.5, color=COL[j],
                            label=l, mec="#fcfcfb", mew=0.6)
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_title(SHORT[t].replace("\\", ""), fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\n(near)", "Q2", "Q3", "Q4\n(far)"])
            ax.set_ylabel(("MAE" if t in REG else "Brier score") + " change vs stack (%)", fontsize=7)
        axes[0].legend(frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(FIG / "fig_support.pdf"); plt.close(fig)
    print("assets written:", sorted(p.name for p in GEN.iterdir()), sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
