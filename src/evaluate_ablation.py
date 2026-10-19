"""Score the post-hoc ablation variants ONCE (ANALYSIS_PLAN.md Section 12). Compares each variant's final
arm (stack_local) with the Amendment-2 final arm and with the leader, using the same paired bootstrap on
the board metric as src.evaluate2. Writes results/ablation_summary.json and manuscript macros/table."""
from __future__ import annotations
import json
import numpy as np
from .data import ROOT, TASKS
from . import evaluate2 as ev2

VARIANTS = {"nodft": TASKS, "nohurdle": ["matbench_expt_gap"]}
SHORT = {"matbench_steels": "steels", "matbench_expt_gap": "expt\\_gap", "matbench_expt_is_metal": "expt\\_is\\_metal",
         "matbench_glass": "glass"}
MACRO = {"matbench_steels": "Steels", "matbench_expt_gap": "Gap", "matbench_expt_is_metal": "Metal", "matbench_glass": "Glass"}
VM = {"nodft": "NoDft", "nohurdle": "NoHurdle"}


def f(x, t):
    return f"{x:.2f}" if t == "matbench_steels" else f"{x:.4f}"


def main():
    rng = np.random.default_rng(12345)
    out, macros, rows = {}, [], []
    for v, tasks in VARIANTS.items():
        for t in tasks:
            av = ev2.load(ROOT / "results" / f"ablation_{v}", t)
            a2 = ev2.load(ev2.A2, t)
            ys = [x[1] for x in a2]
            assert all(np.array_equal(x[1], y) for x, y in zip(av, ys))
            pv = [x[2]["stack_local"] for x in av]; tv = [x[3].get("thresholds", {}).get("stack_local") for x in av]
            p2 = [x[2]["stack_local"] for x in a2]; t2 = [x[3].get("thresholds", {}).get("stack_local") for x in a2]
            lp = ev2.leader_preds(t, [x[0] for x in a2])
            mv = float(np.mean([ev2.board_metric(t, y, q, th) for y, q, th in zip(ys, pv, tv)]))
            m2 = float(np.mean([ev2.board_metric(t, y, q, th) for y, q, th in zip(ys, p2, t2)]))
            ml = float(np.mean([ev2.board_metric(t, y, q) for y, q in zip(ys, lp)]))
            d, lo, hi, p = ev2.paired_boot(t, ys, pv, p2, tv, t2, rng)
            dl, lol, hil, pl = ev2.paired_boot(t, ys, pv, lp, tv, [None] * 5, rng)
            folds = [float(ev2.board_metric(t, y, q, th) - ev2.board_metric(t, y, q2, th2))
                     for y, q, th, q2, th2 in zip(ys, pv, tv, p2, t2)]
            out[f"{v}:{t}"] = {"variant": v, "task": t, "variant_metric": mv, "a2_metric": m2, "leader_metric": ml,
                               "delta_vs_a2": d, "ci_vs_a2": [lo, hi], "p_vs_a2": p, "per_fold_delta_vs_a2": folds,
                               "delta_vs_leader": dl, "ci_vs_leader": [lol, hil], "p_vs_leader": pl,
                               "weights": [x[3]["stack_weights"] for x in av]}
            k = VM[v] + MACRO[t]
            macros += [f"\\newcommand{{\\Abl{k}}}{{{f(mv, t)}}}", f"\\newcommand{{\\Abl{k}Delta}}{{{d:+.4g}}}",
                       f"\\newcommand{{\\Abl{k}Rel}}{{{100 * d / m2:+.1f}}}",
                       f"\\newcommand{{\\Abl{k}P}}{{{'$<$0.001' if p < 1e-3 else f'{p:.3f}'}}}",
                       f"\\newcommand{{\\Abl{k}VsLeader}}{{{dl:+.4g}}}"]
            pr = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
            prl = "$<$0.001" if pl < 1e-3 else f"{pl:.3f}"
            name = {"nodft": "without DFT block", "nohurdle": "without hurdle experts"}[v]
            rows.append(f"{SHORT[t]} & {name} & {f(m2, t)} & {f(mv, t)} & {d:+.4g} [{lo:+.3g}, {hi:+.3g}] & {pr} & "
                        f"{sum((x > 0) if t in ev2.REG else (x < 0) for x in folds)}/5 & {f(ml, t)} ({prl}) \\\\")
    json.dump(out, open(ROOT / "results" / "ablation_summary.json", "w"), indent=1)
    gen = ROOT / "manuscript" / "generated"
    (gen / "numbers_ablation.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    tab = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Task & Variant & A2 final & Variant & $\Delta$ (variant $-$ A2) [95\% CI] & $p_\mathrm{boot}$ & folds worse & Leader ($p$ variant vs.\ leader) \\",
           r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (gen / "table_ablation.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, f"variant={v['variant_metric']:.4f} A2={v['a2_metric']:.4f} delta={v['delta_vs_a2']:+.4f} "
                 f"CI={[round(c, 4) for c in v['ci_vs_a2']]} p={v['p_vs_a2']:.4f} | vs leader {v['delta_vs_leader']:+.4f} p={v['p_vs_leader']:.4f}")


if __name__ == "__main__":
    main()
