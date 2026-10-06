"""Write the report's tables (report/tables/*.tex) straight from results/tables/*.csv.

    python -m threeleaks.report_tables
"""
from __future__ import annotations

import pandas as pd

from .config import REPORT, TABLES


def pct(x, d=2):
    return "" if pd.isna(x) else ("%.*f" % (d, x * 100))


def num(x, d=2):
    return "" if pd.isna(x) else ("%.*f" % (d, x))


def pval(x):
    return "" if pd.isna(x) else ("<0.001" if x < 0.001 else "%.3f" % x)


def tex_escape(s: str) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", " ")


def tabular(df: pd.DataFrame, align: str, header: list) -> str:
    lines = ["\\begin{tabular}{%s}" % align, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v) for v in r.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = REPORT / "tables"
    out.mkdir(parents=True, exist_ok=True)

    c = pd.read_csv(TABLES / "commission_by_category.csv")
    c = c.sort_values("median", ascending=False)
    t = pd.DataFrame({"Category": c["category"].map(tex_escape), "Funds": c["count"].astype(int),
                      "Mean": c["mean"].map(pct), "Median": c["median"].map(pct)})
    (out / "commission_by_category.tex").write_text(tabular(t, "lrrr", ["Category", "Twin pairs", "Mean gap, \\%/yr", "Median gap, \\%/yr"]))

    e = pd.read_csv(TABLES / "commission_events.csv")
    t = pd.DataFrame({"Event": e["event"], "Kind": e["kind"], "Pairs": e["pairs"].astype(int),
                      "Before": e["gap_before"].map(pct), "After": e["gap_after"].map(pct),
                      "Change": e["change_mean"].map(pct), "p": e["p_t"].map(pval)})
    (out / "commission_events.tex").write_text(tabular(t, "llrrrrr", ["Event date", "Role", "Pairs", "Gap before", "Gap after", "Change", "p (t-test)"]))

    f = pd.read_csv(TABLES / "skill_fdr_summary.csv")
    t = pd.DataFrame({"Sample": f["label"].map(tex_escape), "Funds": f["funds"].astype(int),
                      "pi0": f["pi0"].map(pct), "pi+": f["pi_plus"].map(pct),
                      "pi+ CI": ["[" + pct(a, 1) + ", " + pct(b, 1) + "]" for a, b in zip(f["pi_plus_lo"], f["pi_plus_hi"])],
                      "pi-": f["pi_minus"].map(pct),
                      "pi- CI": ["[" + pct(a, 1) + ", " + pct(b, 1) + "]" for a, b in zip(f["pi_minus_lo"], f["pi_minus_hi"])]})
    (out / "skill_fdr.tex").write_text(tabular(t, "lrrrrrr", ["Sample", "Funds", "$\\hat\\pi_0$", "$\\hat\\pi^+$", "95\\% CI", "$\\hat\\pi^-$", "95\\% CI"]))

    b = pd.read_csv(TABLES / "skill_ff_bootstrap_regular.csv")
    t = pd.DataFrame({"Pct": b["percentile"].astype(int), "Actual": b["t_actual"].map(num),
                      "Sim": b["t_sim_mean"].map(num), "Below": b["share_sims_below_actual"].map(lambda x: pct(x, 0))})
    (out / "skill_ff.tex").write_text(tabular(t, "rrrr", ["Percentile", "Actual $t(\\alpha)$", "Zero-alpha world", "\\% of simulations below actual"]))

    p = pd.read_csv(TABLES / "skill_persistence_regular.csv")
    t = pd.DataFrame({"Q": p["portfolio"].map(lambda s: "Top minus bottom" if s == "Q5-Q1" else "Quintile " + s),
                      "a": p["alpha_ann"].map(pct), "t": p["t_nw"].map(num), "m": p["months"].astype(int)})
    (out / "skill_persistence.tex").write_text(tabular(t, "lrrr", ["Portfolio (sorted on prior 36-month $t(\\alpha)$)", "Next-year alpha, \\%/yr", "$t$ (NW)", "Months"]))

    k = pd.read_csv(TABLES / "style_factor_calibration.csv")
    names = {"nifty50": "Nifty 50", "sensex": "Sensex", "next50": "Nifty Next 50", "midcap150": "Nifty Midcap 150",
             "smallcap250": "Nifty Smallcap 250", "nifty500": "Nifty 500"}
    t = pd.DataFrame({"Index": k["index"].map(names), "Period": k["start"] + " to " + k["end"],
                      "a": k["alpha_ann"].map(pct), "t": k["t_nw"].map(num), "r2": k["r2"].map(num)})
    (out / "calibration.tex").write_text(tabular(t, "llrrr", ["Index funds", "Months used", "Four-factor alpha, \\%/yr", "$t$ (NW)", "$R^2$"]))

    tm = pd.read_csv(TABLES / "timing_results.csv")
    t = pd.DataFrame({"Unit": tm["label"].map(tex_escape), "TWR": tm["twr_ann"].map(pct), "MWR": tm["mwr_ann"].map(pct),
                      "Gap": tm["gap_ann"].map(pct), "p1": tm["p_worse_than_random"].map(pval),
                      "p2": tm["p_worse_than_random_within_year"].map(pval)})
    (out / "timing.tex").write_text(tabular(t, "lrrrrr", ["Investors in", "Funds (TWR)", "Investors (MWR)", "Gap", "$p$ vs random", "$p$ within-year"]))

    s = pd.read_csv(TABLES / "selection_performance.csv")
    labels = {"low_r2": "Low $R^2$ quintile", "high_alpha": "High prior alpha quintile", "ridge": "Ridge regression",
              "gbt": "Gradient-boosted trees", "all_funds": "All eligible funds", "nifty50_index": "Nifty 50 index funds"}
    s = s[s["plan"] == "direct"]
    t = pd.DataFrame({"S": s["strategy"].map(labels), "ex": s["mean_excess_ann"].map(pct), "a": s["alpha_ann"].map(pct),
                      "t": s["t_nw"].map(num), "ph": s["p_holm"].map(pval),
                      "va": s["alpha_vs_all_ann"].map(pct), "vt": s["t_vs_all"].map(num)})
    (out / "selection.tex").write_text(tabular(t, "lrrrrrr", ["Strategy (direct plans)", "Excess return", "Alpha", "$t$", "Holm $p$", "Alpha vs all funds", "$t$"]))

    d = pd.read_csv(TABLES / "synthesis_decomposition.csv")
    t = pd.DataFrame({"L": d["leak"].map(tex_escape), "p": d["pct_per_year"].map(pct),
                      "r": d["rs_crore_2025"].map(lambda v: "{:,.0f}".format(v))})
    (out / "synthesis.tex").write_text(tabular(t, "lrr", ["Leak (2019-05 to 2025-12)", "Points a year", "Rs crore a year at 2025 AUM"]))
    print("report tables written")


if __name__ == "__main__":
    main()
