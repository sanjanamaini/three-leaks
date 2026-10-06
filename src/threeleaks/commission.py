"""Stage 3. The commission leak: what the regular plan costs relative to its direct twin.

    python -m threeleaks.commission

Twin plans share one portfolio and one manager; the only difference is the distributor
commission inside the regular plan's expense ratio. So the direct-minus-regular return gap is
a clean, month-by-month price of distribution. Tests follow PREREGISTRATION.md (RQ1).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import FIGURES, INTERIM, PLACEBO_EVENTS, TABLES, TER_EVENT, UPFRONT_BAN_EVENT


def annualise_log(g: pd.Series) -> float:
    """Annualised gap from monthly log-return differences."""
    return float(np.exp(12 * g.mean()) - 1)


def pair_panel(twins: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.set_index(["code", "month"])["ret"]
    rows = []
    for t in twins.itertuples():
        d = r.xs(t.direct_code, level="code") if t.direct_code in r.index.get_level_values(0) else None
        g = r.xs(t.regular_code, level="code") if t.regular_code in r.index.get_level_values(0) else None
        if d is None or g is None:
            continue
        both = pd.concat({"rd": d, "rr": g}, axis=1).dropna()
        both["gap_log"] = np.log1p(both["rd"]) - np.log1p(both["rr"])
        both["fund_id"], both["category"] = t.fund_id, t.category
        rows.append(both.reset_index())
    return pd.concat(rows, ignore_index=True)


def window_gap(p: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Annualised gap per pair over [start, end]; only pairs present in every month."""
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    need = (e - s).n + 1
    w = p[(p["month"] >= s) & (p["month"] <= e)]
    n = w.groupby("fund_id")["gap_log"].count()
    full = n[n == need].index
    return w[w["fund_id"].isin(full)].groupby("fund_id")["gap_log"].apply(annualise_log)


def event_test(p: pd.DataFrame, event: str, before: tuple, after: tuple) -> dict:
    b = window_gap(p, *before)
    a = window_gap(p, *after)
    both = pd.concat({"before": b, "after": a}, axis=1).dropna()
    diff = both["after"] - both["before"]
    t = stats.ttest_1samp(diff, 0.0)
    w = stats.wilcoxon(diff) if len(diff) > 10 else None
    return {"event": event, "pairs": int(len(both)), "gap_before": float(both["before"].mean()),
            "gap_after": float(both["after"].mean()), "change_mean": float(diff.mean()),
            "change_median": float(diff.median()), "t": float(t.statistic), "p_t": float(t.pvalue),
            "p_wilcoxon": float(w.pvalue) if w else None,
            "share_pairs_narrowed": float((diff < 0).mean())}


def windows_for(event: str) -> tuple:
    e = pd.Period(event, "M")
    if event == UPFRONT_BAN_EVENT:   # event inside October 2018: drop the event month
        return (str(e - 12), str(e - 1)), (str(e + 1), str(e + 12))
    return (str(e - 12), str(e - 1)), (str(e), str(e + 11))


def main() -> None:
    twins = pd.read_parquet(INTERIM / "twins.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    p = pair_panel(twins, returns)
    p.to_parquet(INTERIM / "twin_panel.parquet", index=False)

    full = p.groupby(["fund_id", "category"])["gap_log"].agg(["count", "mean"]).reset_index()
    full["gap_ann"] = np.exp(12 * full["mean"]) - 1
    full = full[full["count"] >= 12]
    full.to_csv(TABLES / "commission_pairs.csv", index=False)

    wil = stats.wilcoxon(full["gap_ann"])
    by_cat = full.groupby("category")["gap_ann"].agg(["count", "mean", "median"]).sort_values("median")
    by_cat.to_csv(TABLES / "commission_by_category.csv")
    p["year"] = p["month"].dt.year
    py = p.groupby(["fund_id", "year"])["gap_log"].agg(["count", "mean"])
    py = py[py["count"] >= 6]
    py["gap_ann"] = np.exp(12 * py["mean"]) - 1
    by_year = py.groupby("year")["gap_ann"].agg(["count", "mean", "median"])
    by_year.to_csv(TABLES / "commission_by_year.csv")

    events = []
    for ev, kind in [(TER_EVENT, "primary"), (UPFRONT_BAN_EVENT, "secondary")] + [(e, "placebo") for e in PLACEBO_EVENTS]:
        before, after = windows_for(ev)
        res = event_test(p, ev, before, after)
        res["kind"] = kind
        events.append(res)
    ev = pd.DataFrame(events)
    ev.to_csv(TABLES / "commission_events.csv", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(full["gap_ann"] * 100, bins=40, color="#1F3A5F")
    ax.axvline(0, color="#999", lw=1)
    ax.set_xlabel("Direct minus regular return, % a year (whole twin history)")
    ax.set_ylabel("Funds")
    fig.tight_layout()
    fig.savefig(FIGURES / "commission_hist.png", dpi=200)
    plt.close(fig)

    m = p.groupby("month")["gap_log"].median().rolling(12).mean()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(m.index.to_timestamp(), (np.exp(12 * m) - 1) * 100, color="#1F3A5F")
    y0, y1 = ax.get_ylim()
    for e, lab, ha, frac in [(UPFRONT_BAN_EVENT, "upfront commissions banned ", "right", 0.97),
                             (TER_EVENT, " TER cut (April 2019)", "left", 0.87)]:
        ax.axvline(pd.Timestamp(e), color="#B33", lw=1, ls="--")
        ax.text(pd.Timestamp(e), y0 + (y1 - y0) * frac, lab, color="#B33", fontsize=7, va="top", ha=ha)
    ax.set_ylabel("Median twin gap, % a year (12-month avg)")
    fig.tight_layout()
    fig.savefig(FIGURES / "commission_over_time.png", dpi=200)
    plt.close(fig)

    prim = ev[ev["kind"] == "primary"].iloc[0]
    numbers = {
        "commission_pairs": int(len(full)),
        "commission_share_positive": float((full["gap_ann"] > 0).mean()),
        "commission_gap_mean": float(full["gap_ann"].mean()),
        "commission_gap_median": float(full["gap_ann"].median()),
        "commission_gap_p25": float(full["gap_ann"].quantile(0.25)),
        "commission_gap_p75": float(full["gap_ann"].quantile(0.75)),
        "commission_wilcoxon_p": float(wil.pvalue),
        "commission_ter_pairs": int(prim["pairs"]),
        "commission_ter_before": float(prim["gap_before"]),
        "commission_ter_after": float(prim["gap_after"]),
        "commission_ter_change": float(prim["change_mean"]),
        "commission_ter_p": float(prim["p_t"]),
        "commission_ter_p_wilcoxon": float(prim["p_wilcoxon"]) if prim["p_wilcoxon"] is not None else None,
        "commission_ter_share_narrowed": float(prim["share_pairs_narrowed"]),
    }
    for _, r in ev[ev["kind"] == "placebo"].iterrows():
        k = "commission_placebo_" + r["event"][:4]
        numbers[k + "_change"] = float(r["change_mean"])
        numbers[k + "_p"] = float(r["p_t"])
    sec = ev[ev["kind"] == "secondary"].iloc[0]
    numbers["commission_upfront_change"] = float(sec["change_mean"])
    numbers["commission_upfront_p"] = float(sec["p_t"])
    for yr, row in by_year.iterrows():  # median twin gap by calendar year (years with partial data are labelled in the report)
        numbers["commission_year_%d_median" % yr] = float(row["median"])
    (TABLES / "commission_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
    print(by_cat.to_string())
    print(by_year.to_string())
    print(ev.to_string())


if __name__ == "__main__":
    main()
