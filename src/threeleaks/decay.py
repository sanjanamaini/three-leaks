"""Exploratory, added AFTER the main results (see DEVIATIONS.md): is the active edge fading?

    python -m threeleaks.decay

The main results show positive average alpha, but alphas estimated on 2013-2019 were followed
by much smaller ones in 2020-2025. This module asks, without changing any registered analysis:
1. Rolling 36-month four-factor alpha of the equal-weighted average active fund (regular and direct).
2. Alpha of that average fund in fixed sub-periods.
3. Out-of-sample selection strategies split into 2014-2019 and 2020-2025.
4. Commission gap vs direct-plan alpha within categories (is the overall correlation a category effect?).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import FACTOR_END, FIGURES, INTERIM, NW_LAGS, TABLES
from .models import design, fit
from .skill import DIVERSIFIED_EXCLUDE, fund_series

PERIODS = [("2006-05", "2010-12"), ("2011-01", "2015-12"), ("2016-01", "2020-12"), ("2021-01", "2025-12")]


def ew_excess(schemes, returns, factors, plan) -> pd.Series:
    wide, meta = fund_series(schemes, returns, plan)
    keep = [c for c in wide.columns if meta.loc[c, "category"] not in DIVERSIFIED_EXCLUDE]
    ew = wide[keep].mean(axis=1)
    f = factors.set_index("month")
    ew = ew[ew.index <= pd.Period(FACTOR_END, "M")]
    return (ew - f["rf"].reindex(ew.index)).dropna(), f


def alpha_of(y: pd.Series, f: pd.DataFrame) -> dict:
    ff = f.reindex(y.index)
    ok = y.notna() & ff["mkt_rf"].notna()
    r = fit(y[ok].to_numpy(), design(ff[ok]), NW_LAGS)
    return {"alpha_ann": r["alpha"] * 12, "t_nw": r["t_nw"], "months": int(ok.sum())}


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet")
    numbers = {}

    rows = []
    roll = {}
    for plan in ("regular", "direct"):
        y, f = ew_excess(schemes, returns, factors, plan)
        for a, b in PERIODS:
            sl = y[(y.index >= pd.Period(a, "M")) & (y.index <= pd.Period(b, "M"))]
            if len(sl) >= 24:
                rows.append({"plan": plan, "period": "%s to %s" % (a, b), **alpha_of(sl, f)})
        vals = {}
        for i in range(36, len(y) + 1):
            w = y.iloc[i - 36:i]
            vals[w.index[-1]] = alpha_of(w, f)["alpha_ann"]
        roll[plan] = pd.Series(vals)
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "decay_periods.csv", index=False)
    for _, r in per.iterrows():
        k = "decay_%s_%s" % (r["plan"][:3], r["period"][:4])
        numbers[k + "_alpha"] = float(r["alpha_ann"])
        numbers[k + "_t"] = float(r["t_nw"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for plan, colr in (("regular", "#1F3A5F"), ("direct", "#C9A227")):
        s = roll[plan]
        ax.plot(s.index.to_timestamp(), s * 100, color=colr, label="%s plans" % plan)
    ax.axhline(0, color="#999", lw=1)
    ax.set_ylabel("Rolling 36-month four-factor alpha,\n% a year (average active fund)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "decay_rolling_alpha.png", dpi=200)
    plt.close(fig)
    for plan in ("regular", "direct"):
        numbers["decay_%s_rolling_last" % plan[:3]] = float(roll[plan].iloc[-1])
        numbers["decay_%s_rolling_max" % plan[:3]] = float(roll[plan].max())
        numbers["decay_%s_rolling_max_month" % plan[:3]] = str(roll[plan].idxmax())

    # Selection strategies by sub-period (direct plans).
    sel = pd.read_csv(TABLES / "selection_returns_direct.csv", index_col=0)
    sel.index = pd.PeriodIndex(sel.index, freq="M")
    f = factors.set_index("month")
    srows = []
    for a, b in (("2014-01", "2019-12"), ("2020-01", "2025-12")):
        part = sel[(sel.index >= pd.Period(a, "M")) & (sel.index <= pd.Period(b, "M"))]
        for col in part.columns:
            y = part[col] - f["rf"].reindex(part.index)
            res = alpha_of(y, f)
            row = {"period": "%s to %s" % (a[:4], b[:4]), "strategy": col, **res}
            if col not in ("all_funds", "nifty50_index"):
                d = alpha_of(part[col] - part["all_funds"], f)
                row.update({"alpha_vs_all": d["alpha_ann"], "t_vs_all": d["t_nw"]})
            srows.append(row)
    sp = pd.DataFrame(srows)
    sp.to_csv(TABLES / "decay_selection_periods.csv", index=False)
    for _, r in sp.iterrows():
        k = "decay_sel_%s_%s" % (r["strategy"], r["period"][:4])
        numbers[k + "_alpha"] = float(r["alpha_ann"])
        numbers[k + "_t"] = float(r["t_nw"])
        if pd.notna(r.get("alpha_vs_all")):
            numbers[k + "_vsall"] = float(r["alpha_vs_all"])
            numbers[k + "_vsall_t"] = float(r["t_vs_all"])

    # Commission vs alpha within categories.
    pairs = pd.read_csv(TABLES / "commission_pairs.csv")
    sk = pd.read_csv(TABLES / "skill_funds_direct.csv", dtype={"code": str})
    both = pairs.merge(sk[["fund_id", "alpha_ann"]], on="fund_id").dropna()
    both = both[~both["category"].isin(DIVERSIFIED_EXCLUDE)]
    both["gap_dm"] = both["gap_ann"] - both.groupby("category")["gap_ann"].transform("mean")
    both["alpha_dm"] = both["alpha_ann"] - both.groupby("category")["alpha_ann"].transform("mean")
    rho_all, p_all = stats.spearmanr(both["gap_ann"], both["alpha_ann"])
    rho_w, p_w = stats.spearmanr(both["gap_dm"], both["alpha_dm"])
    numbers.update({"decay_gap_alpha_rho_diversified": float(rho_all), "decay_gap_alpha_p_diversified": float(p_all),
                    "decay_gap_alpha_rho_within": float(rho_w), "decay_gap_alpha_p_within": float(p_w),
                    "decay_gap_alpha_n": int(len(both))})
    # 5. Added after results (DEVIATIONS.md): which years carry the direct plans' alpha? Betas come from
    # the whole direct window; each calendar year's abnormal return is its mean (alpha + residual) x 12.
    y, f = ew_excess(schemes, returns, factors, "direct")
    ff = f.reindex(y.index)
    ok = y.notna() & ff["mkt_rf"].notna()
    full = fit(y[ok].to_numpy(), design(ff[ok]), NW_LAGS)
    abnormal = pd.Series(full["resid"] + full["alpha"], index=y[ok].index)
    years = abnormal.groupby(abnormal.index.year).agg(["count", "mean"])
    years["abnormal_ann"] = years["mean"] * 12
    years.to_csv(TABLES / "decay_years_direct.csv")
    for yr, row in years.iterrows():
        numbers["decay_year_%d_dir" % yr] = float(row["abnormal_ann"])
    ex2020 = alpha_of(y[ok][y[ok].index.year != 2020], f)
    allm = alpha_of(y[ok], f)
    numbers.update({"decay_dir_alpha_full": allm["alpha_ann"], "decay_dir_alpha_full_t": allm["t_nw"],
                    "decay_dir_alpha_ex2020": ex2020["alpha_ann"], "decay_dir_alpha_ex2020_t": ex2020["t_nw"]})

    # 6. Added after results: in 2020-2025, did funds with three or more years of direct-plan history
    # before 2020 earn less alpha than newer funds?
    wide, meta = fund_series(schemes, returns, "direct")
    keep = [c for c in wide.columns if meta.loc[c, "category"] not in DIVERSIFIED_EXCLUDE]
    pre = wide.loc[wide.index < pd.Period("2020-01", "M"), keep].notna().sum()
    win = wide.loc[(wide.index >= pd.Period("2020-01", "M")) & (wide.index <= pd.Period(FACTOR_END, "M")), keep]
    fm = factors.set_index("month")
    rows = []
    for c in keep:
        s = win[c].dropna()
        if len(s) < 36:
            continue
        a = alpha_of(s - fm["rf"].reindex(s.index), fm)
        rows.append({"code": c, "old": bool(pre[c] >= 36), "alpha_ann": a["alpha_ann"], "months": a["months"]})
    on = pd.DataFrame(rows)
    on.to_csv(TABLES / "decay_old_new.csv", index=False)
    old, new = on[on["old"]], on[~on["old"]]
    tt = stats.ttest_ind(old["alpha_ann"], new["alpha_ann"], equal_var=False)
    numbers.update({"decay_old_funds": int(len(old)), "decay_new_funds": int(len(new)),
                    "decay_old_alpha": float(old["alpha_ann"].mean()), "decay_new_alpha": float(new["alpha_ann"].mean()),
                    "decay_oldnew_p": float(tt.pvalue)})
    print(years.to_string())
    print({k: numbers[k] for k in ("decay_dir_alpha_full", "decay_dir_alpha_ex2020", "decay_old_alpha", "decay_new_alpha",
                                   "decay_old_funds", "decay_new_funds", "decay_oldnew_p")})

    (TABLES / "decay_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(per.to_string())
    print(sp.to_string())
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
