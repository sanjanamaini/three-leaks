"""Stage 5. The timing leak: how much less investors earn than their funds, because of when they
add and withdraw money.

    python -m threeleaks.timing

Unit: all open-ended active equity schemes together (AMFI's equity subtotal), monthly, from the
first month of AMFI's consistent category layout (2019-04). The time-weighted return (what the
funds earned) comes from AUM net of flows; the money-weighted return (what investors earned)
is the internal rate of return of the same cash flows. The test asks whether investors' actual
timing is worse than the same flows placed in random months (PREREGISTRATION.md, RQ3).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .config import FIGURES, INTERIM, N_PERM, SEED, TABLES
from .panels import NON_ACTIVE

START = pd.Period("2019-04", "M")   # first month of the consistent category layout
STABLE_CATEGORIES = ["Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Large & Mid Cap Fund", "ELSS"]


def equity_series(flows: pd.DataFrame) -> pd.DataFrame:
    sub = flows[(flows["category"] == "_subtotal")
                & flows["section"].str.contains("equity", case=False, na=False)
                & ~flows["section"].str.contains("hybrid|index|etf|other|solution", case=False, na=False)]
    s = sub.groupby("month")[["net_inflow", "aum", "inflow", "redemption"]].first().sort_index()
    return s


def category_series(flows: pd.DataFrame, name: str) -> pd.DataFrame:
    c = flows[flows["category"].str.lower().str.replace(r"\s+", " ", regex=True).str.startswith(name.lower())]
    c = c[c["section"].str.contains("equity", case=False, na=False)]
    return c.groupby("month")[["net_inflow", "aum"]].first().sort_index()


def implied_returns(s: pd.DataFrame) -> pd.Series:
    a_prev = s["aum"].shift(1)
    return (s["aum"] - a_prev - s["net_inflow"]) / (a_prev + 0.5 * s["net_inflow"])


def mwr(a0: float, flows: np.ndarray, a_end: float) -> float:
    """Monthly IRR: -a0 at t=0, -flow_t at t-0.5 (mid-month), +a_end at T."""
    T = len(flows)
    t = np.arange(1, T + 1) - 0.5

    def npv(x):
        return -a0 - (flows / (1 + x) ** t).sum() + a_end / (1 + x) ** T
    return brentq(npv, -0.5, 0.5)


def simulate_end(a0: float, r: np.ndarray, flows: np.ndarray) -> float:
    """Terminal AUM from a return path and a flow sequence, flows arriving mid-month."""
    a = a0
    for rt, ft in zip(r, flows):
        a = a * (1 + rt) + ft * (1 + 0.5 * rt)
    return a


def analyse(s: pd.DataFrame, rng: np.random.Generator, label: str, n_perm: int) -> dict:
    s = s[s.index >= START].copy()
    r = implied_returns(s).iloc[1:]
    f = s["net_inflow"].iloc[1:].to_numpy()
    a0 = float(s["aum"].iloc[0])
    a_end_model = simulate_end(a0, r.to_numpy(), f)
    T = len(r)
    twr_m = np.prod(1 + r.to_numpy()) ** (1 / T) - 1
    mwr_m = mwr(a0, f, a_end_model)
    rr = r.to_numpy()
    # Pre-registered null: the same flows in random months (any month of the window).
    perm = np.empty(n_perm)
    for k in range(n_perm):
        fp = rng.permutation(f)
        perm[k] = mwr(a0, fp, simulate_end(a0, rr, fp))
    # Added variant (DEVIATIONS.md): shuffle only within each calendar year, which keeps the
    # multi-year rise in flows (SIP growth) and isolates month-to-month timing.
    years = np.array([p.year for p in r.index])
    perm_y = np.empty(n_perm)
    for k in range(n_perm):
        fp = f.copy()
        for yv in np.unique(years):
            ix = np.where(years == yv)[0]
            fp[ix] = rng.permutation(f[ix])
        perm_y[k] = mwr(a0, fp, simulate_end(a0, rr, fp))
    twr_a, mwr_a = (1 + twr_m) ** 12 - 1, (1 + mwr_m) ** 12 - 1
    perm_a = (1 + perm) ** 12 - 1
    perm_ya = (1 + perm_y) ** 12 - 1
    avg_aum = float(s["aum"].iloc[1:].mean())
    # Return chasing: do flows (as % of AUM) follow past returns, and precede future ones?
    fr = pd.Series(f / s["aum"].shift(1).iloc[1:].to_numpy(), index=r.index)
    chase = {}
    import statsmodels.api as sm
    df = pd.DataFrame({"flow": fr, "r_lag1": r.shift(1), "r_lag3": r.shift(1).rolling(3).sum(),
                       "r_lead1": r.shift(-1), "r_lead3": r.shift(-1).rolling(3).sum().shift(-2)}).dropna()
    for name, xcols in (("past", ["r_lag3"]), ("future", ["r_lead3"])):
        m = sm.OLS(df["flow"], sm.add_constant(df[xcols])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        chase[name + "_coef"] = float(m.params[xcols[0]])
        chase[name + "_t"] = float(m.tvalues[xcols[0]])
    return {"label": label, "start": str(r.index[0]), "end": str(r.index[-1]), "months": T,
            "twr_ann": twr_a, "mwr_ann": mwr_a, "gap_ann": twr_a - mwr_a,
            "perm_mwr_mean": float(perm_a.mean()), "perm_gap_mean": float(twr_a - perm_a.mean()),
            "p_worse_than_random": float((perm_a <= mwr_a).mean()),
            "timing_vs_random_ann": float(mwr_a - perm_a.mean()),
            "perm_year_mwr_mean": float(perm_ya.mean()),
            "p_worse_than_random_within_year": float((perm_ya <= mwr_a).mean()),
            "timing_vs_random_within_year_ann": float(mwr_a - perm_ya.mean()),
            "avg_aum_cr": avg_aum, "total_net_inflow_cr": float(f.sum()), **chase}


def nav_check(s: pd.DataFrame, returns: pd.DataFrame, schemes: pd.DataFrame) -> float:
    """Correlation between AUM-implied equity returns and the average NAV return of active equity funds."""
    act = schemes[schemes["is_growth"] & schemes["structure"].str.startswith("open")
                  & ~schemes["category"].isin(NON_ACTIVE | {"Sectoral/Thematic"})]["code"]
    nav_ret = returns[returns["code"].isin(act)].groupby("month")["ret"].mean()
    r = implied_returns(s[s.index >= START]).iloc[1:]
    both = pd.concat({"aum": r, "nav": nav_ret}, axis=1).dropna()
    both.to_csv(TABLES / "timing_return_check.csv")
    return float(both.corr().iloc[0, 1]), float((both["aum"] - both["nav"]).abs().max())


def main() -> None:
    rng = np.random.default_rng(SEED)
    flows = pd.read_parquet(INTERIM / "flows.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    eq = equity_series(flows)
    eq.to_csv(TABLES / "timing_equity_series.csv")
    corr, maxdiff = nav_check(eq, returns, schemes)

    main_res = analyse(eq, rng, "All active equity", N_PERM)
    rows = [main_res]
    for c in STABLE_CATEGORIES:
        cs = category_series(flows, c)
        if len(cs[cs.index >= START]) > 24:
            rows.append(analyse(cs, rng, c, 2000))
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "timing_results.csv", index=False)

    # Figure: cumulative net flows against the equity return index.
    s = eq[eq.index >= START]
    r = implied_returns(s).iloc[1:]
    idx = (1 + r).cumprod()
    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.plot(idx.index.to_timestamp(), idx, color="#1F3A5F", label="fund return index")
    ax1.set_ylabel("Fund return index (start = 1)")
    ax2 = ax1.twinx()
    ax2.bar(s.index[1:].to_timestamp(), s["net_inflow"].iloc[1:] / 1000, width=20, color="#C9A227", alpha=0.6,
            label="net inflow")
    ax2.set_ylabel("Net inflow, Rs thousand crore")
    fig.tight_layout()
    fig.savefig(FIGURES / "timing_flows.png", dpi=200)
    plt.close(fig)

    numbers = {"timing_months": main_res["months"], "timing_start": main_res["start"], "timing_end": main_res["end"],
               "timing_twr": main_res["twr_ann"], "timing_mwr": main_res["mwr_ann"], "timing_gap": main_res["gap_ann"],
               "timing_p_worse_random": main_res["p_worse_than_random"],
               "timing_vs_random": main_res["timing_vs_random_ann"],
               "timing_perm_gap_mean": main_res["perm_gap_mean"],
               "timing_p_worse_random_within_year": main_res["p_worse_than_random_within_year"],
               "timing_vs_random_within_year": main_res["timing_vs_random_within_year_ann"],
               "timing_chase_past_coef": main_res["past_coef"], "timing_chase_past_t": main_res["past_t"],
               "timing_chase_future_coef": main_res["future_coef"], "timing_chase_future_t": main_res["future_t"],
               "timing_avg_aum_cr": main_res["avg_aum_cr"], "timing_total_inflow_cr": main_res["total_net_inflow_cr"],
               "timing_nav_corr": corr, "timing_nav_maxdiff": maxdiff}
    for _, rrow in res.iloc[1:].iterrows():
        k = "timing_cat_" + rrow["label"].split()[0].lower().replace("&", "")
        numbers[k + "_gap"] = rrow["gap_ann"]
        numbers[k + "_p"] = rrow["p_worse_than_random"]
    (TABLES / "timing_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))
    print(res.to_string())


if __name__ == "__main__":
    main()
