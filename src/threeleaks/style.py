"""Stage 4b. The skill leak against investable benchmarks (the pre-registered second pathway).

    python -m threeleaks.style

1. Calibration: the four-factor alpha of each index-fund series. A pure index fund should show
   about zero alpha minus its fees; anything else is a mismatch between the IIMA factor model and
   the indices Indian funds are actually benchmarked to.
2. Sharpe (1992) style analysis: each active fund's monthly return regressed on the returns of
   index funds an investor could buy (Nifty 50, Nifty Next 50, Nifty Midcap 150, Nifty Smallcap
   250), with weights at least zero and summing to one. Style alpha = mean residual. Runs to the
   last NAV month (no factor data needed), from the first month all four index series exist.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .benchmarks import index_returns
from .config import FACTOR_END, INTERIM, MIN_MONTHS_ALPHA, NW_LAGS, TABLES
from .models import design, fit
from .skill import DIVERSIFIED_EXCLUDE, fund_series

STYLE_ASSETS = ["nifty50", "next50", "midcap150", "smallcap250"]


def style_fit(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    k = X.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(lambda w: ((y - X @ w) ** 2).sum(), np.full(k, 1 / k), bounds=[(0, 1)] * k,
                   constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return res.x


def newey_west_mean_t(e: np.ndarray, lags: int) -> float:
    n = len(e)
    d = e - e.mean()
    v = (d ** 2).sum() / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (d[L:] * d[:-L]).sum() / n
    return e.mean() / np.sqrt(v / n)


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet").set_index("month")
    idx, counts, funds = index_returns(schemes, returns)
    funds.to_csv(TABLES / "benchmark_index_funds.csv", index=False)

    # 1. Calibration of the factor model on index funds.
    cal = []
    for col in idx.columns:
        s = idx[col].dropna()
        s = s[s.index <= pd.Period(FACTOR_END, "M")]
        f = factors.reindex(s.index).dropna()
        s = s.reindex(f.index)
        if len(s) < 24:
            continue
        res = fit((s - f["rf"]).to_numpy(), design(f), NW_LAGS)
        cal.append({"index": col, "start": str(s.index.min()), "end": str(s.index.max()), "months": len(s),
                    "alpha_ann": res["alpha"] * 12, "t_nw": res["t_nw"], "r2": res["r2"],
                    "b_mkt": res["betas"][0], "b_smb": res["betas"][1], "b_hml": res["betas"][2], "b_wml": res["betas"][3]})
    cal = pd.DataFrame(cal)
    cal.to_csv(TABLES / "style_factor_calibration.csv", index=False)

    # 2. Style analysis on investable index funds.
    B = idx[STYLE_ASSETS].dropna()
    start = B.index.min()
    out = []
    for plan in ("direct", "regular"):
        wide, meta = fund_series(schemes, returns, plan)
        W = wide.reindex(B.index)
        for code in W.columns:
            if meta.loc[code, "category"] in DIVERSIFIED_EXCLUDE:
                continue
            y = W[code].dropna()
            if len(y) < MIN_MONTHS_ALPHA:
                continue
            X = B.loc[y.index].to_numpy()
            w = style_fit(y.to_numpy(), X)
            e = y.to_numpy() - X @ w
            t = newey_west_mean_t(e, NW_LAGS)
            r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
            out.append({"plan": plan, "code": code, "category": meta.loc[code, "category"], "n": len(y),
                        "style_alpha_ann": e.mean() * 12, "t_nw": t, "p": 2 * (1 - stats.norm.cdf(abs(t))),
                        "style_r2": r2, **{"w_" + a: v for a, v in zip(STYLE_ASSETS, w)}})
    st = pd.DataFrame(out, columns=["plan", "code", "category", "n", "style_alpha_ann", "t_nw", "p", "style_r2"]
                      + ["w_" + a for a in STYLE_ASSETS])
    st.to_csv(TABLES / "style_funds.csv", index=False)
    if st.empty:
        print("style: no fund has %d months inside the style window yet" % MIN_MONTHS_ALPHA)
    summary = st.groupby("plan").agg(funds=("code", "count"), mean_alpha=("style_alpha_ann", "mean"),
                                     median_alpha=("style_alpha_ann", "median"),
                                     share_pos=("style_alpha_ann", lambda x: (x > 0).mean()),
                                     share_sig_pos=("t_nw", lambda x: (x > 1.96).mean()),
                                     share_sig_neg=("t_nw", lambda x: (x < -1.96).mean()),
                                     median_r2=("style_r2", "median"))
    summary.to_csv(TABLES / "style_summary.csv")
    numbers = {"style_start": str(start), "style_end": str(B.index.max()), "style_months": int(len(B))}
    for plan, r in summary.iterrows():
        tag = plan[:3]
        numbers.update({"style_%s_funds" % tag: int(r["funds"]), "style_%s_mean_alpha" % tag: float(r["mean_alpha"]),
                        "style_%s_median_alpha" % tag: float(r["median_alpha"]),
                        "style_%s_share_pos" % tag: float(r["share_pos"]),
                        "style_%s_share_sig_pos" % tag: float(r["share_sig_pos"]),
                        "style_%s_share_sig_neg" % tag: float(r["share_sig_neg"]),
                        "style_%s_median_r2" % tag: float(r["median_r2"])})
    for _, r in cal.iterrows():
        numbers["calib_%s_alpha" % r["index"]] = float(r["alpha_ann"])
        numbers["calib_%s_t" % r["index"]] = float(r["t_nw"])
        numbers["calib_%s_months" % r["index"]] = int(r["months"])
    (TABLES / "style_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(cal.to_string())
    print(summary.to_string())
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
