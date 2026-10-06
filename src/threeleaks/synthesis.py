"""Stage 7a. Synthesis: the three leaks on one window, in percent and in rupees.

    python -m threeleaks.synthesis

Window: 2019-05 to 2025-12, the months where twin plans, flows (consistent layout) and the
IIMA factors all exist. Decomposition (annualised, approximately additive):

    passive alternative - investor's money-weighted return
      = [passive - direct plans]          the skill leak (negative means skill added value)
      + [direct plans - regular plans]    the commission leak
      + [funds' time-weighted - investors' money-weighted]  the timing leak

Each piece is measured on its natural basis (equal-weighted funds for the first two, AMFI's
AUM aggregate for timing), so the sum is an approximation; the report says so.
Rupee figures apply each leak to 2025 average active equity AUM.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .benchmarks import index_returns
from .config import FIGURES, INTERIM, NW_LAGS, TABLES
from .models import design, fit
from .skill import DIVERSIFIED_EXCLUDE, fund_series
from .timing import equity_series, implied_returns, mwr, simulate_end

W_START, W_END = pd.Period("2019-05", "M"), pd.Period("2025-12", "M")
REG_SHARE = {"2020-03": 0.88, "2025-03": 0.79}   # AMFI Annual Report FY2025: direct plans 12% -> 21% of SIP AUM


def ann(r: pd.Series) -> float:
    r = r.dropna()
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)


def ew_returns(schemes, returns, plan) -> pd.Series:
    wide, meta = fund_series(schemes, returns, plan)
    keep = [c for c in wide.columns if meta.loc[c, "category"] not in DIVERSIFIED_EXCLUDE]
    w = wide[keep]
    return w[(w.index >= W_START) & (w.index <= W_END)].mean(axis=1)


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet").set_index("month")
    flows = pd.read_parquet(INTERIM / "flows.parquet")
    twin = pd.read_parquet(INTERIM / "twin_panel.parquet")

    # 1. Commission leak in the window (pairs with at least 12 months in it).
    tw = twin[(twin["month"] >= W_START) & (twin["month"] <= W_END)]
    g = tw.groupby("fund_id")["gap_log"].agg(["count", "mean"])
    g = g[g["count"] >= 12]
    commission = float((np.exp(12 * g["mean"]) - 1).mean())

    # 2. Skill leak: direct plans vs passive alternatives.
    r_dir = ew_returns(schemes, returns, "direct")
    r_reg = ew_returns(schemes, returns, "regular")
    f = factors[(factors.index >= W_START) & (factors.index <= W_END)]
    market = f["mkt_rf"] + f["rf"]
    idx, _, _ = index_returns(schemes, returns)
    n50 = idx["nifty50"].reindex(f.index)
    skill_vs_market = ann(market) - ann(r_dir.reindex(f.index))
    skill_vs_n50 = ann(n50) - ann(r_dir.reindex(f.index))
    al = fit((r_dir.reindex(f.index) - f["rf"]).to_numpy(), design(f), NW_LAGS)
    commission_ew = ann(r_dir.reindex(f.index)) - ann(r_reg.reindex(f.index))

    # 3. Timing leak on the same window.
    eq = equity_series(flows)
    s = eq[(eq.index >= W_START - 1) & (eq.index <= W_END)]
    rr = implied_returns(s).iloc[1:]
    fl = s["net_inflow"].iloc[1:].to_numpy()
    a0 = float(s["aum"].iloc[0])
    twr = float(np.prod(1 + rr) ** (12 / len(rr)) - 1)
    m = mwr(a0, fl, simulate_end(a0, rr.to_numpy(), fl))
    mwr_a = (1 + m) ** 12 - 1
    timing = twr - mwr_a

    # 4. Rupees at 2025 asset levels.
    aum25 = float(eq[(eq.index >= pd.Period("2025-01", "M")) & (eq.index <= pd.Period("2025-12", "M"))]["aum"].mean())
    reg_share_25 = REG_SHARE["2025-03"]
    rupees = {"commission": commission * aum25 * reg_share_25,
              "commission_low": commission * aum25 * 0.60, "commission_high": commission * aum25 * REG_SHARE["2020-03"],
              "skill_vs_market": skill_vs_market * aum25, "skill_vs_n50": skill_vs_n50 * aum25,
              "timing": timing * aum25}

    # 5. Do higher commissions buy better funds? Twin gap vs direct-plan four-factor alpha.
    pairs = pd.read_csv(TABLES / "commission_pairs.csv")
    sk = pd.read_csv(TABLES / "skill_funds_direct.csv")
    both = pairs.merge(sk[["fund_id", "alpha_ann", "t_nw"]], on="fund_id").dropna()
    rho, p_rho = stats.spearmanr(both["gap_ann"], both["alpha_ann"])

    decomposition = pd.DataFrame([
        {"leak": "Skill (passive market minus direct plans)", "pct_per_year": skill_vs_market, "rs_crore_2025": rupees["skill_vs_market"]},
        {"leak": "Commission (direct minus regular twins)", "pct_per_year": commission, "rs_crore_2025": rupees["commission"]},
        {"leak": "Timing (funds' TWR minus investors' MWR)", "pct_per_year": timing, "rs_crore_2025": rupees["timing"]},
    ])
    decomposition.to_csv(TABLES / "synthesis_decomposition.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    vals = decomposition["pct_per_year"] * 100
    colors = ["#1F3A5F" if v >= 0 else "#2E7D4F" for v in vals]
    ax.barh(["Skill", "Commission", "Timing"], vals, color=colors)
    ax.axvline(0, color="#999", lw=1)
    ax.set_xlabel("Points a year lost (bars right) or gained (bars left)\nby the investor, 2019-05 to 2025-12", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthesis_leaks.png", dpi=200)
    plt.close(fig)

    numbers = {
        "syn_window": "%s to %s" % (W_START, W_END), "syn_months": int(len(f)),
        "syn_commission": commission, "syn_commission_ew_returns": commission_ew,
        "syn_ret_direct": ann(r_dir.reindex(f.index)), "syn_ret_regular": ann(r_reg.reindex(f.index)),
        "syn_ret_market": ann(market), "syn_ret_nifty50": ann(n50),
        "syn_skill_vs_market": skill_vs_market, "syn_skill_vs_nifty50": skill_vs_n50,
        "syn_direct_alpha": al["alpha"] * 12, "syn_direct_alpha_t": al["t_nw"],
        "syn_twr": twr, "syn_mwr": mwr_a, "syn_timing": timing,
        "syn_aum_2025_cr": aum25, "syn_reg_share": reg_share_25,
        "syn_rs_commission_cr": rupees["commission"], "syn_rs_commission_low_cr": rupees["commission_low"],
        "syn_rs_commission_high_cr": rupees["commission_high"],
        "syn_rs_skill_cr": rupees["skill_vs_market"], "syn_rs_skill_n50_cr": rupees["skill_vs_n50"],
        "syn_rs_timing_cr": rupees["timing"],
        "syn_gap_alpha_rho": float(rho), "syn_gap_alpha_p": float(p_rho), "syn_gap_alpha_n": int(len(both)),
    }
    (TABLES / "synthesis_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))
    print(decomposition.to_string())


if __name__ == "__main__":
    main()
