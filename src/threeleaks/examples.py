"""Worked examples: each method applied by hand to two real funds.

    python -m threeleaks.examples

Follows HDFC Mid Cap Fund and HDFC Large Cap Fund through every stage and recomputes, step by
step, numbers the pipeline produces in bulk, so the arithmetic can be shown and taught. Every
hand calculation is checked against the stored pipeline result. Writes results/examples.json,
which the report does not use.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from .commission import window_gap, windows_for
from .config import FACTOR_END, INTERIM, MIN_MONTHS_ALPHA, NW_LAGS, RESULTS, TABLES, TER_EVENT
from .models import FACTORS, design, fit
from .selection import RULES, pick
from .skill import DIVERSIFIED_EXCLUDE, align, fund_series, pi0_hat, tails
from .timing import implied_returns

FUNDS = {"mid": "HDFC Asset Management Company Limited | HDFC Mid Cap Fund",
         "large": "HDFC Asset Management Company Limited | HDFC Large Cap Fund"}


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def twin_examples(ex: dict, panel: pd.DataFrame, twins: pd.DataFrame) -> None:
    """1. The commission leak by hand: one month, the whole history, and Rs 1 lakh in each plan."""
    pairs = pd.read_csv(TABLES / "commission_pairs.csv").set_index("fund_id")
    for key, fid in FUNDS.items():
        p = panel[panel["fund_id"] == fid].sort_values("month")
        n, last = len(p), p.iloc[-1]
        gap = float(np.exp(12 * p["gap_log"].mean()) - 1)
        assert close(gap, pairs.loc[fid, "gap_ann"])
        ex.update({f"{key}_direct_code": twins.loc[fid, "direct_code"], f"{key}_regular_code": twins.loc[fid, "regular_code"],
                   f"{key}_corr": float(twins.loc[fid, "corr"]), f"{key}_twin_first": str(p["month"].iloc[0]),
                   f"{key}_twin_last": str(p["month"].iloc[-1]), f"{key}_twin_months": n,
                   f"{key}_m_month": str(last["month"]), f"{key}_m_rd": float(last["rd"]), f"{key}_m_rr": float(last["rr"]),
                   f"{key}_m_gap_log": float(last["gap_log"]), f"{key}_mean_gap_log": float(p["gap_log"].mean()),
                   f"{key}_gap_ann": gap,
                   f"{key}_cagr_direct": float(np.prod(1 + p["rd"]) ** (12 / n) - 1),
                   f"{key}_cagr_regular": float(np.prod(1 + p["rr"]) ** (12 / n) - 1),
                   f"{key}_lakh_direct": float(1e5 * np.prod(1 + p["rd"])),
                   f"{key}_lakh_regular": float(1e5 * np.prod(1 + p["rr"]))})


def event_examples(ex: dict, panel: pd.DataFrame) -> None:
    """2. The April 2019 expense-ratio cut: one pair's change, then the paired t-test by hand."""
    before, after = windows_for(TER_EVENT)
    both = pd.concat({"before": window_gap(panel, *before), "after": window_gap(panel, *after)}, axis=1).dropna()
    diff = both["after"] - both["before"]
    n, sd = len(diff), float(diff.std(ddof=1))
    t = float(diff.mean() / (sd / np.sqrt(n)))
    events = pd.read_csv(TABLES / "commission_events.csv").set_index("event")
    assert close(t, events.loc[TER_EVENT, "t"])
    ex.update({"ter_before_window": "%s to %s" % before, "ter_after_window": "%s to %s" % after, "ter_n": n,
               "ter_mean": float(diff.mean()), "ter_sd": sd, "ter_se": sd / np.sqrt(n), "ter_t": t})
    for key, fid in FUNDS.items():
        if fid in both.index:
            ex[f"{key}_ter_before"], ex[f"{key}_ter_after"] = float(both.loc[fid, "before"]), float(both.loc[fid, "after"])


def factor_examples(ex: dict, twins, schemes, returns, factors) -> None:
    """3. Four-factor alpha by hand: alpha = mean excess return minus each beta times its factor's mean."""
    wide, meta = fund_series(schemes, returns, "direct")
    excess, F = align(wide, factors)
    sk = pd.read_csv(TABLES / "skill_funds_direct.csv", dtype={"code": str}).set_index("code")
    for key, fid in FUNDS.items():
        code = twins.loc[fid, "direct_code"]
        y = excess[code].dropna()
        Fy = F.loc[y.index]
        res = fit(y.to_numpy(), design(Fy), NW_LAGS)
        fmean = Fy[FACTORS].mean()
        contrib = {f: float(b * fmean[f]) for f, b in zip(FACTORS, res["betas"])}
        assert close(y.mean() - sum(contrib.values()), res["alpha"], 1e-12)
        row = sk.loc[code]
        assert close(row["alpha_ann"], res["alpha"] * 12) and close(row["t_nw"], res["t_nw"])
        ex.update({f"{key}_ff_n": len(y), f"{key}_ff_start": str(y.index[0]), f"{key}_ff_end": str(y.index[-1]),
                   f"{key}_ff_mean_excess_ann": float(y.mean() * 12), f"{key}_ff_alpha_m": float(res["alpha"]),
                   f"{key}_ff_alpha_ann": float(res["alpha"] * 12), f"{key}_ff_se_ann": float(res["se"] * 12),
                   f"{key}_ff_se_nw_ann": float(res["se_nw"] * 12), f"{key}_ff_t": float(res["t"]),
                   f"{key}_ff_t_nw": float(res["t_nw"]), f"{key}_ff_r2": float(res["r2"]),
                   f"{key}_ff_resid_sd_ann": float(res["resid_sd"] * np.sqrt(12)),
                   f"{key}_p_boot": float(row["p_boot"]), f"{key}_q": float(row["q_value"]),
                   f"{key}_alpha_eb": float(row["alpha_eb"]), f"{key}_alpha_eb_sd": float(row["alpha_eb_sd"]),
                   f"{key}_prob_pos": float(row["prob_pos"])})
        for f, b in zip(FACTORS, res["betas"]):
            ex[f"{key}_ff_b_{f}"] = float(b)
            ex[f"{key}_ff_fmean_{f}_ann"] = float(fmean[f] * 12)
            ex[f"{key}_ff_contrib_{f}_ann"] = contrib[f] * 12


def fdr_examples(ex: dict, twins) -> pd.DataFrame:
    """4-6. Barras, Scaillet and Wermers by hand; one fund's q-value; empirical Bayes by hand."""
    summ = pd.read_csv(TABLES / "skill_fdr_summary.csv").set_index("label")
    out = None
    for plan, tag in (("direct", "dir"), ("regular", "reg")):
        st = pd.read_csv(TABLES / ("skill_funds_%s.csv" % plan), dtype={"code": str})
        div = st[~st["category"].isin(DIVERSIFIED_EXCLUDE)].reset_index(drop=True)
        p, t = div["p_boot"].to_numpy(), div["t_nw"].to_numpy()
        s = summ.loc[plan + " diversified"]
        N, lam, gp, gm = len(p), float(s["lambda"]), float(s["gamma_plus"]), float(s["gamma_minus"])
        pi0 = pi0_hat(p, lam)
        pi_plus, _, s_plus, _, lucky = tails(p, t, pi0, gp)
        _, pi_minus, _, s_minus, lucky_m = tails(p, t, pi0, gm)
        assert close(pi0, s["pi0"]) and close(max(0.0, pi_plus), s["pi_plus"])
        sig5 = int(((p <= 0.05) & (t > 0)).sum())
        ex.update({f"fdr_{tag}_n": N, f"fdr_{tag}_n_pos_t": int((t > 0).sum()), f"fdr_{tag}_lambda": lam,
                   f"fdr_{tag}_n_above": int((p > lam).sum()), f"fdr_{tag}_share_above": float((p > lam).mean()),
                   f"fdr_{tag}_pi0": pi0, f"fdr_{tag}_gamma": gp, f"fdr_{tag}_n_plus": int(((p <= gp) & (t > 0)).sum()),
                   f"fdr_{tag}_s_plus": float(s_plus), f"fdr_{tag}_lucky": float(lucky), f"fdr_{tag}_pi_plus": float(pi_plus),
                   f"fdr_{tag}_gamma_minus": gm, f"fdr_{tag}_n_minus": int(((p <= gm) & (t < 0)).sum()),
                   f"fdr_{tag}_s_minus": float(s_minus), f"fdr_{tag}_lucky_minus": float(lucky_m),
                   f"fdr_{tag}_pi_minus_raw": float(pi_minus), f"fdr_{tag}_sig5_n": sig5,
                   f"fdr_{tag}_sig5_false": float(pi0 * 0.025 * N), f"fdr_{tag}_sig5_fdr": float(pi0 * 0.025 * N / sig5)})
        # Empirical Bayes: the prior from the cross-section, then each example fund's posterior.
        a = div["alpha_ann"].to_numpy()
        se = np.abs(a / t)
        tau2 = float(np.var(a, ddof=1) - np.mean(se ** 2))
        ex.update({f"eb_{tag}_mu": float(a.mean()), f"eb_{tag}_var": float(np.var(a, ddof=1)),
                   f"eb_{tag}_mean_se2": float(np.mean(se ** 2)), f"eb_{tag}_tau": float(np.sqrt(tau2)),
                   f"eb_{tag}_prior_prob_pos": float(stats.norm.cdf(a.mean() / np.sqrt(tau2)))})
        if tag != "dir":
            continue
        out = div
        for key, fid in FUNDS.items():
            code = twins.loc[fid, "direct_code"]
            i = int(np.where(div["code"] == code)[0][0])
            # Storey q: among funds on the same side, k have a p-value at or below this one.
            same = np.sign(t) == np.sign(t[i])
            k = int((same & (p <= p[i])).sum())
            raw = min(1.0, pi0 * p[i] / 2 * N / k)
            se_i = se[i]
            w = tau2 / (tau2 + se_i ** 2)
            post = a.mean() + w * (a[i] - a.mean())
            post_sd = np.sqrt(tau2 * se_i ** 2 / (tau2 + se_i ** 2))
            assert close(post, div.loc[i, "alpha_eb"]) and close(stats.norm.cdf(post / post_sd), div.loc[i, "prob_pos"])
            ex.update({f"{key}_q_k": k, f"{key}_q_raw": float(raw), f"{key}_q_stored": float(div.loc[i, "q_value"]),
                       f"{key}_eb_se": float(se_i), f"{key}_eb_w": float(w), f"{key}_eb_post": float(post),
                       f"{key}_eb_post_sd": float(post_sd), f"{key}_eb_prob": float(stats.norm.cdf(post / post_sd))})
    return out


def bootstrap_and_style(ex: dict, twins) -> None:
    """7-8. The Fama-French bootstrap table, and each fund against investable index funds."""
    ff = pd.read_csv(TABLES / "skill_ff_bootstrap_regular.csv").set_index("percentile")
    for pc in (1, 5, 10, 50, 90, 95, 99):
        ex.update({f"ff_p{pc}_actual": float(ff.loc[pc, "t_actual"]), f"ff_p{pc}_sim": float(ff.loc[pc, "t_sim_mean"]),
                   f"ff_p{pc}_below": float(ff.loc[pc, "share_sims_below_actual"])})
    sf = pd.read_csv(TABLES / "style_funds.csv", dtype={"code": str})
    for key, fid in FUNDS.items():
        r = sf[(sf["plan"] == "direct") & (sf["code"] == twins.loc[fid, "direct_code"])].iloc[0]
        ex.update({f"{key}_style_n": int(r["n"]), f"{key}_style_alpha": float(r["style_alpha_ann"]),
                   f"{key}_style_t": float(r["t_nw"]), f"{key}_style_r2": float(r["style_r2"]),
                   **{f"{key}_style_w_{a}": float(r["w_" + a]) for a in ("nifty50", "next50", "midcap150", "smallcap250")}})


def persistence_counts(ex: dict, schemes, returns, factors) -> None:
    """9. How many regular plans were sorted each December, and the quintile table."""
    wide, meta = fund_series(schemes, returns, "regular")
    excess, _ = align(wide, factors)
    counts = {}
    for y in range(2009, 2025):
        end = pd.Period("%d-12" % y, "M")
        hist = excess.loc[(excess.index >= end - 35) & (excess.index <= end)]
        counts[y] = sum(1 for c in hist.columns if hist[c].notna().sum() >= MIN_MONTHS_ALPHA
                        and meta.loc[c, "category"] not in DIVERSIFIED_EXCLUDE)
    ex.update({"persist_first_year": min(counts), "persist_n_first": counts[min(counts)],
               "persist_n_2024": counts[2024], "persist_n_min": min(counts.values()), "persist_n_max": max(counts.values())})
    pers = pd.read_csv(TABLES / "skill_persistence_regular.csv").set_index("portfolio")
    for q in pers.index:
        tag = "spread" if q == "Q5-Q1" else "q" + str(q)
        ex[f"persist_{tag}_alpha"], ex[f"persist_{tag}_t"] = float(pers.loc[q, "alpha_ann"]), float(pers.loc[q, "t_nw"])


def timing_examples(ex: dict) -> None:
    """10. Timing by hand: one month's implied return, a toy two-year example, flows by year."""
    eq = pd.read_csv(TABLES / "timing_equity_series.csv", index_col=0)
    eq.index = pd.PeriodIndex(eq.index, freq="M")
    m, prev = pd.Period("2020-03", "M"), pd.Period("2020-02", "M")
    a0, a1, fl = eq.loc[prev, "aum"], eq.loc[m, "aum"], eq.loc[m, "net_inflow"]
    chk = pd.read_csv(TABLES / "timing_return_check.csv", index_col=0)
    ex.update({"tm_month": str(m), "tm_aum_prev": float(a0), "tm_aum": float(a1), "tm_flow": float(fl),
               "tm_ret": float((a1 - a0 - fl) / (a0 + 0.5 * fl)), "tm_nav_ret": float(chk.loc["2020-03", "nav"])})
    # Toy example (illustrative numbers, not data): Rs 100 in, fund +50%, investor adds Rs 150, fund -20%.
    end = (100 * 1.5 + 150) * 0.8
    irr = brentq(lambda x: -100 - 150 / (1 + x) + end / (1 + x) ** 2, -0.9, 1.0)
    ex.update({"toy_end": end, "toy_twr": float(np.sqrt(1.5 * 0.8) - 1), "toy_mwr": float(irr)})
    s = eq[eq.index >= pd.Period("2019-04", "M")]
    r = implied_returns(s).iloc[1:]
    flows = s["net_inflow"].iloc[1:]
    for y in sorted(set(r.index.year)):
        ex[f"tm_flow_{y}"] = float(flows[flows.index.year == y].sum())
        ex[f"tm_ret_{y}"] = float(np.prod(1 + r[r.index.year == y]) - 1)
        ex[f"tm_months_{y}"] = int((r.index.year == y).sum())
    tr = pd.read_csv(TABLES / "timing_results.csv").set_index("label").loc["All active equity"]
    ex.update({"tm_perm_mwr": float(tr["perm_mwr_mean"]), "tm_perm_year_mwr": float(tr["perm_year_mwr_mean"])})


def selection_examples(ex: dict, schemes, returns) -> None:
    """11. Holm by hand on the four rules; one December's picks; one fund's features and label."""
    perf = pd.read_csv(TABLES / "selection_performance.csv")
    d = perf[(perf["plan"] == "direct") & perf["strategy"].isin(RULES)].copy()
    d["p_hand"] = 2 * stats.norm.sf(d["t_nw"].abs())
    d = d.sort_values("p_hand").reset_index(drop=True)
    m = len(d)
    d["mult"] = m - d.index
    d["holm"] = np.minimum(1.0, np.maximum.accumulate(d["mult"] * d["p_hand"]))
    for _, r in d.iterrows():
        assert close(r["holm"], r["p_holm"])
        ex.update({f"holm_{r['strategy']}_t": float(r["t_nw"]), f"holm_{r['strategy']}_p": float(r["p_hand"]),
                   f"holm_{r['strategy']}_mult": int(r["mult"]), f"holm_{r['strategy']}_raw": float(r["mult"] * r["p_hand"]),
                   f"holm_{r['strategy']}_holm": float(r["holm"])})
    ex["holm_order"] = d["strategy"].tolist()

    feat = pd.read_parquet(INTERIM / "selection_features.parquet")
    wide, meta = fund_series(schemes, returns, "direct")
    wide = wide[wide.index <= pd.Period(FACTOR_END, "M")]
    code_of = meta.reset_index().set_index("fund_id")["code"]
    y, dec = 2019, pd.Period("2019-12", "M")
    cand = feat[(feat["year"] == y) & feat["fund_id"].isin(code_of.index)]
    alive = [f for f in cand["fund_id"] if code_of[f] in wide.columns and pd.notna(wide.loc[dec, code_of[f]])]
    cand = cand[cand["fund_id"].isin(alive)]
    train = feat[feat["year"] <= y - 1]
    nxt = wide.loc[(wide.index > dec) & (wide.index <= dec + 12)]

    def year_return(fids):
        return float(np.prod(1 + nxt[[code_of[f] for f in fids]].mean(axis=1)) - 1)
    ex.update({"sel19_n": len(cand), "sel19_q": max(1, len(cand) // 5), "sel19_train_rows": int(train["label"].notna().sum()),
               "sel19_all_ret": year_return(cand["fund_id"])})
    for rule in RULES:
        chosen = pick(cand, train, rule)
        ex[f"sel19_{rule}_ret"] = year_return(chosen)
        for key, fid in FUNDS.items():
            ex[f"sel19_{rule}_has_{key}"] = fid in chosen
    for key, fid in FUNDS.items():
        r = feat[(feat["year"] == y) & (feat["fund_id"] == fid)].iloc[0]
        ex.update({f"{key}_f19_alpha36_ann": float(r["alpha36"] * 12), f"{key}_f19_t36": float(r["t36"]),
                   f"{key}_f19_r2": float(r["r2_36"]), f"{key}_f19_ret12": float(r["ret12"]),
                   f"{key}_f19_ret12_rel": float(r["ret12_rel"]), f"{key}_f19_age": int(r["age"]),
                   f"{key}_f19_twin_gap12": float(r["twin_gap12"]), f"{key}_f19_label_ann": float(r["label"] * 12)})


def rest(ex: dict) -> None:
    """12-15. Synthesis arithmetic, closet groups, the external anchor, the scorecard rows."""
    n = json.loads((RESULTS / "numbers.json").read_text())
    aum = n["syn_aum_2025_cr"]
    ex.update({"syn_aum_lakh_cr": aum / 1e5, "syn_check_commission": n["syn_commission"] * aum * n["syn_reg_share"],
               "syn_check_low": n["syn_commission"] * aum * 0.60, "syn_check_high": n["syn_commission"] * aum * 0.88,
               "syn_check_timing": n["syn_timing"] * aum, "syn_check_skill": n["syn_skill_vs_market"] * aum})
    assert close(ex["syn_check_commission"], n["syn_rs_commission_cr"], 1e-6)
    cl = pd.read_csv(TABLES / "closet_funds.csv").dropna(subset=["te_before", "te_after"])
    ex.update({"closet_n_mid": int((cl["category"] == "Mid Cap").sum()), "closet_n_small": int((cl["category"] == "Small Cap").sum())})
    sc = pd.read_csv(TABLES / "scorecard.csv").set_index("fund")
    for key, name in (("mid", "HDFC Mid Cap Fund"), ("large", "HDFC Large Cap Fund")):
        ex.update({f"{key}_card_commission": float(sc.loc[name, "commission"]), f"{key}_card_verdict": sc.loc[name, "verdict"]})


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet")
    twins = pd.read_parquet(INTERIM / "twins.parquet").set_index("fund_id")
    panel = pd.read_parquet(INTERIM / "twin_panel.parquet")
    ex = {}
    twin_examples(ex, panel, twins)
    event_examples(ex, panel)
    factor_examples(ex, twins, schemes, returns, factors)
    fdr_examples(ex, twins)
    bootstrap_and_style(ex, twins)
    persistence_counts(ex, schemes, returns, factors)
    timing_examples(ex)
    selection_examples(ex, schemes, returns)
    rest(ex)
    (RESULTS / "examples.json").write_text(json.dumps(ex, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    print(json.dumps(ex, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


if __name__ == "__main__":
    main()
