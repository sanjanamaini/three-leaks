"""Stage 4. The skill leak: after fees, how many funds truly beat the factors, once luck is removed?

    python -m threeleaks.skill

1. Four-factor alpha for every active equity fund (regular plans 2006-2025; direct plans 2013-2025).
2. Barras, Scaillet and Wermers (2010) false discovery rate: Newey-West t-statistics, per-fund
   residual bootstrap p-values (1,000 draws), Storey's pi-zero with lambda chosen by bootstrap MSE,
   pi-plus and pi-minus with gamma chosen by bootstrap MSE.
3. Fama and French (2010) cross-sectional bootstrap: actual vs simulated percentiles of t(alpha).
4. Carhart (1997) persistence: quintiles on prior 36-month alpha t-statistic, held one year.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (DIRECT_PLANS_START, FACTOR_END, FIGURES, INTERIM, MIN_MONTHS_ALPHA, N_BOOT, N_FUND_BOOT,
                     NW_LAGS, SEED, TABLES)
from .panels import NON_ACTIVE
from .models import FACTORS, batch_tstats, design, fit

LAMBDAS = np.round(np.arange(0.30, 0.81, 0.05), 2)
GAMMAS = np.round(np.arange(0.20, 0.51, 0.05), 2)
DIVERSIFIED_EXCLUDE = {"Sectoral/Thematic"} | NON_ACTIVE


# --- universe -------------------------------------------------------------------------------
def fund_series(schemes: pd.DataFrame, returns: pd.DataFrame, plan: str) -> pd.DataFrame:
    """One growth code per fund for the given plan (longest history if several), wide by month."""
    s = schemes[schemes["is_growth"] & ~schemes["institutional"] & schemes["structure"].str.startswith("open")
                & ~schemes["category"].isin(NON_ACTIVE) & (schemes["plan"] == plan)]
    n = returns.groupby("code")["ret"].count()
    s = s.assign(n=s["code"].map(n).fillna(0)).sort_values("n").groupby("fund_id").tail(1)
    r = returns[returns["code"].isin(s["code"])]
    wide = r.pivot(index="month", columns="code", values="ret").sort_index()
    if plan == "direct":
        # Direct plans as a market-wide choice start in January 2013 (pre-registered window). Before that,
        # only Quantum Mutual Fund's schemes were direct-sold (AMFI kept their codes as "Direct Plan").
        wide = wide[wide.index >= pd.Period(DIRECT_PLANS_START, "M")]
    meta = s.set_index("code")[["fund_id", "name", "category", "amc"]]
    return wide, meta


def align(wide: pd.DataFrame, factors: pd.DataFrame) -> tuple:
    f = factors.set_index("month")
    f = f[f.index <= pd.Period(FACTOR_END, "M")]
    idx = wide.index.intersection(f.index)
    W = wide.loc[idx]
    F = f.loc[idx]
    excess = W.sub(F["rf"], axis=0)
    return excess, F


# --- per-fund regressions and bootstrap p-values ---------------------------------------------
def nw_var_alpha(z: np.ndarray, E: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West variance of alpha for many residual columns at once.

    var = sum_t q_t^2 + 2 sum_L w_L sum_t q_t q_{t-L}, with q = z * e and z = X a (a: first row of (X'X)^-1).
    """
    q = z[:, None] * E
    v = (q ** 2).sum(0)
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (q[L:] * q[:-L]).sum(0)
    return v


def fund_stats(excess: pd.DataFrame, F: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for code in excess.columns:
        y = excess[code].dropna()
        if len(y) < MIN_MONTHS_ALPHA:
            continue
        X = design(F.loc[y.index])
        res = fit(y.to_numpy(), X, NW_LAGS)
        # Residual-only bootstrap under alpha = 0 (Kosowski et al. 2006; Barras et al. 2010).
        xtx_inv = np.linalg.inv(X.T @ X)
        a = xtx_inv[0]
        z = X @ a
        n = len(y)
        e = res["resid"]
        draws = rng.integers(0, n, size=(n, N_FUND_BOOT))
        Ystar = (X[:, 1:] @ res["betas"])[:, None] + e[draws]
        B = xtx_inv @ X.T @ Ystar
        Estar = Ystar - X @ B
        se_star = np.sqrt(nw_var_alpha(z, Estar, NW_LAGS))
        t_star = B[0] / se_star
        p_boot = float((np.abs(t_star) >= abs(res["t_nw"])).mean())
        rows.append({"code": code, "n": n, "alpha_m": res["alpha"], "alpha_ann": res["alpha"] * 12,
                     "t_ols": res["t"], "t_nw": res["t_nw"], "p_boot": max(p_boot, 1 / N_FUND_BOOT),
                     "r2": res["r2"], "resid_sd": res["resid_sd"],
                     **{"b_" + f: b for f, b in zip(FACTORS, res["betas"])}})
    return pd.DataFrame(rows)


# --- false discovery rate (Barras, Scaillet and Wermers 2010) -----------------------------
def pi0_hat(p: np.ndarray, lam: float) -> float:
    return min(1.0, (p > lam).mean() / (1 - lam))


def choose_lambda(p: np.ndarray, rng: np.random.Generator, B: int = 1000) -> float:
    est = np.array([pi0_hat(p, l) for l in LAMBDAS])
    target = est.min()
    mse = np.zeros(len(LAMBDAS))
    for _ in range(B):
        pb = rng.choice(p, size=len(p), replace=True)
        mse += (np.array([pi0_hat(pb, l) for l in LAMBDAS]) - target) ** 2
    return float(LAMBDAS[np.argmin(mse)])


def tails(p: np.ndarray, t: np.ndarray, pi0: float, g: float) -> tuple:
    s_plus = ((p <= g) & (t > 0)).mean()
    s_minus = ((p <= g) & (t < 0)).mean()
    f = pi0 * g / 2
    return s_plus - f, s_minus - f, s_plus, s_minus, f


def choose_gamma(p: np.ndarray, t: np.ndarray, pi0: float, rng: np.random.Generator, B: int = 1000) -> tuple:
    est_plus = np.array([tails(p, t, pi0, g)[0] for g in GAMMAS])
    est_minus = np.array([tails(p, t, pi0, g)[1] for g in GAMMAS])
    mse_p, mse_m = np.zeros(len(GAMMAS)), np.zeros(len(GAMMAS))
    n = len(p)
    for _ in range(B):
        i = rng.integers(0, n, n)
        pb, tb = p[i], t[i]
        mse_p += (np.array([tails(pb, tb, pi0, g)[0] for g in GAMMAS]) - est_plus.max()) ** 2
        mse_m += (np.array([tails(pb, tb, pi0, g)[1] for g in GAMMAS]) - est_minus.max()) ** 2
    return float(GAMMAS[np.argmin(mse_p)]), float(GAMMAS[np.argmin(mse_m)])


def fdr_summary(st: pd.DataFrame, rng: np.random.Generator, label: str) -> dict:
    p, t = st["p_boot"].to_numpy(), st["t_nw"].to_numpy()
    lam = choose_lambda(p, rng)
    pi0 = pi0_hat(p, lam)
    g_plus, g_minus = choose_gamma(p, t, pi0, rng)
    pi_plus = max(0.0, tails(p, t, pi0, g_plus)[0])
    pi_minus = max(0.0, tails(p, t, pi0, g_minus)[1])
    # Uncertainty: bootstrap over funds (ignores cross-fund dependence; reported as such).
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(p), len(p))
        pb, tb = p[i], t[i]
        p0 = pi0_hat(pb, lam)
        bs.append((p0, max(0.0, tails(pb, tb, p0, g_plus)[0]), max(0.0, tails(pb, tb, p0, g_minus)[1])))
    bs = np.array(bs)
    out = {"label": label, "funds": int(len(p)), "lambda": lam, "gamma_plus": g_plus, "gamma_minus": g_minus,
           "pi0": pi0, "pi_plus": pi_plus, "pi_minus": pi_minus,
           "pi0_lo": float(np.quantile(bs[:, 0], 0.025)), "pi0_hi": float(np.quantile(bs[:, 0], 0.975)),
           "pi_plus_lo": float(np.quantile(bs[:, 1], 0.025)), "pi_plus_hi": float(np.quantile(bs[:, 1], 0.975)),
           "pi_minus_lo": float(np.quantile(bs[:, 2], 0.025)), "pi_minus_hi": float(np.quantile(bs[:, 2], 0.975))}
    # Fixed-parameter sensitivity (lambda 0.6, gamma 0.4), as the paper suggests.
    p0f = pi0_hat(p, 0.6)
    out["pi_plus_fixed"] = max(0.0, tails(p, t, p0f, 0.40)[0])
    out["pi_minus_fixed"] = max(0.0, tails(p, t, p0f, 0.40)[1])
    out["pi0_fixed"] = p0f
    # How many "significant winners" at 5% are lucky?
    tp, tm, sp, sm, f = tails(p, t, pi0, 0.05)
    out["sig_winners_5pct"] = float(sp)
    out["fdr_winners_5pct"] = float(min(1.0, f / sp)) if sp > 0 else None
    out["sig_losers_5pct"] = float(sm)
    out["fdr_losers_5pct"] = float(min(1.0, f / sm)) if sm > 0 else None
    return out


def storey_q(p: np.ndarray, t: np.ndarray, pi0: float) -> np.ndarray:
    """Directional q-value: the false discovery rate if every fund with p at or below this one,
    on the same side of zero, were called skilled (or unskilled)."""
    q = np.full(len(p), np.nan)
    for sign in (1, -1):
        idx = np.where(np.sign(t) == sign)[0]
        order = idx[np.argsort(p[idx])]
        n_tot = len(p)
        fdr = np.array([min(1.0, pi0 * p[i] / 2 * n_tot / (k + 1)) for k, i in enumerate(order)])
        fdr = np.minimum.accumulate(fdr[::-1])[::-1]
        q[order] = fdr
    return q


# --- Fama and French (2010) cross-sectional bootstrap --------------------------------------
def ff_bootstrap(excess: pd.DataFrame, F: pd.DataFrame, stats_df: pd.DataFrame, rng) -> pd.DataFrame:
    codes = stats_df["code"].tolist()
    Y = excess[codes].to_numpy()
    M = (~np.isnan(Y)).astype(float)
    alpha = stats_df.set_index("code").loc[codes, "alpha_m"].to_numpy()
    Y0 = np.where(M > 0, Y - alpha[None, :], 0.0)  # zero true alpha by construction
    X = design(F)
    T = len(F)
    pct = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]
    actual = np.nanpercentile(stats_df["t_ols"].to_numpy(), pct)
    sims = np.zeros((N_BOOT, len(pct)))
    for b in range(N_BOOT):
        idx = rng.integers(0, T, T)
        _, tb = batch_tstats(Y0[idx], M[idx], X[idx], min_obs=8)
        tb = tb[~np.isnan(tb)]
        sims[b] = np.percentile(tb, pct)
    below = (sims < actual[None, :]).mean(0)
    return pd.DataFrame({"percentile": pct, "t_actual": actual, "t_sim_mean": sims.mean(0),
                         "share_sims_below_actual": below})


# --- persistence (Carhart 1997) ------------------------------------------------------------
def persistence(excess: pd.DataFrame, F: pd.DataFrame, raw: pd.DataFrame, meta: pd.DataFrame) -> tuple:
    years = range(2009, 2025)
    port = {q: [] for q in range(1, 6)}
    for y in years:
        end = pd.Period("%d-12" % y, "M")
        start = end - 35
        hist = excess.loc[(excess.index >= start) & (excess.index <= end)]
        ok = [c for c in hist.columns if hist[c].notna().sum() >= MIN_MONTHS_ALPHA
              and meta.loc[c, "category"] not in DIVERSIFIED_EXCLUDE]
        if len(ok) < 25:
            continue
        tstats = {}
        for c in ok:
            yy = hist[c].dropna()
            tstats[c] = fit(yy.to_numpy(), design(F.loc[yy.index]), NW_LAGS)["t"]
        ts = pd.Series(tstats)
        qs = pd.qcut(ts.rank(method="first"), 5, labels=False) + 1
        nxt = raw.loc[(raw.index > end) & (raw.index <= end + 12)]
        for q in range(1, 6):
            members = qs[qs == q].index
            port[q].append(nxt[members].mean(axis=1))   # equal-weighted; dead funds drop out
    rets = pd.DataFrame({q: pd.concat(v) for q, v in port.items()}).sort_index()
    rets["Q5-Q1"] = rets[5] - rets[1]
    f = F.reindex(rets.index)
    rows = []
    for col in rets.columns:
        y = rets[col] - (0 if col == "Q5-Q1" else f["rf"])
        ok = y.notna() & f["mkt_rf"].notna()
        res = fit(y[ok].to_numpy(), design(f[ok]), NW_LAGS)
        rows.append({"portfolio": str(col), "alpha_ann": res["alpha"] * 12, "t_nw": res["t_nw"], "months": int(ok.sum())})
    return pd.DataFrame(rows), rets


def empirical_bayes(alpha: np.ndarray, se: np.ndarray) -> tuple:
    """Normal-normal shrinkage: true alpha ~ N(mu, tau^2), estimate ~ N(true, se^2).
    Method-of-moments prior; returns posterior mean, posterior sd, P(true alpha > 0), mu, tau."""
    from scipy.stats import norm
    mu = float(np.mean(alpha))
    tau2 = max(0.0, float(np.var(alpha, ddof=1) - np.mean(se ** 2)))
    w = tau2 / (tau2 + se ** 2)
    post = mu + w * (alpha - mu)
    post_sd = np.sqrt(tau2 * se ** 2 / (tau2 + se ** 2)) if tau2 > 0 else np.zeros_like(se)
    prob = norm.cdf(post / np.where(post_sd > 0, post_sd, np.inf)) if tau2 > 0 else (post > 0).astype(float)
    return post, post_sd, prob, mu, float(np.sqrt(tau2))


def shrinkage_oos(excess: pd.DataFrame, F: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Does shrinkage predict later alpha better than the raw estimate? Estimate 2013-2019, realise 2020-2025."""
    est_w = (excess.index >= pd.Period("2013-01", "M")) & (excess.index <= pd.Period("2019-12", "M"))
    real_w = (excess.index >= pd.Period("2020-01", "M")) & (excess.index <= pd.Period(FACTOR_END, "M"))
    rows = []
    for c in excess.columns:
        if meta.loc[c, "category"] in DIVERSIFIED_EXCLUDE:
            continue
        a, b = excess.loc[est_w, c].dropna(), excess.loc[real_w, c].dropna()
        if len(a) < MIN_MONTHS_ALPHA or len(b) < 24:
            continue
        ra = fit(a.to_numpy(), design(F.loc[a.index]), NW_LAGS)
        rb = fit(b.to_numpy(), design(F.loc[b.index]), NW_LAGS)
        rows.append((ra["alpha"] * 12, ra["se_nw"] * 12, rb["alpha"] * 12))
    if len(rows) < 30:
        return {}
    d = np.array(rows)
    post, *_ = empirical_bayes(d[:, 0], d[:, 1])
    from scipy.stats import spearmanr
    return {"funds": len(d), "mse_raw": float(np.mean((d[:, 0] - d[:, 2]) ** 2)),
            "mse_shrunk": float(np.mean((post - d[:, 2]) ** 2)),
            "rho_raw": float(spearmanr(d[:, 0], d[:, 2])[0]), "rho_shrunk": float(spearmanr(post, d[:, 2])[0]),
            "mean_later_alpha": float(d[:, 2].mean())}


def run_plan(plan: str, schemes, returns, factors, rng) -> dict:
    wide, meta = fund_series(schemes, returns, plan)
    excess, F = align(wide, factors)
    st = fund_stats(excess, F, rng)
    st = st.merge(meta, left_on="code", right_index=True)
    div = st[~st["category"].isin(DIVERSIFIED_EXCLUDE)].copy()
    sec = st[st["category"] == "Sectoral/Thematic"].copy()
    summ = fdr_summary(div, rng, plan + " diversified")
    div["q_value"] = storey_q(div["p_boot"].to_numpy(), div["t_nw"].to_numpy(), summ["pi0"])
    se_ann = (div["alpha_ann"] / div["t_nw"]).abs().to_numpy()
    post, post_sd, prob, mu, tau = empirical_bayes(div["alpha_ann"].to_numpy(), se_ann)
    div["alpha_eb"], div["alpha_eb_sd"], div["prob_pos"] = post, post_sd, prob
    summ.update({"eb_mu": mu, "eb_tau": tau, "eb_share_prob_pos_90": float((prob >= 0.9).mean())})
    # Survivorship: average alpha of funds alive at the end vs funds that closed earlier.
    last = schemes.set_index("code")["last_month"]
    end_m = last.max()
    alive = div["code"].map(last) == end_m
    summ.update({"alive_funds": int(alive.sum()), "dead_funds": int((~alive).sum()),
                 "alpha_alive": float(div.loc[alive, "alpha_ann"].mean()),
                 "alpha_dead": float(div.loc[~alive, "alpha_ann"].mean()) if (~alive).any() else None,
                 "alpha_all": float(div["alpha_ann"].mean())})
    st = st.merge(div[["code", "q_value", "alpha_eb", "alpha_eb_sd", "prob_pos"]], on="code", how="left")
    st.to_csv(TABLES / ("skill_funds_%s.csv" % plan), index=False)
    out = {"plan": plan, "summary": summ}
    if len(sec) >= 30:
        out["sectoral"] = fdr_summary(sec, rng, plan + " sectoral/thematic")
    # Minimum-history sensitivity: 60 months, as in Barras et al. (2010).
    long = div[div["n"] >= 60]
    if len(long) >= 50:
        out["min60"] = fdr_summary(long, rng, plan + " diversified, 60+ months")
    by_cat = div.groupby("category").agg(funds=("code", "count"), mean_alpha=("alpha_ann", "mean"),
                                         median_alpha=("alpha_ann", "median"),
                                         share_sig_pos=("t_nw", lambda x: (x > 1.96).mean()),
                                         share_sig_neg=("t_nw", lambda x: (x < -1.96).mean()),
                                         median_r2=("r2", "median"))
    by_cat.to_csv(TABLES / ("skill_by_category_%s.csv" % plan))
    out["by_category"] = by_cat
    out["shrinkage_oos"] = shrinkage_oos(excess, F, meta) if plan == "direct" else {}
    out["stats"] = div
    out["excess"], out["F"], out["wide"], out["meta"] = excess, F, wide, meta
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet")

    reg = run_plan("regular", schemes, returns, factors, rng)
    dire = run_plan("direct", schemes, returns, factors, rng)

    ff = ff_bootstrap(reg["excess"], reg["F"], reg["stats"], rng)
    ff.to_csv(TABLES / "skill_ff_bootstrap_regular.csv", index=False)
    ffd = ff_bootstrap(dire["excess"], dire["F"], dire["stats"], rng)
    ffd.to_csv(TABLES / "skill_ff_bootstrap_direct.csv", index=False)

    pers, pers_rets = persistence(reg["excess"], reg["F"], reg["wide"], reg["meta"])
    pers.to_csv(TABLES / "skill_persistence_regular.csv", index=False)

    summaries = [reg["summary"], dire["summary"]] + [x[k] for x in (reg, dire) for k in ("sectoral", "min60") if k in x]
    pd.DataFrame(summaries).to_csv(TABLES / "skill_fdr_summary.csv", index=False)

    # Figure: alpha t-stat distribution vs the bootstrap null (regular plans).
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ff["t_sim_mean"], ff["t_actual"], "o-", color="#1F3A5F", label="actual vs zero-alpha world")
    lim = [min(ff["t_sim_mean"].min(), ff["t_actual"].min()) - 0.3, max(ff["t_sim_mean"].max(), ff["t_actual"].max()) + 0.3]
    ax.plot(lim, lim, color="#999", lw=1, ls="--", label="luck alone")
    ax.set_xlabel("t(alpha) percentile if every fund had zero true alpha")
    ax.set_ylabel("Actual t(alpha) at the same percentile")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "skill_ff_bootstrap.png", dpi=200)
    plt.close(fig)

    def g(d, k):
        return d.get(k)
    numbers = {}
    for tag, res in (("reg", reg), ("dir", dire)):
        s = res["summary"]
        div = res["stats"]
        numbers.update({
            "skill_%s_funds" % tag: s["funds"],
            "skill_%s_mean_alpha" % tag: float(div["alpha_ann"].mean()),
            "skill_%s_median_alpha" % tag: float(div["alpha_ann"].median()),
            "skill_%s_share_pos_alpha" % tag: float((div["alpha_ann"] > 0).mean()),
            "skill_%s_share_sig_pos" % tag: float((div["t_nw"] > 1.96).mean()),
            "skill_%s_pi0" % tag: s["pi0"], "skill_%s_pi_plus" % tag: s["pi_plus"],
            "skill_%s_pi_minus" % tag: s["pi_minus"],
            "skill_%s_pi_plus_lo" % tag: s["pi_plus_lo"], "skill_%s_pi_plus_hi" % tag: s["pi_plus_hi"],
            "skill_%s_pi_minus_lo" % tag: s["pi_minus_lo"], "skill_%s_pi_minus_hi" % tag: s["pi_minus_hi"],
            "skill_%s_pi0_lo" % tag: s["pi0_lo"], "skill_%s_pi0_hi" % tag: s["pi0_hi"],
            "skill_%s_lambda" % tag: s["lambda"], "skill_%s_gamma_plus" % tag: s["gamma_plus"],
            "skill_%s_pi_plus_fixed" % tag: s["pi_plus_fixed"], "skill_%s_pi_minus_fixed" % tag: s["pi_minus_fixed"],
            "skill_%s_sig_winners_5pct" % tag: s["sig_winners_5pct"], "skill_%s_fdr_winners_5pct" % tag: g(s, "fdr_winners_5pct"),
            "skill_%s_median_r2" % tag: float(div["r2"].median()),
            "skill_%s_eb_mu" % tag: s["eb_mu"], "skill_%s_eb_tau" % tag: s["eb_tau"],
            "skill_%s_eb_share_prob90" % tag: s["eb_share_prob_pos_90"],
            "skill_%s_alive_funds" % tag: s["alive_funds"], "skill_%s_dead_funds" % tag: s["dead_funds"],
            "skill_%s_alpha_alive" % tag: s["alpha_alive"], "skill_%s_alpha_dead" % tag: s["alpha_dead"],
            "skill_%s_alpha_all" % tag: s["alpha_all"],
        })
        for k2, v2 in res.get("shrinkage_oos", {}).items():
            numbers["skill_%s_oos_%s" % (tag, k2)] = v2
        if "min60" in res:
            numbers["skill_%s_min60_pi_plus" % tag] = res["min60"]["pi_plus"]
            numbers["skill_%s_min60_funds" % tag] = res["min60"]["funds"]
        if "sectoral" in res:
            numbers["skill_%s_sector_pi_plus" % tag] = res["sectoral"]["pi_plus"]
            numbers["skill_%s_sector_funds" % tag] = res["sectoral"]["funds"]
    top = ff[ff["percentile"] == 99].iloc[0]
    numbers["skill_ff_reg_p99_t"] = float(top["t_actual"])
    numbers["skill_ff_reg_p99_share_below"] = float(top["share_sims_below_actual"])
    q = pers.set_index("portfolio")
    numbers["skill_persist_top_alpha"] = float(q.loc["5", "alpha_ann"])
    numbers["skill_persist_top_t"] = float(q.loc["5", "t_nw"])
    numbers["skill_persist_bottom_alpha"] = float(q.loc["1", "alpha_ann"])
    numbers["skill_persist_bottom_t"] = float(q.loc["1", "t_nw"])
    numbers["skill_persist_spread_alpha"] = float(q.loc["Q5-Q1", "alpha_ann"])
    numbers["skill_persist_spread_t"] = float(q.loc["Q5-Q1", "t_nw"])
    (TABLES / "skill_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))
    print(pd.DataFrame(summaries).to_string())
    print(ff.to_string())
    print(pers.to_string())
    print(reg["by_category"].to_string())


if __name__ == "__main__":
    main()
