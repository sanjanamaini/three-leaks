"""Stage 6. Can the leaks be avoided in advance? Out-of-sample fund selection, simple rules vs ML.

    python -m threeleaks.selection

Each December a rule picks funds using only information dated on or before that day; picks
are held for the next 12 months (equal-weighted; funds that close drop out). Rules: low
R-squared (Amihud and Goyenko 2013), high prior alpha, ridge regression and gradient-boosted
trees (DeMiguel et al. 2023) trained on an expanding window of earlier years. Baselines: all
eligible funds, and the Nifty 50 index funds. Outcome: four-factor alpha of each strategy's
monthly returns, Holm-corrected across the four selection rules (PREREGISTRATION.md, RQ4).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .benchmarks import index_returns
from .config import FACTOR_END, FIGURES, INTERIM, NW_LAGS, SEED, TABLES
from .models import FACTORS, design, fit
from .skill import DIVERSIFIED_EXCLUDE, align, fund_series

LOOKBACK = 36
FEATURES = ["alpha36", "t36", "r2_36", "idio36", "b_mkt_rf", "b_smb", "b_hml", "b_wml",
            "ret12", "ret12_rel", "age", "twin_gap12"]
RULES = ["low_r2", "high_alpha", "ridge", "gbt"]


def features_and_labels(excess, F, raw, meta, twin_gap) -> pd.DataFrame:
    rows = []
    first_month = raw.apply(lambda c: c.first_valid_index())
    for y in range(2009, 2025):
        end = pd.Period("%d-12" % y, "M")
        hist = excess.loc[(excess.index > end - LOOKBACK) & (excess.index <= end)]
        if len(hist) < LOOKBACK:
            continue
        fut = excess.loc[(excess.index > end) & (excess.index <= end + 12)]
        Ff = F.reindex(fut.index)
        for c in hist.columns:
            if meta.loc[c, "category"] in DIVERSIFIED_EXCLUDE:
                continue
            h = hist[c]
            if h.notna().sum() < LOOKBACK:
                continue
            res = fit(h.to_numpy(), design(F.loc[h.index]), NW_LAGS)
            ret12 = float(np.prod(1 + raw.loc[(raw.index > end - 12) & (raw.index <= end), c].fillna(0)) - 1)
            row = {"code": c, "year": y, "category": meta.loc[c, "category"], "fund_id": meta.loc[c, "fund_id"],
                   "alpha36": res["alpha"], "t36": res["t"], "r2_36": res["r2"], "idio36": res["resid_sd"],
                   **{"b_" + f: b for f, b in zip(FACTORS, res["betas"])},
                   "ret12": ret12, "age": (end - first_month[c]).n if first_month[c] is not None else np.nan,
                   "twin_gap12": twin_gap.get((meta.loc[c, "fund_id"], y), np.nan)}
            # Label: mean monthly abnormal return over the next 12 months, betas fixed from the lookback.
            fy = fut[c].dropna()
            if len(fy) >= 6 and Ff.loc[fy.index, FACTORS].notna().all().all():
                ab = fy.to_numpy() - Ff.loc[fy.index, FACTORS].to_numpy() @ res["betas"]
                row["label"] = float(ab.mean())
            else:
                row["label"] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df["ret12_rel"] = df["ret12"] - df.groupby(["year", "category"])["ret12"].transform("median")
    return df


def twin_gaps() -> dict:
    p = pd.read_parquet(INTERIM / "twin_panel.parquet")
    p["year"] = p["month"].dt.year
    g = p.groupby(["fund_id", "year"])["gap_log"].agg(["count", "mean"])
    g = g[g["count"] >= 6]
    return {k: float(np.exp(12 * v) - 1) for k, v in g["mean"].items()}


def pick(df_year: pd.DataFrame, train: pd.DataFrame, rule: str) -> list:
    d = df_year
    q = max(1, len(d) // 5)
    if rule == "low_r2":
        return d.nsmallest(q, "r2_36")["fund_id"].tolist()
    if rule == "high_alpha":
        return d.nlargest(q, "alpha36")["fund_id"].tolist()
    tr = train.dropna(subset=["label"])
    if len(tr) < 100:
        return []
    X_tr, y_tr = tr[FEATURES], tr["label"]
    if rule == "ridge":
        med = X_tr.median().fillna(0.0)   # a feature with no history yet (twin gap before 2013) is filled with 0
        sc = StandardScaler().fit(X_tr.fillna(med))
        m = Ridge(alpha=1.0).fit(sc.transform(X_tr.fillna(med)), y_tr)
        pred = m.predict(sc.transform(d[FEATURES].fillna(med)))
    else:
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=SEED)
        m.fit(X_tr, y_tr)
        pred = m.predict(d[FEATURES])
    return d.assign(pred=pred).nlargest(q, "pred")["fund_id"].tolist()


def strategy_returns(feat: pd.DataFrame, eval_wide: pd.DataFrame, eval_meta: pd.DataFrame, first_year: int) -> pd.DataFrame:
    """Monthly returns of each strategy, evaluated on the plan whose returns are in eval_wide."""
    code_of = eval_meta.reset_index().set_index("fund_id")["code"]
    out = {k: [] for k in RULES + ["all_funds"]}
    for y in range(first_year, 2025):
        d = feat[feat["year"] == y]
        d = d[d["fund_id"].isin(code_of.index)]          # must exist in the evaluated plan
        alive = [f for f in d["fund_id"] if code_of[f] in eval_wide.columns
                 and eval_wide.loc[pd.Period("%d-12" % y, "M"), code_of[f]] == eval_wide.loc[pd.Period("%d-12" % y, "M"), code_of[f]]]
        d = d[d["fund_id"].isin(alive)]
        if len(d) < 20:
            continue
        train = feat[feat["year"] <= y - 1]
        nxt = eval_wide.loc[(eval_wide.index > pd.Period("%d-12" % y, "M")) & (eval_wide.index <= pd.Period("%d-12" % y, "M") + 12)]
        out["all_funds"].append(nxt[[code_of[f] for f in d["fund_id"]]].mean(axis=1))
        for rule in RULES:
            chosen = pick(d, train, rule)
            if chosen:
                out[rule].append(nxt[[code_of[f] for f in chosen]].mean(axis=1))
    return pd.DataFrame({k: pd.concat(v) for k, v in out.items() if v}).sort_index()


def evaluate(rets: pd.DataFrame, F: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    f = F.reindex(rets.index)
    for col in rets.columns:
        y = rets[col] - f["rf"]
        ok = y.notna() & f["mkt_rf"].notna()
        res = fit(y[ok].to_numpy(), design(f[ok]), NW_LAGS)
        row = {"plan": label, "strategy": col, "months": int(ok.sum()), "mean_excess_ann": float(y[ok].mean() * 12),
               "alpha_ann": res["alpha"] * 12, "t_nw": res["t_nw"]}
        if col in RULES and "all_funds" in rets:
            d = (rets[col] - rets["all_funds"])[ok]
            rd = fit(d.to_numpy(), design(f[ok]), NW_LAGS)
            row.update({"alpha_vs_all_ann": rd["alpha"] * 12, "t_vs_all": rd["t_nw"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    from scipy import stats
    out["p"] = 2 * (1 - stats.norm.cdf(out["t_nw"].abs()))
    sel = out["strategy"].isin(RULES)
    out.loc[sel, "p_holm"] = multipletests(out.loc[sel, "p"], method="holm")[1]
    return out


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    factors = pd.read_parquet(INTERIM / "factors.parquet")
    reg_wide, reg_meta = fund_series(schemes, returns, "regular")
    dir_wide, dir_meta = fund_series(schemes, returns, "direct")
    excess, F = align(reg_wide, factors)
    feat = features_and_labels(excess, F, reg_wide, reg_meta, twin_gaps())
    feat.to_parquet(INTERIM / "selection_features.parquet", index=False)

    end = pd.Period(FACTOR_END, "M")
    dir_wide = dir_wide[dir_wide.index <= end]
    reg_eval = reg_wide[reg_wide.index <= end]
    idx, counts, funds = index_returns(schemes, returns)
    idx.to_csv(TABLES / "benchmark_index_returns.csv")

    d = strategy_returns(feat, dir_wide, dir_meta.reset_index().set_index("code"), 2013)
    d["nifty50_index"] = idx["nifty50"].reindex(d.index)
    r = strategy_returns(feat, reg_eval, reg_meta.reset_index().set_index("code"), 2009)
    r["nifty50_index"] = idx["nifty50"].reindex(r.index)
    perf = pd.concat([evaluate(d, F, "direct"), evaluate(r, F, "regular")], ignore_index=True)
    perf.to_csv(TABLES / "selection_performance.csv", index=False)
    d.to_csv(TABLES / "selection_returns_direct.csv")

    # Cumulative growth of Rs 1, direct plans, 2014-2025.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for col, colr in zip(d.columns, ["#1F3A5F", "#3B7DD8", "#7A4EAB", "#C9A227", "#888888", "#B33"]):
        ax.plot(d.index.to_timestamp(), (1 + d[col].fillna(0)).cumprod(), label=col.replace("_", " "), color=colr, lw=1.2)
    ax.set_ylabel("Growth of Rs 1 (direct plans)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "selection_growth.png", dpi=200)
    plt.close(fig)

    numbers = {}
    for _, row in perf.iterrows():
        k = "sel_%s_%s" % (row["plan"][:3], row["strategy"])
        numbers[k + "_alpha"] = float(row["alpha_ann"])
        numbers[k + "_t"] = float(row["t_nw"])
        if pd.notna(row.get("p_holm")):
            numbers[k + "_pholm"] = float(row["p_holm"])
        if pd.notna(row.get("alpha_vs_all_ann")):
            numbers[k + "_vsall"] = float(row["alpha_vs_all_ann"])
            numbers[k + "_vsall_t"] = float(row["t_vs_all"])
    numbers["sel_feature_rows"] = int(len(feat))
    numbers["sel_years_direct"] = "%s to %s" % (d.index.min(), d.index.max())
    (TABLES / "selection_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(perf.to_string())


if __name__ == "__main__":
    main()
