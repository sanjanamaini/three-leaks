"""Stage 2: turn raw downloads into the clean panels every analysis uses.

    python -m threeleaks.panels

Writes to data/interim/:
  nav_monthly.parquet   code, month, nav, date (last NAV of each month)
  schemes.parquet       one row per scheme code: names, AMC, category, plan, option, fund id
  returns.parquet       code, month, ret (monthly simple return, consecutive months only)
  twins.parquet         validated direct/regular growth pairs of the same fund
  factors.parquet       IIMA monthly factors in decimals (mkt_rf, smb, hml, wml, rf)
  flows.parquet         AMFI monthly category flows and AUM (Rs crore)
and a data-quality summary to results/tables/data_quality.csv.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import amfi_monthly, amfi_nav
from .config import INTERIM, RAW, TABLES, TWIN_MIN_CORR

# --- classification rules (fixed before any return statistic was computed) -------------
ACTIVE_CATEGORIES = [
    ("Large & Mid Cap", r"large\s*&\s*mid"),
    ("Large Cap", r"large cap"),
    ("Mid Cap", r"mid cap"),
    ("Small Cap", r"small cap"),
    ("Multi Cap", r"multi cap"),
    ("Flexi Cap", r"flexi cap"),
    ("Focused", r"focused"),
    ("Value", r"value"),
    ("Contra", r"contra"),
    ("Dividend Yield", r"dividend yield"),
    ("ELSS", r"elss"),
    ("Sectoral/Thematic", r"sectoral|thematic"),
]
GROWTH = re.compile(r"growth|cumulative|appreciation", re.I)
NOT_GROWTH = re.compile(r"idcw|dividend|bonus|payout|reinvest|income distribution|\bdiv\b|-div\b|income option", re.I)
FUND_NAME_PHRASES = re.compile(r"dividend\s+yield|dividend\s+opportunit|growth\s*&\s*value", re.I)
NOT_EQUITY_LABEL = re.compile(r"hybrid|arbitrage|savings|fund of funds|fof|solution|retirement|children|"
                              r"income|debt|liquid|gilt|money market", re.I)
PASSIVE_NAME = re.compile(r"index fund|\betf\b|exchange traded|\bbees\b", re.I)
# Overseas or multi-asset products filed under domestic equity labels. "Emerging" alone is not
# here on purpose: JM Emerging Leaders and Principal Emerging Bluechip are domestic funds.
OVERSEAS = re.compile(r"global|international|overseas|\bworld\b|china|asian|\basia\b|commodit|optimix|"
                      r"multi manager|precious metal|nasdaq|\bus\b|u\.s\.|japan|europe|brazil|emerging market", re.I)
DIRECT = re.compile(r"direct", re.I)
INSTITUTIONAL = re.compile(r"institutional|\binst\b|premium plan|wholesale", re.I)
DEBT_WORDS = re.compile(r"gilt|bond|sdl|g-sec|gsec|debt|liquid|crisil|treasury|t-bill|money market|"
                        r"gold|silver|overnight|corporate|psu|target maturity|\bibx\b|cpse bond", re.I)
PLAN_TOKENS = re.compile(r"\b(direct|regular|retail|plan|option|growth|cumulative|appreciation|"
                         r"normal|fund|scheme)\b|[^a-z0-9]+", re.I)
MAX_UP, MAX_DOWN = 1.0, -0.6   # monthly NAV changes beyond these are treated as data errors
NON_ACTIVE = {"Passive", "Other", "Overseas"}  # never part of the active equity universe


def classify_category(label: str, name: str = "") -> str:
    if label is None:
        return "Other"
    low = label.lower()
    if "index" in low or "etf" in low or PASSIVE_NAME.search(name or ""):
        return "Passive"
    if NOT_EQUITY_LABEL.search(low) and "equity scheme" not in low:
        return "Other"
    if OVERSEAS.search(name or ""):
        return "Overseas"
    if low.strip() in ("growth", "elss"):
        return "Pre-2018 equity" if low.strip() == "growth" else "ELSS"
    if "equity" not in low:
        return "Other"
    for cat, pat in ACTIVE_CATEGORIES:
        if re.search(pat, low):
            return cat
    return "Other equity"


def is_growth_option(name: str) -> bool:
    core = FUND_NAME_PHRASES.sub(" ", name)
    return bool(GROWTH.search(core)) and not NOT_GROWTH.search(core)


def base_name(name: str) -> str:
    return re.sub(r"\s+", " ", PLAN_TOKENS.sub(" ", name.lower())).strip()


def build_nav() -> tuple:
    files = sorted((RAW / "amfi_nav").glob("nav_*.txt.gz"))
    nav, labels = amfi_nav.month_end_panel(files)
    return nav, labels


def build_schemes(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per scheme code, using its most recent label (AMFI labels history with current category)."""
    lab = labels.sort_values("month")
    last = lab.groupby("code").tail(1).set_index("code")
    span = lab.groupby("code")["month"].agg(first_month="min", last_month="max")
    s = last.join(span).reset_index()
    s["category"] = [classify_category(l, n) for l, n in zip(s["category_label"], s["name"])]
    s["is_growth"] = s["name"].map(is_growth_option)
    s["plan"] = np.where(s["name"].str.contains(DIRECT), "direct", "regular")
    s["institutional"] = s["name"].str.contains(INSTITUTIONAL)
    s["passive_equity"] = (s["category"] == "Passive") & ~s["name"].str.contains(DEBT_WORDS)
    # Fund identity: AMFI's own fund-level name from the scheme master where available,
    # otherwise AMC + the scheme name with plan and option words removed.
    master = sorted((RAW / "amfi_master").glob("scheme_master_*.csv"))[-1]
    m = pd.read_csv(master, dtype={"Code": str})
    m.columns = [c.strip() for c in m.columns]
    m = m.rename(columns={"Code": "code", "Scheme Name": "fund_name", "AMC": "master_amc",
                          "Launch Date": "launch_date", "Closure Date": "closure_date"})
    s = s.merge(m[["code", "fund_name", "master_amc", "launch_date", "closure_date"]], on="code", how="left")
    s["in_master"] = s["fund_name"].notna()
    s["fund_id"] = np.where(s["in_master"],
                            s["master_amc"].fillna("") + " | " + s["fund_name"].fillna(""),
                            s["amc"].fillna("") + " | " + s["name"].map(base_name))
    s["structure"] = s["structure"].fillna("")
    return s


def build_returns(nav: pd.DataFrame) -> tuple:
    nav = nav.sort_values(["code", "month"]).copy()
    prev_nav = nav.groupby("code")["nav"].shift(1)
    prev_month = nav.groupby("code")["month"].shift(1)
    consecutive = prev_month.notna() & ((nav["month"] - prev_month).apply(lambda x: getattr(x, "n", 0)) == 1)
    ret = nav["nav"] / prev_nav - 1
    ret[~consecutive] = np.nan
    bad = (ret > MAX_UP) | (ret < MAX_DOWN)
    quality = {"returns_total": int(ret.notna().sum()), "returns_flagged_as_errors": int(bad.sum())}
    ret[bad] = np.nan
    out = nav.assign(ret=ret)[["code", "month", "ret"]].dropna()
    return out, quality


def build_twins(schemes: pd.DataFrame, returns: pd.DataFrame) -> tuple:
    """Direct/regular growth options of the same open-ended fund, validated by return correlation."""
    eq = schemes[schemes["is_growth"] & ~schemes["institutional"] & schemes["structure"].str.startswith("open")
                 & ~schemes["category"].isin(NON_ACTIVE)]
    wide = returns.pivot(index="month", columns="code", values="ret")
    rows, rejected = [], []
    for fid, g in eq.groupby("fund_id"):
        d, r = g[g["plan"] == "direct"], g[g["plan"] == "regular"]
        if len(d) != 1 or len(r) == 0:
            continue
        dc = d["code"].iloc[0]
        # If a fund has several regular growth codes (retail/normal), keep the one with the longest history.
        r = r.assign(n=r["code"].map(lambda c: wide[c].notna().sum() if c in wide else 0)).sort_values("n")
        rc = r["code"].iloc[-1]
        if dc not in wide or rc not in wide:
            continue
        both = wide[[dc, rc]].dropna()
        if len(both) < 12:
            continue
        corr = both[dc].corr(both[rc])
        row = {"fund_id": fid, "direct_code": dc, "regular_code": rc, "category": d["category"].iloc[0],
               "overlap_months": len(both), "corr": corr, "first_month": both.index.min(),
               "last_month": both.index.max()}
        (rows if corr >= TWIN_MIN_CORR else rejected).append(row)
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def build_factors() -> pd.DataFrame:
    f = pd.read_csv(RAW / "iima" / "2025-12_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv")
    f["month"] = pd.PeriodIndex(f["Date"], freq="M")
    f = f.rename(columns={"MF": "mkt_rf", "SMB": "smb", "HML": "hml", "WML": "wml", "RF": "rf"})
    for c in ["mkt_rf", "smb", "hml", "wml", "rf"]:
        f[c] = pd.to_numeric(f[c], errors="coerce") / 100.0
    return f[["month", "mkt_rf", "smb", "hml", "wml", "rf"]].dropna()


def build_flows() -> pd.DataFrame:
    files = sorted((RAW / "amfi_monthly").glob("am*"))
    return amfi_monthly.panel(files)


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    nav, labels = build_nav()
    schemes = build_schemes(labels)
    returns, quality = build_returns(nav)
    twins, rejected = build_twins(schemes, returns)
    factors = build_factors()
    flows = build_flows()

    nav.to_parquet(INTERIM / "nav_monthly.parquet", index=False)
    labels.to_parquet(INTERIM / "labels.parquet", index=False)
    schemes.to_parquet(INTERIM / "schemes.parquet", index=False)
    returns.to_parquet(INTERIM / "returns.parquet", index=False)
    twins.to_parquet(INTERIM / "twins.parquet", index=False)
    rejected.to_parquet(INTERIM / "twins_rejected.parquet", index=False)
    factors.to_parquet(INTERIM / "factors.parquet", index=False)
    flows.to_parquet(INTERIM / "flows.parquet", index=False)

    eq = schemes[schemes["structure"].str.startswith("open") & ~schemes["category"].isin(NON_ACTIVE)]
    quality.update({
        "nav_months": int(nav["month"].nunique()),
        "scheme_codes_all": int(schemes["code"].nunique()),
        "open_equity_codes": int(len(eq)),
        "open_equity_growth_codes": int(eq["is_growth"].sum()),
        "open_equity_codes_not_in_master": int((~eq["in_master"]).sum()),
        "twins_accepted": int(len(twins)),
        "twins_rejected_low_corr": int(len(rejected)),
        "factor_months": int(len(factors)),
        "flow_months": int(flows["month"].nunique()),
    })
    pd.Series(quality).to_csv(TABLES / "data_quality.csv", header=["value"])
    print(pd.Series(quality).to_string())


if __name__ == "__main__":
    main()
