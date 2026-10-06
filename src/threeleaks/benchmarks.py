"""Investable benchmarks: monthly returns of the index funds an investor could actually buy.

Each index's series is the equal-weighted average return of its open-ended index funds' growth
options in that month (direct plans where they exist, which is what a cost-aware investor buys;
regular plans before 2013). ETFs are excluded because investors trade them at market prices.
"""
from __future__ import annotations

import re

import pandas as pd

INDEXES = {
    "nifty50": (re.compile(r"nifty\s*50\b(?!\s*(equal|value|arbitrage))", re.I),
                re.compile(r"next\s*50|nifty\s*500|equal|value|arbitrage|sensex|bank|it\b|50:50|shariah|dividend|"
                           r"alpha|quality|momentum|low vol|200|100", re.I)),
    "sensex": (re.compile(r"sensex", re.I), re.compile(r"next|equal|50", re.I)),
    "next50": (re.compile(r"next\s*50", re.I), re.compile(r"equal|value", re.I)),
    "midcap150": (re.compile(r"midcap\s*150", re.I), re.compile(r"momentum|quality|alpha|value|equal|small|large", re.I)),
    "smallcap250": (re.compile(r"smallcap\s*250", re.I), re.compile(r"momentum|quality|alpha|value|equal", re.I)),
    "nifty500": (re.compile(r"nifty\s*500", re.I), re.compile(r"momentum|quality|alpha|value|equal|multicap|50:25", re.I)),
}


def index_funds(schemes: pd.DataFrame) -> pd.DataFrame:
    s = schemes[schemes["passive_equity"] & schemes["is_growth"] & schemes["structure"].str.startswith("open")
                & ~schemes["name"].str.contains(r"\betf\b|exchange traded|fund of fund|\bfof\b", case=False, regex=True)]
    rows = []
    for key, (inc, exc) in INDEXES.items():
        hit = s[s["name"].str.contains(inc) & ~s["name"].str.contains(exc)]
        rows.append(hit.assign(index=key))
    return pd.concat(rows, ignore_index=True)


def index_returns(schemes: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    funds = index_funds(schemes)
    r = returns.merge(funds[["code", "index", "plan"]], on="code")
    # Direct plans from the month they exist; regular plans only for months with no direct plan.
    has_direct = r[r["plan"] == "direct"].groupby("index")["month"].min()
    keep = (r["plan"] == "direct") | r.apply(lambda x: x["index"] not in has_direct or x["month"] < has_direct[x["index"]], axis=1)
    r = r[keep]
    out = r.groupby(["month", "index"])["ret"].mean().unstack("index").sort_index()
    counts = r.groupby(["month", "index"])["code"].nunique().unstack("index")
    return out, counts, funds
