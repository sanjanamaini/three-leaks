"""Exploratory (not pre-registered): did SEBI's 2017 categorisation turn large-cap funds into
closet index funds?

    python -m threeleaks.closet

SEBI's circular of 2017-10-06 required large-cap funds to hold at least 80% in the 100 largest
stocks, implemented by mid-2018. If that squeezed out active bets, large-cap funds' tracking
error against a Nifty 50 index fund should fall after mid-2018 relative to funds whose mandate
was not tied to the largest stocks (mid- and small-cap funds). Difference-in-differences on tracking error, fund
by fund, 24 months before (2016-07 to 2018-06) vs 24 months after (2018-07 to 2020-06), with a
second post window (2022-07 to 2024-06) to see whether it persisted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .benchmarks import index_returns
from .config import INTERIM, TABLES
from .skill import fund_series

WINDOWS = {"before": ("2016-07", "2018-06"), "after": ("2018-07", "2020-06"), "later": ("2022-07", "2024-06")}
GROUPS = {"Large Cap": "treated", "Mid Cap": "control", "Small Cap": "control"}


def te_and_r2(fund: pd.Series, bench: pd.Series) -> tuple:
    both = pd.concat({"f": fund, "b": bench}, axis=1).dropna()
    if len(both) < 20:
        return np.nan, np.nan, np.nan
    d = both["f"] - both["b"]
    r2 = both.corr().iloc[0, 1] ** 2
    return float(d.std() * np.sqrt(12)), float(r2), float(d.mean() * 12)


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    returns = pd.read_parquet(INTERIM / "returns.parquet")
    wide, meta = fund_series(schemes, returns, "regular")
    idx, _, _ = index_returns(schemes, returns)
    bench = idx["nifty50"]
    rows = []
    for code in wide.columns:
        cat = meta.loc[code, "category"]
        if cat not in GROUPS:
            continue
        rec = {"code": code, "category": cat, "group": GROUPS[cat]}
        for w, (a, b) in WINDOWS.items():
            sl = wide.loc[(wide.index >= pd.Period(a, "M")) & (wide.index <= pd.Period(b, "M")), code]
            te, r2, act = te_and_r2(sl, bench)
            rec.update({"te_" + w: te, "r2_" + w: r2, "active_" + w: act})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "closet_funds.csv", index=False)
    out = {}
    for post in ("after", "later"):
        d = df.dropna(subset=["te_before", "te_" + post]).copy()
        d["dte"] = d["te_" + post] - d["te_before"]
        d["dr2"] = d["r2_" + post] - d["r2_before"]
        tr, co = d[d["group"] == "treated"], d[d["group"] == "control"]
        did = tr["dte"].mean() - co["dte"].mean()
        t = stats.ttest_ind(tr["dte"], co["dte"], equal_var=False)
        out.update({
            "closet_%s_large_n" % post: int(len(tr)), "closet_%s_control_n" % post: int(len(co)),
            "closet_%s_large_te_before" % post: float(tr["te_before"].mean()),
            "closet_%s_large_te_post" % post: float(tr["te_" + post].mean()),
            "closet_%s_control_te_before" % post: float(co["te_before"].mean()),
            "closet_%s_control_te_post" % post: float(co["te_" + post].mean()),
            "closet_%s_did_te" % post: float(did), "closet_%s_did_p" % post: float(t.pvalue),
            "closet_%s_large_r2_before" % post: float(tr["r2_before"].mean()),
            "closet_%s_large_r2_post" % post: float(tr["r2_" + post].mean()),
            "closet_%s_did_r2" % post: float(tr["dr2"].mean() - co["dr2"].mean()),
        })
    (TABLES / "closet_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
