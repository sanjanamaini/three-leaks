"""Parse AMFI NAV-history files into a month-end NAV panel.

Each raw file holds every scheme's daily NAVs for the last 7 calendar days of one month,
grouped under section headers ("Open Ended Schemes ( Equity Scheme - Large Cap Fund )")
and AMC names. AMFI labels each scheme with its *current* category, so a scheme alive
today carries its SEBI 2017 category in every year; schemes that closed earlier keep the
old labels ("Growth", "ELSS", "Income").
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

SECTION = re.compile(r"^(Open Ended|Close Ended|Interval Fund)\s*Schemes?\s*\(\s*(.*?)\s*\)\s*$", re.I)


def parse_file(path: Path) -> pd.DataFrame:
    """Rows: code, name, nav, date, structure (open/close/interval), category_label, amc."""
    text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    lines = text.splitlines()
    header = [h.strip().lower() for h in lines[0].split(";")]
    i_code = header.index("scheme code")
    i_name = next(i for i, h in enumerate(header) if "name" in h)
    i_nav = header.index("net asset value")
    i_date = header.index("date")
    structure, label, amc = None, None, None
    out = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if ";" not in line:
            m = SECTION.match(line)
            if m:
                structure, label = m.group(1).lower(), m.group(2)
            else:
                amc = line
            continue
        f = line.split(";")
        if len(f) <= max(i_code, i_nav, i_date):
            continue
        try:
            nav = float(f[i_nav])
        except ValueError:
            continue  # "N.A.", "#N/A", blanks
        if not np.isfinite(nav) or nav <= 0:
            continue
        out.append((f[i_code].strip(), f[i_name].strip(), nav, f[i_date].strip(), structure, label, amc))
    df = pd.DataFrame(out, columns=["code", "name", "nav", "date", "structure", "category_label", "amc"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["date"])


def month_end_panel(files: list) -> tuple:
    """Last available NAV of each scheme in each month, plus every label a scheme carried."""
    navs, labels = [], []
    for p in sorted(files):
        month = pd.Period(p.name[4:11], freq="M")
        df = parse_file(p)
        df = df[df["date"].dt.to_period("M") == month]
        last = df.sort_values("date").groupby("code", as_index=False).tail(1)
        last = last.assign(month=month)
        navs.append(last[["code", "month", "nav", "date"]])
        labels.append(last[["code", "month", "name", "structure", "category_label", "amc"]])
    return pd.concat(navs, ignore_index=True), pd.concat(labels, ignore_index=True)
