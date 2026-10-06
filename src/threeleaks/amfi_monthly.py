"""Parse AMFI monthly reports into a category-level flow and AUM panel (Rs crore).

Two layouts exist:
- 2018-04 to 2019-03: sheets "sales" and "red+assets", with only two equity rows
  ("Equity Schemes (Excluding ELSS)" and "ELSS - Equity"); net flow = sales - redemptions.
- 2019-04 onward: one sheet with a row per SEBI category and a subtotal per section.
Only open-ended schemes are kept. Column positions are found from header text, not fixed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                      "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}


def file_month(path: Path) -> pd.Period:
    m = re.search(r"am([a-z]{3})(\d{4})", path.name)
    return pd.Period(year=int(m.group(2)), month=MONTHS[m.group(1)], freq="M")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _text(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else ""


def parse_new(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    cols = {}
    for _, row in df.iterrows():
        cells = [_text(c).lower() for c in row]
        if any("folio" in c for c in cells) and any("redemption" in c for c in cells):
            for j, c in enumerate(cells):
                if "no. of schemes" in c or c.startswith("no. of scheme"):
                    cols["schemes"] = j
                elif "folio" in c:
                    cols["folios"] = j
                elif "mobili" in c:
                    cols["inflow"] = j
                elif "redemption" in c:
                    cols["redemption"] = j
                elif "net inflow" in c:
                    cols["net_inflow"] = j
                elif ("net assets" in c or c.startswith("aum as on")) and "average" not in c:
                    cols["aum"] = j
                elif "average" in c or c.startswith("aaum"):
                    cols["aaum"] = j
            break
    need = {"inflow", "redemption", "net_inflow", "aum"}
    if not need <= set(cols):
        raise ValueError("%s: header columns not found (%s)" % (path.name, cols))
    structure, section, out = None, None, []
    for _, row in df.iterrows():
        a, b = _text(row.iloc[0]), _text(row.iloc[1])
        if a in ("A", "B", "C") and b:
            structure = b.lower()
            continue
        if a in ROMAN and b:
            section = b
            continue
        label = b if b else a
        if not label or structure is None or not structure.startswith("open"):
            continue
        vals = {k: _num(row.iloc[j]) for k, j in cols.items()}
        if vals["aum"] is None or vals["inflow"] is None:
            continue
        is_sub = label.lower().startswith("sub total")
        out.append({"section": section, "category": "_subtotal" if is_sub else label, **vals})
    res = pd.DataFrame(out)
    res["month"] = file_month(path)
    res["layout"] = "category"
    return res


def _open_end_rows(df: pd.DataFrame, start_label: str, labels: tuple, value_col: int) -> dict:
    """Values in one column for the given row labels, searching below the row matching start_label."""
    start = None
    for i, row in df.iterrows():
        if start_label.lower() in _text(row.iloc[0]).lower():
            start = i
            break
    out = {}
    for i in range(start if start is not None else 0, len(df)):
        lab = _text(df.iloc[i, 0]).lower()
        for want in labels:
            if lab.startswith(want.lower()) and want not in out:
                cells = [c for c in df.iloc[i, 1:].tolist() if pd.notna(c) and _num(c) is not None]
                out[want] = _num(cells[value_col]) if len(cells) > value_col else None
    return out


def parse_old(path: Path) -> pd.DataFrame:
    labels = ("Equity Schemes", "ELSS - Equity")
    sales = pd.read_excel(path, sheet_name="sales", header=None)
    red = pd.read_excel(path, sheet_name="red+assets", header=None)
    s = _open_end_rows(sales, "2.3", labels, value_col=1)        # open-end amount, all schemes
    r = _open_end_rows(red, "REDEMPTIONS", labels, value_col=0)  # open-end redemptions
    a = _open_end_rows(red, "ASSETS UNDER", labels, value_col=0) # open-end AUM
    rows = []
    for lab, name in zip(labels, ("Equity (excluding ELSS)", "ELSS")):
        rows.append({"section": "Equity", "category": name, "inflow": s.get(lab),
                     "redemption": r.get(lab), "aum": a.get(lab)})
    res = pd.DataFrame(rows)
    res["net_inflow"] = res["inflow"] - res["redemption"]
    sub = res[["inflow", "redemption", "aum", "net_inflow"]].sum().to_dict()
    res = pd.concat([res, pd.DataFrame([{"section": "Equity", "category": "_subtotal", **sub}])], ignore_index=True)
    res["month"] = file_month(path)
    res["layout"] = "two-row"
    return res


def parse(path: Path) -> pd.DataFrame:
    sheets = pd.ExcelFile(path).sheet_names
    return parse_old(path) if "sales" in sheets else parse_new(path)


def panel(files: list) -> pd.DataFrame:
    """All months stacked; where a revised file exists for a month, it replaces the original."""
    by_month = {}
    for p in sorted(files, key=lambda p: ("revised" in p.name, p.name)):
        by_month[file_month(p)] = p
    frames = [parse(p) for _, p in sorted(by_month.items())]
    return pd.concat(frames, ignore_index=True)
