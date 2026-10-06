"""Independent check of the parsed NAV panel against mfapi.in (a separate mirror of AMFI data).

    python -m threeleaks.validate

A fixed random sample of equity growth schemes; for each, every month-end NAV in our panel is
compared with mfapi's NAV on the same date. Reports the share of matches within 0.01%.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from .config import INTERIM, SEED, TABLES, USER_AGENT
from .panels import NON_ACTIVE

SAMPLE = 30


def main() -> None:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    nav = pd.read_parquet(INTERIM / "nav_monthly.parquet")
    pool = schemes[schemes["is_growth"] & schemes["structure"].str.startswith("open")
                   & ~schemes["category"].isin(NON_ACTIVE)]["code"].sort_values()
    codes = pool.sample(SAMPLE, random_state=SEED % (2 ** 32)).tolist()
    rows = []
    for c in codes:
        try:
            r = requests.get("https://api.mfapi.in/mf/%s" % c, headers={"User-Agent": USER_AGENT}, timeout=60)
            data = r.json().get("data", [])
        except (requests.RequestException, ValueError):
            data = []
        time.sleep(0.5)
        if not data:
            rows.append({"code": c, "checked": 0, "matched": 0, "status": "no data at mfapi"})
            continue
        m = pd.DataFrame(data)
        m["date"] = pd.to_datetime(m["date"], format="%d-%m-%Y")
        m["nav"] = pd.to_numeric(m["nav"], errors="coerce")
        ours = nav[nav["code"] == c][["date", "nav"]]
        j = ours.merge(m, on="date", suffixes=("_ours", "_mfapi")).dropna()
        ok = (np.abs(j["nav_ours"] / j["nav_mfapi"] - 1) <= 1e-4)
        rows.append({"code": c, "checked": int(len(j)), "matched": int(ok.sum()), "status": "ok"})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "validation_mfapi.csv", index=False)
    checked, matched = int(df["checked"].sum()), int(df["matched"].sum())
    numbers = {"validate_schemes": int((df["checked"] > 0).sum()), "validate_navs_checked": checked,
               "validate_navs_matched": matched, "validate_match_share": matched / checked if checked else None}
    (TABLES / "validate_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(df.to_string())
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
