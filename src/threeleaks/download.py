"""Stage 1: download every raw input from its publisher into data/raw/ (never committed).

    python -m threeleaks.download

Idempotent: a file already on disk and valid is not downloaded again. Every file's SHA-256
goes into data/raw/MANIFEST.csv. Requests are sequential with a pause between them.
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
import gzip
import hashlib
import re
import sys
import time

import requests

from .config import NAV_END, NAV_START, RAW, USER_AGENT

NAV_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx?frmdt={a}&todt={b}"
MONTHLY_PAGE = "https://www.amfiindia.com/research-information/amfi-monthly"
IIMA_BASE = "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
IIMA_FILES = ["DATA/2025-12_FourFactors_and_Market_Returns_Monthly_SurvivorshipBiasAdjusted.csv",
              "DATA/2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv",
              "four-factors-India-90s-onwards-IIM-WP-Version.pdf"]
PAUSE = 1.5
SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


def get(url: str, tries: int = 4, timeout: int = 180) -> bytes:
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            err = "HTTP %d" % r.status_code
        except requests.RequestException as e:  # network hiccup: back off and retry
            err = str(e)
        time.sleep(PAUSE * (2 ** k))
    raise RuntimeError("failed after %d tries: %s (%s)" % (tries, url, err))


def month_ends(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield dt.date(y, m, calendar.monthrange(y, m)[1])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fmt(d: dt.date) -> str:
    return d.strftime("%d-%b-%Y")


def nav_history(reverse: bool = False) -> list:
    """One file per month: every scheme's NAVs in the last 7 calendar days of the month.

    reverse=True walks from the latest month backwards, so a second process can fill the
    archive from the other end; each skips files already on disk.
    """
    out = RAW / "amfi_nav"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    months = list(month_ends(NAV_START, NAV_END))
    for me in (reversed(months) if reverse else months):
        path = out / ("nav_%s.txt.gz" % me.strftime("%Y-%m"))
        files.append(path)
        if path.exists() and path.stat().st_size > 1000:
            continue
        url = NAV_URL.format(a=fmt(me - dt.timedelta(days=6)), b=fmt(me))
        body = get(url)
        if not body.startswith(b"Scheme Code;"):
            raise RuntimeError("unexpected content for %s: %r" % (me, body[:80]))
        path.write_bytes(gzip.compress(body))
        print("nav", me.strftime("%Y-%m"), "%.1f MB" % (len(body) / 1e6), flush=True)
        time.sleep(PAUSE)
    return files


def monthly_reports() -> list:
    """AMFI monthly category reports (Excel), every month the AMFI page lists."""
    out = RAW / "amfi_monthly"
    out.mkdir(parents=True, exist_ok=True)
    page = get(MONTHLY_PAGE).decode("utf-8", "replace")
    links = sorted(set(re.findall(r'https://portal\.amfiindia\.com/spages/am[a-z]{3}\d{4}repo(?:revised)?\.xls', page)))
    files = []
    for url in links:
        stem = url.rsplit("/", 1)[1].rsplit(".", 1)[0]
        done = [p for p in (out / (stem + ".xls"), out / (stem + ".xlsx")) if p.exists() and p.stat().st_size > 1000]
        if done:
            files.append(done[0])
            continue
        body = get(url)
        # AMFI serves both legacy .xls (OLE2) and .xlsx (zip) files under a .xls name; keep the true type.
        if body.startswith(b"\xd0\xcf\x11\xe0"):
            path = out / (stem + ".xls")
        elif body.startswith(b"PK\x03\x04"):
            path = out / (stem + ".xlsx")
        else:
            print("skipped (not a spreadsheet):", url, flush=True)
            continue
        path.write_bytes(body)
        files.append(path)
        print("monthly", path.name, flush=True)
        time.sleep(PAUSE)
    return files


def iima() -> list:
    out = RAW / "iima"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for rel in IIMA_FILES:
        path = out / rel.rsplit("/", 1)[-1]
        files.append(path)
        if not (path.exists() and path.stat().st_size > 1000):
            path.write_bytes(get(IIMA_BASE + rel))
            print("iima", path.name, flush=True)
            time.sleep(PAUSE)
    return files


def scheme_master() -> list:
    """AMFI scheme master: fund-level name shared by every plan, category, launch and closure dates."""
    out = RAW / "amfi_master"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("scheme_master_%s.csv" % dt.date.today().strftime("%Y-%m-%d"))
    existing = sorted(out.glob("scheme_master_*.csv"))
    if existing:
        return [existing[-1]]
    body = get("https://portal.amfiindia.com/DownloadSchemeData_Po.aspx?mf=0")
    if not body.startswith(b"AMC,Code,"):
        raise RuntimeError("unexpected scheme master content: %r" % body[:80])
    path.write_bytes(body)
    print("master", path.name, flush=True)
    return [path]


def distributor_commissions() -> list:
    """AMFI's annual disclosure of commissions paid to each top distributor (FY 2024-25)."""
    out = RAW / "amfi_commissions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ACDVol-2024-2025.pdf"
    if not (path.exists() and path.stat().st_size > 1000):
        body = get("https://portal.amfiindia.com/spages/ACDVol-2024-2025.pdf")
        if not body.startswith(b"%PDF"):
            raise RuntimeError("commission disclosure is not a PDF")
        path.write_bytes(body)
        print("commissions", path.name, flush=True)
    return [path]


def manifest(paths: list) -> None:
    rows = []
    for p in sorted(paths):
        data = p.read_bytes()
        rows.append({"file": str(p.relative_to(RAW)), "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    with open(RAW / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "bytes", "sha256", "modified"])
        w.writeheader()
        w.writerows(rows)
    print("manifest: %d files" % len(rows))


def main() -> None:
    if "--nav-reverse" in sys.argv:
        nav_history(reverse=True)
        return
    files = iima() + scheme_master() + distributor_commissions() + monthly_reports() + nav_history()
    manifest(files)


if __name__ == "__main__":
    sys.exit(main())
