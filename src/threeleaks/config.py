"""Paths, dates and fixed choices shared by every stage (see PREREGISTRATION.md)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "report"

# Sample windows (pre-registered)
NAV_START = "2006-04"          # first month of AMFI NAV history used
NAV_END = "2026-09"            # last complete month at project start
FACTOR_END = "2025-12"         # last month in the IIMA release used
DIRECT_PLANS_START = "2013-01" # SEBI direct plans launched 2013-01-01

# Fixed analysis choices (pre-registered)
MIN_MONTHS_ALPHA = 36          # minimum monthly observations for a fund alpha
TWIN_MIN_CORR = 0.98           # twin plans must have return correlation >= this
NW_LAGS = 3                    # Newey-West lags for the HAC check
N_BOOT = 2000                  # Fama-French (2010) bootstrap simulations
N_PERM = 10000                 # timing-test permutations
SEED = 20261006                # every random draw uses this seed

TER_EVENT = "2019-04-01"       # SEBI TER cut (primary natural experiment)
UPFRONT_BAN_EVENT = "2018-10-22"
PLACEBO_EVENTS = ("2016-04-01", "2022-04-01")

USER_AGENT = "three-leaks research script (open-source; contact via GitHub)"
