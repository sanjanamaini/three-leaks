# Data sources and licences

No raw data is committed. `make data` downloads everything from the original publishers into `data/raw/` (git-ignored) and records a SHA-256 checksum of every file in `data/raw/MANIFEST.csv`.

| Source | What | Access | Terms |
|---|---|---|---|
| Association of Mutual Funds in India (AMFI), NAV history | daily NAVs of every scheme, including closed and merged schemes, with AMFI category headers | `https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx` (public, no login) | published for public information; no reuse licence found, so raw files are not redistributed |
| AMFI monthly reports | category-level gross sales, redemptions, net flows, folios and AUM (Rs crore) | `https://portal.amfiindia.com/spages/am<mon><year>repo.xls` | as above |
| IIMA Indian Fama-French-Momentum data library | monthly market, size, value and momentum factors and the risk-free rate, survivorship-bias-adjusted release | `https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/` | free; cite Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013), *Four factor model in Indian equities market*, IIMA Working Paper 2013-09-05 |
| AMFI scheme master | every scheme's fund-level name (shared by all its plans), category, launch and closure dates | `https://portal.amfiindia.com/DownloadSchemeData_Po.aspx?mf=0` | as above |
| AMFI distributor commission disclosure, FY 2024-25 | gross commission (incl. GST) and average AUM per top distributor | `https://portal.amfiindia.com/spages/ACDVol-2024-2025.pdf` | as above; used only as an external check |
| AMFI Annual Report FY 2025 (with Crisil) | direct plans' share of SIP AUM (12% in March 2020, 21% in March 2025) | `https://www.amfiindia.com/Themes/Theme1/downloads/AMFI_AnnualMFReport2025.pdf` | cited, not downloaded by the pipeline |
| mfapi.in | per-scheme NAV history and metadata (a mirror of AMFI data) | `https://api.mfapi.in/mf/<code>` | open-source API (MIT code); used only to cross-check AMFI parsing |

Derived results in `results/` (aggregates, model outputs, figures) are published under the repository licence.
