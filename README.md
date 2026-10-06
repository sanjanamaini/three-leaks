# The Three Leaks

**Where do Indian equity mutual fund investors lose their returns: to commissions, to luck dressed up as skill, or to their own timing? And can any of it be avoided in advance?**

India offers a design almost no other market does. Since January 2013 every fund has a direct plan and a regular plan with the same portfolio and the same manager; the only difference is the distributor's commission. This project uses that, plus every scheme in AMFI's NAV history (alive and dead, 2006 to 2026), the IIMA four-factor library and AMFI's flow data, to measure all three leaks on one dataset, with every test pre-registered.

**Report:** [`report/report.pdf`](report/report.pdf) (11 pages). **Scorecard:** [`docs/index.html`](docs/index.html) (one offline page, 521 live funds).

## What it finds

- **The commission leak is universal and growing.** In all 622 twin pairs, the direct plan beat the regular plan of the same fund, by a median 1.2% a year. The median yearly gap rose from 0.57% in 2013 to 1.24% in 2025. SEBI's April 2019 expense-ratio cut narrowed it by 0.11 points (t = -5.0, placebo dates move the other way or not at all), but only for about a year.
- **Skill is real, but the commission takes most of it.** After the Barras, Scaillet and Wermers false-discovery correction, 29% of regular plans and 60% of direct plans have truly positive four-factor alpha; almost none have negative alpha. The best funds beat luck in a Fama-French bootstrap, and past winners keep a modest edge (top-minus-bottom quintile 2.67% a year, t = 3.61). Against index funds an investor can actually buy (2021 to 2026), the median direct plan earned 1.35% a year and the median regular plan only 0.34%.
- **The timing leak is mostly not timing.** Investors' money-weighted return trailed their funds by 0.65 points a year (2019 to 2026). Shuffling flows across all months says investors timed badly, but flows grew strongly over the period; shuffling within each year says their timing was no worse than random (p = 0.20). Flows even lean against recent returns.
- **Some of it could have been avoided.** Out of sample (2014 to 2025), rules built only from past data picked direct plans that beat the average fund: gradient-boosted trees by 2.37% a year (t = 3.56), high prior alpha by 1.71%. The edge halved after 2019.
- **Together (2019 to 2025):** the commission is the largest leak, about Rs 24,000 to 35,500 crore a year at 2025 assets; timing costs about Rs 9,200 crore; skill was a net gain.
- **Exploratory:** the weaker alpha after 2019 is mostly 2020 (-4.6% that year) and older funds (1.1% against 3.3% for newer funds in 2020 to 2025). SEBI's 2017 categorisation did not make large-cap funds more index-like.

## The scorecard

For each of the 521 live active equity funds with both plans: what the regular plan costs a year, the direct plan's alpha with its false-discovery q-value, an expected alpha after discounting luck, the chance that alpha is positive, R-squared, a verdict and the cheaper index alternative. The verdict rule was fixed before any fund's result was read. 154 funds show evidence of skill, none shows evidence of underperformance, 89 are indistinguishable from luck; sector funds get no verdict.

## Reproduce everything

Needs Python 3.9 or later, `pdflatex` and an internet connection.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all        # download from AMFI and IIMA, build panels, run every analysis, scorecard, report
make test       # unit tests
```

The AMFI downloads are slow (about 250 monthly NAV files). After they are on disk, `make panels analysis report` rebuilds every number in about three minutes, and a rerun reproduces every number exactly. Every number in the report is written by the pipeline into `results/numbers.json` and `report/numbers.tex`.

## How it works

| Idea | Source | Used for |
|---|---|---|
| Twin plans as a natural experiment | India's 2013 direct-plan rule | the price of distribution, fund by fund |
| Event study with placebo dates | SEBI's 2018 and 2019 rule changes | whether regulation narrowed the gap |
| Four-factor alpha | Carhart (1997); IIMA factors (Agarwalla, Jacob and Varma 2013) | skill after fees |
| False discovery rate for funds | Barras, Scaillet and Wermers (2010); Storey (2002) | how many funds have real skill |
| Cross-sectional bootstrap | Fama and French (2010) | whether the best funds beat luck |
| Investable benchmarks | Sharpe (1992); Berk and van Binsbergen (2015) | skill against funds an investor can buy |
| Money-weighted returns | Dichev (2007); Friesen and Sapp (2007) | the timing leak |
| Fund selection with machine learning | Amihud and Goyenko (2013); DeMiguel et al. (2023) | whether leaks can be avoided in advance |

## Layout

| Path | What |
|---|---|
| `PREREGISTRATION.md` | the plan, fixed before any return statistic was computed |
| `DEVIATIONS.md` | every clarification, addition and correction, dated; additions made after results are marked |
| `DATA.md` | sources and licences |
| `src/threeleaks/` | the pipeline, one module per stage |
| `results/` | derived tables, figures and `numbers.json` |
| `report/` | the LaTeX report, its generated numbers and tables, the PDF |
| `tool/`, `docs/` | the scorecard builder and the scorecard |
| `tests/` | unit tests |

## Data and licence

Raw data is never committed: `make data` fetches it from AMFI and IIMA (see `DATA.md`). Code and derived aggregates: MIT licence.
