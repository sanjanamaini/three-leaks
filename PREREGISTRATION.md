# Pre-registration: The Three Leaks

Written and committed on 2026-10-06, before any fund return, alpha, gap or flow statistic was computed. The only data inspected so far: file formats and headers (AMFI NAV history, AMFI monthly report layout, IIMA factor file header). Every later deviation from this plan is recorded in `DEVIATIONS.md` with its date and reason.

## The question

An Indian investor who buys an actively managed equity mutual fund can lose return, relative to a cheap index alternative, in three separate places:

1. **The commission leak:** a regular plan carries a distributor commission that the direct plan of the *same* fund does not.
2. **The skill leak:** after fees, the manager may not beat the market once risk and luck are accounted for.
3. **The timing leak:** investors who add money after good months and withdraw after bad ones earn less than the fund itself.

The project measures each leak on one open, survivorship-free dataset, tests whether any of them can be avoided by selecting funds in advance, and turns the results into a scorecard an investor or platform can use.

## Data (fixed in advance)

| Source | Use | Period |
|---|---|---|
| AMFI NAV history (portal.amfiindia.com) | month-end NAVs of every scheme, alive or dead | 2006-04 to 2026-09 |
| AMFI monthly reports (`am<mon><year>repo.xls`) | category net flows and AUM | 2018-04 onward, as far as the format allows |
| IIMA Indian Fama-French-Momentum library, survivorship-bias-adjusted monthly file (Agarwalla, Jacob and Varma 2013) | market, size, value, momentum factors and the risk-free rate | to 2025-12 |
| Index funds' own NAVs from AMFI | investable benchmarks | as available |

**Universe:** open-ended, actively managed equity schemes, growth options only. Excluded: index funds, ETFs, fund of funds, arbitrage, hybrid, solution-oriented, overseas schemes, IDCW (dividend) options. Sectoral and thematic funds are kept in the commission analysis and reported separately in the skill analysis (their benchmark is a sector, not the market). Dead and merged schemes stay in the sample for the months they existed.

**Returns:** monthly, from month-end NAVs; no return is computed across a gap longer than one month.

## RQ1. The commission leak

- **Twin plans:** the direct-growth and regular-growth options of the same fund, matched by AMC and normalised scheme name, and accepted only if their monthly returns correlate at 0.98 or more over the overlap. The gap itself is never used to accept or reject a pair.
- **Measure:** annualised direct-minus-regular return per pair; cross-sectional mean and median; by category and by year.
- **H1a:** the gap is positive for at least 95% of pairs. *Null:* the median gap is zero.
- **H1b (natural experiment):** SEBI's expense-ratio cut effective 2019-04-01 narrowed the gap. Test: within-pair change in the annualised gap, 12 months after minus 12 months before, paired t-test and Wilcoxon signed-rank. *Null:* no change. Placebo dates 2016-04-01 and 2022-04-01 must show no comparable change, otherwise the result is reported as a trend, not an effect. Secondary event: 2018-10-22 (upfront commissions abolished).
- **Rupees:** annual cost = gap x regular-plan AUM. The regular-plan share of AUM comes from a cited source; if none is found, a stated range is used and labelled an assumption.

## RQ2. The skill leak

- **Model (primary):** monthly fund excess return regressed on the IIMA four factors, 2006-04 to 2025-12, for funds with at least 36 monthly observations. OLS t-statistics are primary; Newey-West (3 lags) is reported as a check.
- **Luck versus skill:**
  - Fama and French (2010) bootstrap: residuals resampled by month, jointly across funds, under zero true alpha; 2,000 simulations.
  - Barras, Scaillet and Wermers (2010) false discovery rate: estimate pi-zero (truly zero-alpha share), pi-plus and pi-minus; Storey q-values.
- **Primary outcome:** pi-plus, the share of funds with truly positive alpha, for regular plans (what most investors hold) and for direct plans (closest to gross skill).
- **H2:** pi-plus is below 10%. *Null:* pi-plus is zero; reported as such if the estimate's interval includes zero.
- **Benchmark fork (both run, four-factor is primary):** (a) the IIMA four-factor model; (b) an investable benchmark, a constrained style regression (Sharpe 1992: weights at least zero, summing to one) on index-fund returns.
- **Persistence (Carhart 1997):** each December, funds sorted into quintiles on prior 36-month four-factor alpha t-statistic; equal-weighted quintile portfolios held 12 months; four-factor alpha of each. *Null:* the top-minus-bottom spread is zero.

## RQ3. The timing leak

- **Unit (primary):** all active equity schemes together (AMFI subtotal "Equity Schemes"), which is immune to reclassification between equity categories. Secondary: categories whose definition was stable over the window, with reclassification months flagged.
- **Time-weighted return:** monthly r = (AUM_t - AUM_{t-1} - Flow_t) / (AUM_{t-1} + 0.5 x Flow_t), chained. Cross-checked against NAV-based category returns; if they disagree materially, NAV-based returns are used and the switch is logged.
- **Money-weighted return:** the internal rate of return of the cash flows (starting AUM, monthly net flows, ending AUM).
- **Gap:** time-weighted minus money-weighted, annualised.
- **H3:** the gap is positive for the active equity aggregate. *Null:* the actual flow timing is no worse than random timing. Test: the actual money-weighted return is compared with 10,000 permutations of the same flows in random months, with the return path held fixed; one-sided p-value.

## RQ4. Can the leaks be avoided in advance?

- **Design:** each December from 2013 to 2024, a fund-selection rule picks funds using only information available that day; picks are held for the next 12 months. Out-of-sample evaluation 2014-01 to 2025-12 on direct-plan returns (primary) and regular-plan returns (secondary).
- **Features:** prior 36-month alpha and its t-statistic, R-squared against the four-factor model (Amihud and Goyenko 2013), idiosyncratic volatility, prior 12-month return, factor loadings, fund age, category, the twin-plan gap.
- **Strategies:**
  - low R-squared quintile;
  - high prior-alpha quintile;
  - ridge regression;
  - gradient-boosted trees (expanding-window training, hyperparameters fixed in advance: 200 trees, depth 3, learning rate 0.05);
  - all active funds equal-weighted;
  - the Nifty 50 index fund.
- **Outcome:** the four-factor alpha of each strategy's monthly returns. *Null:* no strategy's alpha is significant at the 5% level after Holm correction across strategies.

## Integrity rules

- No look-ahead: every feature is computed from data dated on or before the selection date.
- Baselines first: equal-weighted active funds and the index fund are reported before any model.
- Every number in the report is written by the pipeline (`results/numbers.json` -> `report/numbers.tex`); no number is typed by hand.
- Nulls are results and are published.
- One command (`make all`) regenerates every number from the raw downloads.
