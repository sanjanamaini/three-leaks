"""Collect every number the report uses into results/numbers.json and report/numbers.tex.

    python -m threeleaks.numbers

Each analysis writes results/tables/<stage>_numbers.json. This merges them and writes one LaTeX
macro per number (e.g. commission_gap_median -> \\CommissionGapMedian), formatted by type, so
no number in the report is typed by hand.
"""
from __future__ import annotations

import json
import re

from .config import REPORT, RESULTS, TABLES

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def macro_name(key: str) -> str:
    parts = re.split(r"[_\W]+", key)
    out = ""
    for p in parts:
        if not p:
            continue
        p = re.sub(r"\d", lambda m: DIGITS[int(m.group())], p)
        out += p[0].upper() + p[1:]
    return out


def is_pvalue(k: str) -> bool:
    return bool(re.search(r"(^|_)(p|pholm|p_t|p_wilcoxon|wilcoxon_p|p_worse_random|p_worse_random_within_year)$", k)) \
        or bool(re.search(r"(^|_)p(_|$)", k))  # also p_within, p_diversified, ...


def fmt(k: str, v) -> str:
    k = k.replace("low_r2", "lowrsq")  # the strategy's name is a label, not an R-squared
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return "{:,}".format(v)  # counts stay counts (never percents)
    if re.search(r"(^|_)rho(_|$)", k):
        return "%.2f" % v  # correlations
    if "mse" in k:
        return "%.4f" % v
    if is_pvalue(k) or k.endswith("pholm"):
        return "<0.001" if v < 0.001 else ("%.3f" % v)
    if k.endswith("_t") or k.endswith("_t_nw") or k.endswith("_tstat") or re.search(r"_t$|_t_|_vsall_t$", k):
        return "%.2f" % v
    if re.search(r"(pairs|funds|months|_n$|rows|_months$)", k) and float(v).is_integer():
        return "{:,}".format(int(v))
    if k.endswith("_cr"):
        # Rs crore: lakh crore when large, otherwise thousands separators.
        return ("%.2f lakh crore" % (v / 1e5)) if abs(v) >= 1e5 else ("{:,.0f} crore".format(v))
    if k in ("skill_reg_lambda", "skill_dir_lambda", "skill_reg_gamma_plus", "skill_dir_gamma_plus"):
        return "%.2f" % v
    if k.endswith("_coef") or k.endswith("_rho") or k.endswith("_corr") or "r2" in k:
        return "%.2f" % v
    # Default: a share or a return, shown in percent.
    return ("%.2f" % (v * 100)).rstrip("0").rstrip(".") + "\\%" if abs(v) < 0.1 else "%.1f\\%%" % (v * 100)


def main() -> None:
    merged = {}
    for p in sorted(TABLES.glob("*_numbers.json")):
        merged.update(json.loads(p.read_text()))
    dq = TABLES / "data_quality.csv"
    if dq.exists():  # panel counts (schemes, returns, twins, months) written by threeleaks.panels
        for line in dq.read_text().splitlines()[1:]:
            name, value = line.split(",")
            merged["dq_" + name] = int(float(value))
    (RESULTS / "numbers.json").write_text(json.dumps(merged, indent=1, sort_keys=True, default=float))
    lines = ["% Written by threeleaks.numbers from results/numbers.json. Do not edit by hand."]
    for k in sorted(merged):
        lines.append("\\newcommand{\\%s}{%s}" % (macro_name(k), fmt(k, merged[k])))
    REPORT.mkdir(exist_ok=True)
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("numbers: %d macros" % len(merged))


if __name__ == "__main__":
    main()
