"""Build docs/index.html: a self-contained scorecard of every live active equity fund.

    PYTHONPATH=src python tool/build_scorecard.py

Per fund: the commission its regular plan costs (direct-minus-regular return over the last
36 months), the direct plan's four-factor alpha with its false-discovery-rate q-value and a
verdict, R-squared against the four-factor model, and the cheaper index alternative.
Verdict rule (fixed here, before reading any fund's result): q-value at or below 0.10 with a
positive t-statistic -> "Evidence of skill"; with a negative t-statistic -> "Evidence of
underperformance"; otherwise "Indistinguishable from luck". Sector funds get no verdict
because the market factors are not their benchmark. No external files; works offline.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from threeleaks.config import FACTOR_END, INTERIM, NAV_END, TABLES  # noqa: E402

Q_CUTOFF = 0.10
ALTERNATIVE = {"Large Cap": "Nifty 50 index fund", "Mid Cap": "Nifty Midcap 150 index fund",
               "Small Cap": "Nifty Smallcap 250 index fund", "Large & Mid Cap": "Nifty 50 + Midcap 150 index funds",
               "Sectoral/Thematic": "the matching sector index fund, if one exists"}


def verdict(row) -> str:
    if row["category"] == "Sectoral/Thematic":
        return "No verdict (sector fund)"
    if pd.isna(row.get("q_value")):
        return "Too little history"
    if row["q_value"] <= Q_CUTOFF:
        return "Evidence of skill" if row["t_nw"] > 0 else "Evidence of underperformance"
    return "Indistinguishable from luck"


ACRONYMS = {"SBI", "HDFC", "ICICI", "UTI", "DSP", "IDFC", "LIC", "ITI", "PGIM", "HSBC", "JM", "PPFAS", "IDBI", "BOI", "NJ",
            "ESG", "MNC", "PSU", "ELSS", "BSE", "IT", "FMCG", "US", "II", "III", "IV", "SIP", "AMC"}


def display_name(name: str) -> str:
    """AMFI mixes 'SBI CONTRA FUND' and 'Kotak Contra Fund'; title-case the all-capitals ones, keeping acronyms."""
    if not name.isupper():
        return name
    return " ".join(w if w.strip("()-,.&") in ACRONYMS else w.capitalize() for w in name.split())


def build() -> pd.DataFrame:
    schemes = pd.read_parquet(INTERIM / "schemes.parquet")
    twins = pd.read_parquet(INTERIM / "twins.parquet")
    panel = pd.read_parquet(INTERIM / "twin_panel.parquet")
    sk = pd.read_csv(TABLES / "skill_funds_direct.csv", dtype={"code": str})
    last = pd.Period(NAV_END, "M")
    recent = panel[panel["month"] > last - 36]
    gap = recent.groupby("fund_id")["gap_log"].agg(["count", "mean"])
    gap = gap[gap["count"] >= 12]
    gap["commission"] = np.exp(12 * gap["mean"]) - 1
    alive = schemes.set_index("code")["last_month"]
    t = twins[twins["direct_code"].map(alive) == last].copy()
    t = t.merge(gap[["commission"]], left_on="fund_id", right_index=True, how="left")
    t = t.merge(sk[["code", "alpha_ann", "t_nw", "q_value", "alpha_eb", "prob_pos", "r2", "n"]], left_on="direct_code", right_on="code", how="left")
    names = schemes.set_index("code")
    t["fund"] = t["fund_id"].str.split(" | ", regex=False).str[-1].map(display_name)  # literal split: a multi-character pattern is a regex by default
    t["amc"] = t["direct_code"].map(names["master_amc"]).fillna(t["direct_code"].map(names["amc"]))
    t["verdict"] = t.apply(verdict, axis=1)
    t["alternative"] = t["category"].map(ALTERNATIVE).fillna("Nifty 500 or Nifty 50 index fund")
    cols = ["fund", "amc", "category", "commission", "alpha_ann", "alpha_eb", "prob_pos", "t_nw", "q_value", "r2", "n", "verdict", "alternative"]
    return t[cols].sort_values(["category", "fund"])


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Three Leaks Scorecard</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1a;--muted:#5f5f5a;--line:#e2e1dc;--accent:#1F3A5F;
--skill:#2e7d4f;--under:#b3261e;--luck:#6b6b66;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#a3a29b;--line:#34342f;
--accent:#8fb4e3;--skill:#6cc08f;--under:#f08b84;--luck:#a3a29b;--chip:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:24px;margin:0 0 6px}p.lede{color:var(--muted);margin:0 0 18px;max-width:780px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.card span{color:var(--muted);font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px}
.controls input,.controls select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.controls input{flex:1;min-width:200px}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:900px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;cursor:pointer;white-space:nowrap;user-select:none}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.v{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;background:var(--chip);white-space:nowrap}
.v.skill{color:var(--skill)}.v.under{color:var(--under)}.v.luck{color:var(--luck)}
.sub{color:var(--muted);font-size:12px}
details{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
summary{cursor:pointer;font-weight:600}
footer{color:var(--muted);font-size:12px;margin-top:22px}
</style></head><body><div class="wrap">
<h1>Three Leaks Scorecard</h1>
<p class="lede">Every live active equity mutual fund in India, checked for two of the three ways investors lose returns: the commission inside the regular plan, and whether the fund's record beats luck. NAVs to __NAV_END__; factor alphas to __FACTOR_END__.</p>
<div class="cards">__CARDS__</div>
<div class="controls">
<input id="q" type="search" placeholder="Search fund or AMC" aria-label="Search fund or AMC">
<select id="cat" aria-label="Category"><option value="">All categories</option>__CATS__</select>
<select id="ver" aria-label="Verdict"><option value="">All verdicts</option>__VERS__</select>
</div>
<div class="tablewrap"><table><thead><tr>
<th data-k="fund">Fund</th><th data-k="category">Category</th>
<th data-k="commission" title="Direct plan return minus regular plan return, last 36 months, % a year">Regular-plan cost</th>
<th data-k="alpha_ann" title="Direct plan four-factor alpha, % a year">Alpha</th>
<th data-k="alpha_eb" title="Alpha after shrinking toward the average fund to discount luck (empirical Bayes), % a year">Expected alpha</th>
<th data-k="prob_pos" title="Estimated probability that the fund's true alpha is above zero">Chance alpha &gt; 0</th>
<th data-k="t_nw">t-stat</th><th data-k="q_value" title="False discovery rate if this fund were called skilled or unskilled">q-value</th>
<th data-k="verdict">Verdict</th><th data-k="r2" title="Share of monthly return variation explained by market, size, value and momentum">R-squared</th>
<th data-k="alternative">Cheaper alternative</th></tr></thead><tbody id="rows"></tbody></table></div>
<details><summary>How to read this</summary>
<p><b>Regular-plan cost</b> is what the regular plan returned less than its direct twin over the last three years. Both plans hold the same portfolio with the same manager, so the gap is the price of distribution.</p>
<p><b>Alpha</b> is the direct plan's average return beyond what the market, size, value and momentum factors (IIMA data library) explain, after the direct plan's own expenses. Many funds show positive alpha by chance. The <b>q-value</b> is the estimated share of false discoveries if this fund, and every fund with stronger evidence on the same side, were called skilled (or unskilled), following Barras, Scaillet and Wermers (2010). The verdict uses q at or below 0.10.</p>
<p><b>Expected alpha</b> shrinks each fund's alpha toward the average fund, more so when the estimate is noisy (empirical Bayes). <b>Chance alpha &gt; 0</b> is the estimated probability that the fund's true alpha is positive. Both discount luck; the raw alpha does not.</p>
<p><b>R-squared</b> near 1 means the fund's returns are almost fully explained by the factor portfolio, which an index fund can deliver for less.</p>
<p>Past alpha is a weak guide to future alpha; the project's out-of-sample tests are in the report. This is research, not investment advice.</p>
</details>
<footer>Built by the open-source <i>Three Leaks</i> project from AMFI NAV history and the IIMA Indian Fama-French-Momentum library (Agarwalla, Jacob and Varma 2013). Code and method: see the repository README.</footer>
</div>
<script>
const DATA = __DATA__;
const pct = v => v===null||v===undefined||Number.isNaN(v) ? "" : (v*100).toFixed(2)+"%";
const num = (v,d) => v===null||v===undefined||Number.isNaN(v) ? "" : v.toFixed(d);
const cls = v => v.startsWith("Evidence of skill") ? "skill" : v.startsWith("Evidence of under") ? "under" : "luck";
let sortK = "category", asc = true;
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function render(){
  const q=document.getElementById("q").value.toLowerCase(), c=document.getElementById("cat").value, v=document.getElementById("ver").value;
  let rows=DATA.filter(r=>(!q||(r.fund+" "+r.amc).toLowerCase().includes(q))&&(!c||r.category===c)&&(!v||r.verdict===v));
  rows.sort((a,b)=>{const x=a[sortK],y=b[sortK];if(x===y)return 0;if(x===null||x===undefined)return 1;if(y===null||y===undefined)return -1;return (x<y?-1:1)*(asc?1:-1)});
  document.getElementById("rows").innerHTML=rows.map(r=>`<tr><td><b>${esc(r.fund)}</b><br><span class="sub">${esc(r.amc)}</span></td><td>${esc(r.category)}</td>
<td class="num">${pct(r.commission)}</td><td class="num">${pct(r.alpha_ann)}</td><td class="num">${pct(r.alpha_eb)}</td><td class="num">${r.prob_pos==null?"":Math.round(r.prob_pos*100)+"%"}</td><td class="num">${num(r.t_nw,2)}</td><td class="num">${num(r.q_value,2)}</td>
<td><span class="v ${cls(r.verdict)}">${esc(r.verdict)}</span></td><td class="num">${num(r.r2,2)}</td><td>${esc(r.alternative)}</td></tr>`).join("");
}
document.querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{const k=th.dataset.k;asc=(sortK===k)?!asc:true;sortK=k;render()}));
["q","cat","ver"].forEach(id=>document.getElementById(id).addEventListener("input",render));
render();
</script></body></html>
"""


def main() -> None:
    df = build()
    recs = json.loads(df.replace({np.nan: None}).to_json(orient="records"))
    n = len(df)
    sv = df[df["verdict"] != "No verdict (sector fund)"]
    cards = [(str(n), "live active equity funds with both plans"),
             (("%.2f%%" % (df["commission"].median() * 100)) if n else "", "median yearly cost of the regular plan"),
             (str(int((sv["verdict"] == "Evidence of skill").sum())), "funds with evidence of skill (q at or below 0.10)"),
             (str(int((sv["verdict"] == "Indistinguishable from luck").sum())), "funds indistinguishable from luck")]
    page = (PAGE.replace("__DATA__", json.dumps(recs))
            .replace("__NAV_END__", NAV_END).replace("__FACTOR_END__", FACTOR_END)
            .replace("__CARDS__", "".join('<div class="card"><b>%s</b><span>%s</span></div>' % (html.escape(a), html.escape(b)) for a, b in cards))
            .replace("__CATS__", "".join('<option>%s</option>' % html.escape(c) for c in sorted(df["category"].dropna().unique())))
            .replace("__VERS__", "".join('<option>%s</option>' % html.escape(v) for v in sorted(df["verdict"].unique()))))
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    df.to_csv(TABLES / "scorecard.csv", index=False)
    print("scorecard: %d funds -> %s" % (n, out))


if __name__ == "__main__":
    main()
