"""External check: commissions AMFI says distributors were actually paid in FY 2024-25.

    python -m threeleaks.external

Parses AMFI's distributor-wise disclosure (gross commission incl. GST, average AUM, Rs lakh)
and reports the total and the implied commission rate on distributor-serviced AUM. This is an
independent anchor for the commission leak measured from twin plans.
"""
from __future__ import annotations

import json
import re
import subprocess

from .config import RAW, TABLES

ROW = re.compile(r"^\s*(\d+)\s+(.+?)\s{2,}(-?[\d,]+\.\d+)\s+(-?[\d,]+\.\d+)\s+(-?[\d,]+\.\d+)\s+(-?[\d,]+\.\d+)\s+(-?[\d,]+\.\d+)\s*$")


def main() -> None:
    pdf = RAW / "amfi_commissions" / "ACDVol-2024-2025.pdf"
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], stdout=subprocess.PIPE, check=True).stdout.decode("utf-8", "replace")
    n, commission, aum = 0, 0.0, 0.0
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            n += 1
            commission += float(m.group(3).replace(",", ""))
            aum += float(m.group(6).replace(",", ""))
    numbers = {"ext_distributors_parsed": n,
               "ext_commission_total_cr": commission / 100,      # Rs lakh -> Rs crore
               "ext_distributor_aum_cr": aum / 100,
               "ext_commission_rate": commission / aum if aum else None}
    (TABLES / "external_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
