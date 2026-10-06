"""Unit tests for the logic most likely to break silently.

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from threeleaks.models import batch_tstats, fit
from threeleaks.numbers import fmt, macro_name
from threeleaks.panels import classify_category, is_growth_option
from threeleaks.skill import empirical_bayes, pi0_hat, storey_q
from threeleaks.timing import implied_returns, mwr, simulate_end


class Classification(unittest.TestCase):
    """The eight cases that exposed classification bugs during development."""

    def test_growth_options(self):
        self.assertTrue(is_growth_option("ABN AMRO Dividend Yield Fund-Growth Option"))
        self.assertTrue(is_growth_option("Franklin India Dividend Yield Fund-Growth Plan"))
        self.assertTrue(is_growth_option("UTI-Dividend Yield Fund.-Growth"))
        self.assertTrue(is_growth_option("JM Emerging Leaders Fund-Growth"))
        self.assertTrue(is_growth_option("DBS Chola Global Advantage Fund.-Cumulative Option"))
        self.assertFalse(is_growth_option("Sundaram Growth Fund-Dividend"))
        self.assertFalse(is_growth_option("Nippon India Small Cap Fund - Growth Plan - Bonus Option"))
        self.assertFalse(is_growth_option("quant Mid Cap Fund - IDCW Option - Regular Plan"))

    def test_categories(self):
        self.assertEqual(classify_category("Growth", "IIFL Dividend Opportunities Index Fund - Growth"), "Passive")
        self.assertEqual(classify_category("Hybrid Scheme - Equity Savings", "Principal Equity Savings Fund - Growth Option"), "Other")
        self.assertEqual(classify_category("Growth", "DBS Chola Global Advantage Fund.-Cumulative Option"), "Overseas")
        self.assertEqual(classify_category("Growth", "JM Emerging Leaders Fund-Growth"), "Pre-2018 equity")
        self.assertEqual(classify_category("Equity Scheme - Large & Mid Cap Fund", "X Fund"), "Large & Mid Cap")
        self.assertEqual(classify_category("Equity Schemes - Large Cap Fund", "X Fund"), "Large Cap")
        self.assertEqual(classify_category("Equity Scheme - ELSS", "X Fund"), "ELSS")
        self.assertEqual(classify_category("Equity Schemes - Thematic Fund", "X Fund"), "Sectoral/Thematic")


class TimingMaths(unittest.TestCase):
    def test_aum_identity(self):
        """Implied returns plus the flows must rebuild the AUM path exactly."""
        aum = pd.Series([100.0, 112.0, 108.0, 130.0], index=pd.period_range("2020-01", periods=4, freq="M"))
        flow = pd.Series([0.0, 5.0, -3.0, 10.0], index=aum.index)
        s = pd.DataFrame({"aum": aum, "net_inflow": flow})
        r = implied_returns(s).iloc[1:].to_numpy()
        end = simulate_end(100.0, r, flow.iloc[1:].to_numpy())
        self.assertAlmostEqual(end, 130.0, places=9)

    def test_mwr_equals_twr_without_flows(self):
        r = np.array([0.01, -0.02, 0.03, 0.005])
        end = simulate_end(100.0, r, np.zeros(4))
        twr = np.prod(1 + r) ** (1 / 4) - 1
        self.assertAlmostEqual(mwr(100.0, np.zeros(4), end), twr, places=10)

    def test_bad_timing_lowers_mwr(self):
        """Money added just before a fall must earn less than the fund."""
        r = np.array([0.05, 0.05, -0.10, 0.02])
        flows = np.array([0.0, 50.0, 0.0, 0.0])
        end = simulate_end(100.0, r, flows)
        twr = np.prod(1 + r) ** (1 / 4) - 1
        self.assertLess(mwr(100.0, flows, end), twr)


class Statistics(unittest.TestCase):
    def test_batch_matches_single(self):
        rng = np.random.default_rng(1)
        T, N = 60, 5
        X = np.column_stack([np.ones(T), rng.normal(size=(T, 2))])
        Y = X @ rng.normal(size=(3, N)) + rng.normal(scale=0.1, size=(T, N))
        M = np.ones((T, N))
        M[:10, 0] = 0
        a, t = batch_tstats(np.where(M > 0, Y, 0.0), M, X)
        single = fit(Y[10:, 0], X[10:], 3)
        self.assertAlmostEqual(a[0], single["alpha"], places=10)
        self.assertAlmostEqual(t[0], single["t"], places=8)

    def test_pi0_uniform(self):
        p = np.random.default_rng(2).uniform(size=20000)
        self.assertAlmostEqual(pi0_hat(p, 0.5), 1.0, delta=0.03)

    def test_q_values_bounded_and_monotone(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(size=200)
        t = rng.normal(size=200)
        q = storey_q(p, t, 0.8)
        self.assertTrue(np.all((q >= 0) & (q <= 1)))
        pos = np.where(t > 0)[0]
        order = pos[np.argsort(p[pos])]
        self.assertTrue(np.all(np.diff(q[order]) >= -1e-12))

    def test_shrinkage_pulls_noisy_estimates_harder(self):
        a = np.array([0.05, 0.05, -0.02, 0.0, 0.01])
        se = np.array([0.01, 0.05, 0.02, 0.02, 0.03])
        post, *_ = empirical_bayes(a, se)
        mu = a.mean()
        self.assertLess(abs(post[1] - mu), abs(post[0] - mu))


class Numbers(unittest.TestCase):
    def test_macro_names_are_letters(self):
        self.assertEqual(macro_name("skill_ff_reg_p99_t"), "SkillFfRegPNineNineT")
        self.assertTrue(macro_name("commission_placebo_2016_p").isalpha())

    def test_formats(self):
        self.assertEqual(fmt("commission_gap_median", 0.0115), "1.15\\%")
        self.assertEqual(fmt("commission_pairs", 512), "512")
        self.assertEqual(fmt("commission_ter_p", 0.0004), "<0.001")
        self.assertEqual(fmt("syn_aum_2025_cr", 3500000.0), "35.00 lakh crore")


if __name__ == "__main__":
    unittest.main()
