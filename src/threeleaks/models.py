"""Regression helpers shared by every analysis: single-fund OLS with OLS and Newey-West
t-statistics, and a vectorised many-fund OLS used inside the bootstraps."""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS = ["mkt_rf", "smb", "hml", "wml"]


def design(f: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(f))] + [f[c].to_numpy() for c in FACTORS])


def newey_west_se(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    s = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        g = u[L:].T @ u[:-L]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return np.sqrt(np.diag(cov))


def fit(y: np.ndarray, X: np.ndarray, nw_lags: int = 3) -> dict:
    """OLS of y on X (first column constant). Returns alpha, OLS and NW t-stats, betas, R2, residuals."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    dof = n - k
    s2 = e @ e / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    se_nw = newey_west_se(X, e, nw_lags)
    tss = ((y - y.mean()) ** 2).sum()
    return {"alpha": beta[0], "t": beta[0] / se[0], "t_nw": beta[0] / se_nw[0], "betas": beta[1:],
            "r2": 1 - (e @ e) / tss if tss > 0 else np.nan, "resid_sd": np.sqrt(s2), "n": n, "resid": e,
            "se": se[0], "se_nw": se_nw[0]}


def batch_tstats(Y: np.ndarray, M: np.ndarray, X: np.ndarray, min_obs: int = 8) -> tuple:
    """OLS alpha and its t-stat for many funds at once.

    Y: T x N returns (zeros where missing), M: T x N 0/1 availability mask, X: T x K design.
    Returns (alpha, t) arrays of length N; NaN where a fund has fewer than min_obs observations.
    """
    T, N = Y.shape
    K = X.shape[1]
    xtx = np.einsum("tn,ti,tj->nij", M, X, X)
    xty = np.einsum("tn,ti,tn->ni", M, X, Y)
    nobs = M.sum(0)
    ok = nobs >= max(min_obs, K + 2)
    alpha = np.full(N, np.nan)
    tstat = np.full(N, np.nan)
    if not ok.any():
        return alpha, tstat
    xtx_ok = xtx[ok]
    # Guard against singular X'X when a fund's drawn months repeat too few distinct rows.
    det = np.linalg.det(xtx_ok)
    good = np.abs(det) > 1e-18
    idx = np.where(ok)[0][good]
    inv = np.linalg.inv(xtx_ok[good])
    b = np.einsum("nij,nj->ni", inv, xty[idx])
    fitted = X @ b.T
    resid = (Y[:, idx] - fitted) * M[:, idx]
    s2 = (resid ** 2).sum(0) / (nobs[idx] - K)
    se = np.sqrt(s2 * inv[:, 0, 0])
    alpha[idx] = b[:, 0]
    tstat[idx] = b[:, 0] / se
    return alpha, tstat
