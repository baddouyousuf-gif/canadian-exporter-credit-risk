"""Merton (1974) structural credit model: Black-Scholes applied to a firm's balance sheet.

Shareholders own a call option on the firm's assets V with strike equal to its debt D:
if assets end up below the debt, equity is worthless and the firm defaults.

We observe equity value E and equity volatility sigma_E, but not V or sigma_V. Two equations
pin them down (solved numerically for each firm):

    E           = V * N(d1) - D * exp(-rT) * N(d2)          (Black-Scholes call price)
    sigma_E * E = N(d1) * sigma_V * V                       (Ito's lemma: how equity vol relates to asset vol)

    d1 = [ln(V/D) + (r + sigma_V^2 / 2) T] / (sigma_V sqrt(T)),   d2 = d1 - sigma_V sqrt(T)

Distance to default DD = d2: how many standard deviations assets sit above the default point.
Probability of default PD = N(-DD). (This is the risk-neutral PD; it uses the risk-free rate as
the asset drift, so it is a slightly conservative estimate.)
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

T = 1.0              # one-year horizon
PD_FLOOR = 0.0003    # 0.03%: the Basel II floor for corporate PDs; no firm is truly default-free


def solve_firm(E, sigma_E, D, r, T=T):
    def eqs(x):
        V, sV = x
        d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
        d2 = d1 - sV * np.sqrt(T)
        return [(V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E) / E,
                (norm.cdf(d1) * sV * V - sigma_E * E) / (sigma_E * E)]

    x0 = [E + D, sigma_E * E / (E + D)]        # start: assets = equity + debt
    sol = least_squares(eqs, x0, bounds=([E, 1e-4], [10 * (E + D), 5.0]), xtol=1e-12, ftol=1e-12)
    V, sV = sol.x
    dd = (np.log(V / D) + (r - 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    return {"asset_value": V, "asset_vol": sV, "DD": dd, "PD_raw": norm.cdf(-dd),
            "max_residual": float(np.max(np.abs(sol.fun)))}


def merton_table(fund, sigma_E, r, equity_shock=0.0, vol_mult=1.0):
    rows = {}
    for t in fund.index:
        E = fund.loc[t, "equity"] * (1 + equity_shock)
        res = solve_firm(E, sigma_E[t] * vol_mult, fund.loc[t, "default_point"], r)
        rows[t] = {"name": fund.loc[t, "short_name"], "equity": E, "equity_vol": sigma_E[t] * vol_mult,
                   "default_point": fund.loc[t, "default_point"], **res}
    out = pd.DataFrame(rows).T
    num = [c for c in out.columns if c != "name"]
    out[num] = out[num].astype(float)
    out["PD"] = out["PD_raw"].clip(lower=PD_FLOOR)
    return out


if __name__ == "__main__":
    from data import RISK_FREE, load_fundamentals, load_prices
    import vol_model
    fund = load_fundamentals()
    v = vol_model.run(*load_prices())
    tab = merton_table(fund, v["vol_ml"], RISK_FREE)
    pd.set_option("display.width", 200)
    print(tab[["name", "equity_vol", "asset_vol", "DD", "PD_raw", "PD", "max_residual"]]
          .sort_values("DD").round(5).to_string())
    h = merton_table(fund, v["vol_hist"], RISK_FREE)
    print("mean PD ML", tab.PD.mean(), " hist", h.PD.mean())
