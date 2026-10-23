"""Monte Carlo credit portfolio losses with correlated defaults (CreditMetrics-style copula).

Each firm i has a latent "asset return" X_i. Firms default when X_i falls below the threshold
that matches their Merton PD. The X_i are correlated using the firms' actual stock-return
correlations, so defaults cluster the way the firms' businesses co-move (lumber with lumber,
miners with miners, everyone with the TSX).

  Gaussian copula:  X = L z,                  default if X_i < Phi^-1(PD_i)
  Student-t copula: X = L z / sqrt(W / nu),    default if X_i < t_nu^-1(PD_i)

L is the Cholesky factor of the correlation matrix. The t copula shares one random "panic"
scale W across all firms, so extreme moves happen together (tail dependence) — the Gaussian
copula famously underestimates this.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from data import MARKET, log_returns

N_SIMS = 2_000_000
EXPOSURE = 50e6              # hypothetical insured exposure per firm, C$
LGD_MEAN, LGD_SD = 0.45, 0.20
T_DF = 4                     # degrees of freedom for the fat-tailed scenario
SEED = 2026


def return_correlation(px, years=3):
    """Correlation of weekly log returns over the last `years` (weekly smooths out timing noise)."""
    weekly = log_returns(px.drop(columns=[MARKET]).resample("W-FRI").last()).dropna(how="all")
    weekly = weekly[weekly.index >= weekly.index.max() - pd.DateOffset(years=years)]
    corr = weekly.corr()
    # Guarantee positive-definiteness for the Cholesky step
    vals, vecs = np.linalg.eigh(corr.values)
    vals = np.clip(vals, 1e-6, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return pd.DataFrame(fixed / np.outer(d, d), index=corr.index, columns=corr.columns)


def beta_params(mean, sd):
    k = mean * (1 - mean) / sd ** 2 - 1
    return mean * k, (1 - mean) * k


def simulate(pd_, corr, ead, copula="gaussian", n_sims=N_SIMS, seed=SEED, chunk=250_000):
    """Return simulated portfolio losses and per-firm default counts."""
    pd_ = np.asarray(pd_, float)
    ead = np.asarray(ead, float)
    L = np.linalg.cholesky(np.asarray(corr, float))
    thresh = norm.ppf(pd_) if copula == "gaussian" else student_t.ppf(pd_, T_DF)
    a, b = beta_params(LGD_MEAN, LGD_SD)
    rng = np.random.default_rng(seed)
    losses = np.empty(n_sims)
    default_counts = np.zeros(len(pd_))
    for s in range(0, n_sims, chunk):
        n = min(chunk, n_sims - s)
        x = rng.standard_normal((n, len(pd_))) @ L.T
        if copula == "t":
            x = x / np.sqrt(rng.chisquare(T_DF, size=(n, 1)) / T_DF)
        d = x < thresh
        default_counts += d.sum(axis=0)
        lgd = rng.beta(a, b, size=d.shape)
        losses[s:s + n] = (d * lgd * ead).sum(axis=1)
    return losses, default_counts / n_sims


def risk_measures(losses):
    out = {"Expected Loss": losses.mean(), "P(any default)": (losses > 0).mean()}
    srt = np.sort(losses)
    for c in (0.99, 0.995, 0.999):
        out[f"VaR {c:.1%}"] = np.quantile(losses, c)
        # Expected Shortfall = average of the worst (1-c) share of scenarios
        out[f"ES {c:.1%}"] = srt[int(np.floor(c * len(srt))):].mean()
    out["Economic Capital (VaR 99.9% - EL)"] = out["VaR 99.9%"] - out["Expected Loss"]
    return out


def expected_loss_analytic(pd_, ead):
    return float(np.sum(np.asarray(pd_) * LGD_MEAN * np.asarray(ead)))
