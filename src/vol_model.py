"""Forecast each firm's next-month equity volatility with machine learning, against econometric baselines.

Target: realised volatility over the next 21 trading days (annualised).
Panel: 20 firms, one observation every 21 trading days (non-overlapping targets).
Selection: the champion is picked by time-series cross-validation inside the training period
(2016-2022). The 2023+ test set is only used once, to report out-of-sample accuracy.

Models (all fit in log-volatility, which is close to normal; converted back with Duan's smearing
correction so forecasts are unbiased in volatility terms):
  - Naive: next month's vol = last month's vol (random-walk benchmark)
  - HAR: Corsi (2009) heterogeneous autoregressive model, the standard econometric benchmark;
         regresses future vol on weekly, monthly and quarterly realised vol
  - HAR-X: HAR plus market, return, drawdown, volume and beta features (regularised-free OLS)
  - Gradient boosting: boosted trees on all features, learning what HAR misses (fit to HAR residuals)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from data import MARKET, log_returns

H = 21                    # forecast horizon (trading days)
STEP = 21                 # sample every month so targets never overlap
TEST_START = "2023-01-01"
ANN = np.sqrt(252)
HAR_FEATURES = ["lrv_5", "lrv_21", "lrv_63"]
FEATURES = HAR_FEATURES + ["lrv_252", "ret_21", "ret_63", "drawdown_252",
                           "mkt_lrv_21", "mkt_ret_21", "volume_ratio", "beta_252"]
FEATURE_LABELS = {
    "lrv_5": "Vol, last week", "lrv_21": "Vol, last month", "lrv_63": "Vol, last quarter",
    "lrv_252": "Vol, last year", "ret_21": "Return, last month", "ret_63": "Return, last quarter",
    "drawdown_252": "Drawdown from 1-yr high", "mkt_lrv_21": "TSX vol, last month",
    "mkt_ret_21": "TSX return, last month", "volume_ratio": "Trading volume vs normal",
    "beta_252": "Market beta",
}


def realised_vol(r, w):
    """Annualised realised volatility over the trailing w days (zero-mean, standard for daily data)."""
    return np.sqrt((r ** 2).rolling(w, min_periods=int(w * 0.8)).mean()) * ANN


def build_features(px, volume):
    r = log_returns(px)
    mkt = r[MARKET]
    mkt_rv21, mkt_ret21 = realised_vol(mkt, 21), mkt.rolling(21).sum()
    rows = []
    for f in [c for c in px.columns if c != MARKET]:
        ri = r[f]
        d = pd.DataFrame(index=px.index)
        for w in (5, 21, 63, 252):
            d[f"lrv_{w}"] = np.log(realised_vol(ri, w))
        d["ret_21"] = ri.rolling(21).sum()
        d["ret_63"] = ri.rolling(63).sum()
        d["drawdown_252"] = np.log(px[f] / px[f].rolling(252, min_periods=200).max())
        d["mkt_lrv_21"] = np.log(mkt_rv21)
        d["mkt_ret_21"] = mkt_ret21
        d["volume_ratio"] = np.log(volume[f].rolling(21).mean() / volume[f].rolling(252, min_periods=200).mean())
        d["beta_252"] = ri.rolling(252).cov(mkt) / mkt.rolling(252).var()
        # Target: realised vol over the NEXT 21 days, aligned to today's features
        d["target_lrv"] = np.log(np.sqrt((ri ** 2).rolling(H).mean().shift(-H)) * ANN)
        d["ticker"] = f
        rows.append(d)
    panel = pd.concat(rows).replace([np.inf, -np.inf], np.nan)
    panel.index.name = "date"
    return panel


def sample_monthly(panel):
    dates = panel.index.unique().sort_values()
    return panel.loc[panel.index.isin(dates[252::STEP])]   # start after a full year of history


# ---- models: each exposes fit(df) and predict(df) returning annualised vol --------------------------
class Naive:
    def fit(self, df):
        return self

    def predict(self, df):
        return np.exp(df["lrv_21"].values)


class LinearVol:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, df):
        self.m = LinearRegression().fit(df[self.feats], df["target_lrv"])
        resid = df["target_lrv"] - self.m.predict(df[self.feats])
        self.smear = np.mean(np.exp(resid))            # Duan (1983) smearing correction
        return self

    def predict(self, df):
        return np.exp(self.m.predict(df[self.feats])) * self.smear


class BoostedHAR:
    """HAR captures persistence; gradient boosting learns the non-linear remainder from all features."""

    def fit(self, df):
        self.har = LinearRegression().fit(df[HAR_FEATURES], df["target_lrv"])
        resid = df["target_lrv"] - self.har.predict(df[HAR_FEATURES])
        self.gb = HistGradientBoostingRegressor(
            learning_rate=0.02, max_iter=300, max_depth=2, min_samples_leaf=80,
            l2_regularization=1.0, random_state=42).fit(df[FEATURES], resid)
        self.smear = np.mean(np.exp(resid - self.gb.predict(df[FEATURES])))
        return self

    def predict(self, df):
        return np.exp(self.har.predict(df[HAR_FEATURES]) + self.gb.predict(df[FEATURES])) * self.smear


def make_models():
    return {
        "Naive (last month)": Naive(),
        "HAR regression": LinearVol(HAR_FEATURES),
        "HAR-X regression": LinearVol(FEATURES),
        "Gradient boosting": BoostedHAR(),
    }


def score(y, p):
    return {"RMSE (vol pts)": 100 * np.sqrt(np.mean((y - p) ** 2)),
            "MAE (vol pts)": 100 * np.mean(np.abs(y - p)),
            "Bias (vol pts)": 100 * np.mean(p - y)}


def time_series_cv(train, n_folds=4):
    """Expanding-window CV inside the training period: fit on the past, score on the next block."""
    dates = train.index.unique().sort_values()
    rmse = {name: [] for name in make_models()}
    for k in range(n_folds):
        cut = dates[int(len(dates) * (0.5 + 0.125 * k))]
        end = dates[min(int(len(dates) * (0.625 + 0.125 * k)), len(dates) - 1)]
        a, b = train[train.index < cut], train[(train.index >= cut) & (train.index < end)]
        for name, m in make_models().items():
            rmse[name].append(score(np.exp(b["target_lrv"].values), m.fit(a).predict(b))["RMSE (vol pts)"])
    return {k: float(np.mean(v)) for k, v in rmse.items()}


def run(px, volume):
    panel = build_features(px, volume)
    data = sample_monthly(panel).dropna(subset=FEATURES + ["target_lrv"]).sort_index()
    train, test = data[data.index < TEST_START], data[data.index >= TEST_START]

    cv = time_series_cv(train)
    rows, test_preds = {}, {}
    for name, m in make_models().items():
        p = m.fit(train).predict(test)
        test_preds[name] = p
        rows[name] = {"CV RMSE (vol pts)": cv[name], **score(np.exp(test["target_lrv"].values), p)}
    table = pd.DataFrame(rows).T
    champion = table["CV RMSE (vol pts)"].idxmin()          # chosen on CV, never on the test set

    # Refit the champion on all labelled data, forecast from the latest date
    latest = panel.loc[panel.index.max()].set_index("ticker")
    vol_ml = pd.Series(make_models()[champion].fit(data).predict(latest), index=latest.index)
    vol_hist = np.exp(latest["lrv_252"])                    # industry default: trailing 1-year vol

    har_x = make_models()["HAR-X regression"].fit(train)
    coefs = pd.Series(har_x.m.coef_, index=FEATURES)
    std = train[FEATURES].std()
    importance = (coefs * std).rename(index=FEATURE_LABELS)  # effect of a 1-sd move in each feature

    return {"table": table, "champion": champion, "vol_ml": vol_ml, "vol_hist": vol_hist,
            "test": test, "test_preds": test_preds, "importance": importance,
            "n_train": len(train), "n_test": len(test),
            "train_span": (train.index.min().date(), train.index.max().date()),
            "test_span": (test.index.min().date(), test.index.max().date()),
            "as_of": panel.index.max().date()}


if __name__ == "__main__":
    from data import load_prices
    out = run(*load_prices())
    print(out["table"].round(2))
    print("champion:", out["champion"], "| train", out["n_train"], out["train_span"],
          "| test", out["n_test"], out["test_span"])
    print(out["importance"].sort_values().round(3))
    print(pd.DataFrame({"forecast": out["vol_ml"], "hist 1y": out["vol_hist"]}).round(3))
