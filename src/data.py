"""Load and clean the Yahoo Finance data produced by fetch_data.py.

Prices are adjusted closes in CAD (TSX listings). Balance sheets for some firms are reported in
USD, so debt is converted to CAD. Market capitalisation comes straight from Yahoo (in CAD, and it
counts every share class, which matters for dual-class firms such as Bombardier and BRP).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "yahoo_data"
MARKET = "^GSPTSE"

# Market inputs as of 23 Oct 2026 (sources in README)
USDCAD = 1.4005          # USD/CAD close, 23 Oct 2026
RISK_FREE = 0.0301       # Government of Canada 1-year T-bill, 21 Oct 2026 (Bank of Canada Valet)


def load_prices():
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index()
    vol = pd.read_csv(DATA / "volume.csv", index_col=0, parse_dates=True).sort_index()
    return px, vol


def log_returns(px):
    return np.log(px / px.shift(1))


def load_fundamentals():
    f = pd.read_csv(DATA / "fundamentals.csv")
    fx = np.where(f["currency"].eq("USD"), USDCAD, 1.0)
    for col in ["Total Debt", "Current Liabilities", "Long Term Debt"]:
        f[col + " (CAD)"] = f[col] * fx
    # Some firms report no separate long-term debt line; fall back to total debt
    ltd = f["Long Term Debt (CAD)"].fillna(f["Total Debt (CAD)"])
    # KMV / Moody's default point: short-term liabilities + half of long-term debt
    f["default_point"] = f["Current Liabilities (CAD)"] + 0.5 * ltd
    f["equity"] = f["market_cap"]
    f["short_name"] = f["name"].str.split(" \\(").str[0]
    return f.set_index("ticker")


if __name__ == "__main__":
    px, vol = load_prices()
    print(px.shape, px.index.min().date(), "->", px.index.max().date())
    f = load_fundamentals()
    print((f[["short_name", "equity", "default_point"]].assign(
        leverage=lambda d: d.default_point / (d.default_point + d.equity))).round(3).to_string())
