"""Download Yahoo Finance data for a portfolio of Canadian exporters (run on your own laptop).

    pip install yfinance pandas
    python fetch_data.py

Creates a folder `yahoo_data/` with three CSVs. Attach all three back in the chat.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

# Canadian exporters across sectors (TSX tickers on Yahoo end in .TO)
TICKERS = {
    "CAE.TO": "CAE (flight simulators)",
    "BBD-B.TO": "Bombardier (business jets)",
    "MG.TO": "Magna (auto parts)",
    "LNR.TO": "Linamar (auto parts)",
    "DOO.TO": "BRP (recreational vehicles)",
    "ATS.TO": "ATS (industrial automation)",
    "GIL.TO": "Gildan (apparel)",
    "CCL-B.TO": "CCL Industries (packaging)",
    "SAP.TO": "Saputo (dairy)",
    "NTR.TO": "Nutrien (fertilizer)",
    "MX.TO": "Methanex (methanol)",
    "WFG.TO": "West Fraser (lumber)",
    "CFP.TO": "Canfor (lumber)",
    "IFP.TO": "Interfor (lumber)",
    "TECK-B.TO": "Teck (mining)",
    "FM.TO": "First Quantum (copper)",
    "CCO.TO": "Cameco (uranium)",
    "CNQ.TO": "Canadian Natural (oil)",
    "MDA.TO": "MDA Space (satellites)",
    "BLDP.TO": "Ballard Power (fuel cells)",
}
MARKET = "^GSPTSE"   # S&P/TSX Composite index
START = "2015-01-01"

out = Path("yahoo_data")
out.mkdir(exist_ok=True)

# 1. Daily prices and volume (10+ years)
raw = yf.download(list(TICKERS) + [MARKET], start=START, auto_adjust=True, progress=True)
raw["Close"].to_csv(out / "prices.csv")
raw["Volume"].to_csv(out / "volume.csv")

# 2. Balance-sheet debt and shares outstanding (latest annual report)
rows = []
for t, name in TICKERS.items():
    tk = yf.Ticker(t)
    row = {"ticker": t, "name": name}
    try:
        bs = tk.balance_sheet
        latest = bs.columns[0]
        row["report_date"] = latest.date()
        for field in ["Total Debt", "Current Liabilities", "Long Term Debt", "Total Liabilities Net Minority Interest"]:
            row[field] = bs.loc[field, latest] if field in bs.index else None
    except Exception as e:
        row["error"] = str(e)
    try:
        info = tk.info
        row["shares_outstanding"] = info.get("sharesOutstanding")
        row["market_cap"] = info.get("marketCap")
        row["currency"] = info.get("financialCurrency")
        # Some tickers (e.g. CFP.TO) lack sharesOutstanding in .info; fall back to share-count history
        if row["shares_outstanding"] is None:
            shares = tk.get_shares_full(start="2024-01-01")
            if shares is not None and len(shares):
                row["shares_outstanding"] = int(shares.iloc[-1])
        if row["market_cap"] is None and row["shares_outstanding"] and info.get("currentPrice"):
            row["market_cap"] = row["shares_outstanding"] * info["currentPrice"]
    except Exception as e:
        row["error"] = row.get("error", "") + " | " + str(e)
    rows.append(row)
    print(f"{t:10s} done")

pd.DataFrame(rows).to_csv(out / "fundamentals.csv", index=False)
print(f"\nSaved to {out.resolve()}  ->  attach prices.csv, volume.csv, fundamentals.csv")