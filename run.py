"""Canadian Exporter Credit Risk Engine — end-to-end pipeline.

    python fetch_data.py   # once: downloads Yahoo Finance data into yahoo_data/
    python run.py          # ~1 minute: volatility ML -> Merton PDs -> Monte Carlo -> outputs/
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import RISK_FREE, USDCAD, load_fundamentals, load_prices   # noqa: E402
import merton                                                       # noqa: E402
import monte_carlo as mc                                            # noqa: E402
import vol_model                                                    # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ---- chart style ------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def chart_vol_models(table):
    t = table.sort_values("RMSE (vol pts)")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = [MUTED if n.startswith("Naive") else BLUE for n in t.index]
    ax.barh(t.index, t["RMSE (vol pts)"], color=colors, height=0.6)
    for y, v in enumerate(t["RMSE (vol pts)"]):
        ax.text(v + 0.2, y, f"{v:.1f}", va="center", color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Forecast error, RMSE (volatility points, lower is better)",
           title="Next-month volatility forecasts, out of sample (2023–2026)")
    save(fig, "vol_model_accuracy.png")


def chart_distance_to_default(base):
    b = base.sort_values("DD")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(b["name"], b["DD"], color=BLUE, height=0.65)
    for y, (dd, p) in enumerate(zip(b["DD"], b["PD_raw"])):
        lab = f"PD {p:.2%}" if p >= 0.0003 else "PD < 0.03%"
        ax.text(dd + 0.15, y, lab, va="center", color=INK2, fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, b["DD"].max() * 1.25)
    ax.set(xlabel="Distance to default (standard deviations of asset value)",
           title="Merton distance to default, 20 Canadian exporters")
    save(fig, "distance_to_default.png")


def chart_tail(losses):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = [("Baseline (Gaussian copula)", BLUE, "-"), ("Fat tails (t copula)", ORANGE, "-"),
              ("Market crash", BLUE, "--"), ("Market crash + fat tails", ORANGE, "--")]
    for name, c, ls in styles:
        x = np.sort(losses[name])[::-1] / 1e6
        p = np.arange(1, len(x) + 1) / len(x)
        keep = (x > 0) & (p >= 1e-5)          # below 1e-5 there are too few scenarios to be reliable
        ax.step(x[keep], p[keep], where="post", color=c, ls=ls, lw=2, label=name)
    ax.set_yscale("log")
    ax.axhline(0.001, color=MUTED, lw=1, ls=":")
    ax.text(ax.get_xlim()[1] * 0.99, 0.00115, "1-in-1,000 year (99.9%)", ha="right", color=MUTED, fontsize=8.5)
    ax.set(xlabel="Portfolio loss (C\\$ millions)", ylabel="Probability loss exceeds x",
           title="How bad can it get? Loss exceedance curves")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "loss_tail.png")


def chart_scenarios(table):
    cols = [("Expected Loss", AQUA), ("VaR 99.9%", BLUE), ("ES 99.9%", ORANGE)]
    names = list(table.index)
    y = np.arange(len(names))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(names) + 1.4))
    for j, (col, c) in enumerate(cols):
        ax.barh(y + (j - 1) * h, table[col] / 1e6, height=h - 0.03, color=c, label=col)
        for yi, val in zip(y + (j - 1) * h, table[col] / 1e6):
            ax.text(val + 1.5, yi, f"{val:,.1f}", va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, (table["ES 99.9%"].max() / 1e6) * 1.12)
    ax.set(xlabel="C\\$ millions (portfolio of C\\$1 billion)", title="Risk measures by scenario")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "scenarios.png")


def chart_correlation(corr, names):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.outline.set_visible(False)
    ax.set_title("Weekly stock-return correlations, last 3 years")
    save(fig, "correlations.png")


def main():
    t0 = time.time()
    px, volume = load_prices()
    fund = load_fundamentals()
    print(f"Data: {px.shape[1]-1} firms + TSX, {px.index.min().date()} to {px.index.max().date()}")

    # 1) Machine learning: next-month volatility
    v = vol_model.run(px, volume)
    print("\nVolatility models\n", v["table"].round(2), f"\nChampion (by CV): {v['champion']}")

    # 2) Merton: PD per firm, with ML vol vs historical vol, and under a market crash
    base = merton.merton_table(fund, v["vol_ml"], RISK_FREE)
    hist = merton.merton_table(fund, v["vol_hist"], RISK_FREE)
    crash = merton.merton_table(fund, v["vol_ml"], RISK_FREE, equity_shock=-0.30, vol_mult=1.5)
    assert base["max_residual"].max() < 1e-6, "Merton solver did not converge"
    print("\nMerton (ML vol)\n", base[["name", "equity_vol", "asset_vol", "DD", "PD"]].sort_values("DD").round(4))

    # 3) Monte Carlo
    corr = mc.return_correlation(px).loc[fund.index, fund.index]
    ead = np.full(len(fund), mc.EXPOSURE)
    runs = {
        "Baseline (Gaussian copula)": (base["PD"], "gaussian"),
        "Historical vol instead of ML": (hist["PD"], "gaussian"),
        "Fat tails (t copula)": (base["PD"], "t"),
        "Market crash": (crash["PD"], "gaussian"),
        "Market crash + fat tails": (crash["PD"], "t"),
    }
    losses, checks = {}, {}
    for name, (pdv, cop) in runs.items():
        losses[name], sim_pd = mc.simulate(pdv.values, corr.values, ead, cop)
        checks[name] = {"MC EL": losses[name].mean(),
                        "MC EL 95% CI (+/-)": 1.96 * losses[name].std() / np.sqrt(len(losses[name])),
                        "Analytic EL": mc.expected_loss_analytic(pdv, ead),
                        "Max |simulated PD - input PD|": float(np.abs(sim_pd - pdv.values).max())}
    table = pd.DataFrame({k: mc.risk_measures(l) for k, l in losses.items()}).T
    checks = pd.DataFrame(checks).T
    print("\nRisk measures (C$ millions)\n", (table.drop(columns="P(any default)") / 1e6).round(2))
    print("\nValidation\n", checks.round(5))

    # Save tables
    v["table"].round(3).to_csv(OUT / "vol_model_metrics.csv")
    pd.DataFrame({"name": fund["short_name"], "forecast_vol": v["vol_ml"], "hist_1y_vol": v["vol_hist"]}
                 ).round(4).to_csv(OUT / "vol_forecasts.csv")
    for lbl, tab in [("baseline", base), ("hist_vol", hist), ("crash", crash)]:
        tab.round(6).to_csv(OUT / f"merton_{lbl}.csv")
    table.round(2).to_csv(OUT / "risk_scenarios.csv")
    checks.round(6).to_csv(OUT / "validation.csv")
    corr.round(3).to_csv(OUT / "correlation_matrix.csv")
    summary = {
        "as_of": str(v["as_of"]), "risk_free": RISK_FREE, "usdcad": USDCAD,
        "portfolio_C$": float(ead.sum()), "n_sims": mc.N_SIMS, "vol_champion": v["champion"],
        "avg_pairwise_corr": float(corr.values[np.triu_indices(len(corr), 1)].mean()),
        "baseline": {k: float(x) for k, x in table.loc["Baseline (Gaussian copula)"].items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Charts
    chart_vol_models(v["table"])
    chart_distance_to_default(base)
    chart_tail(losses)
    chart_scenarios(table)
    chart_correlation(corr, list(fund["short_name"]))
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
