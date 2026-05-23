"""
visualization.py
================
Module 3 — Visualization
Tạo 8 charts và lưu vào output/charts/.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gaussian_kde, norm

from config import (
    TICKERS, CHART_DIR, PROCESSED_CSV,
    COLORS, STYLE_BG, STYLE_PANEL, STYLE_GRID,
    STYLE_TEXT, STYLE_SUB, COLOR_UP, COLOR_DOWN,
)

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────

def _load_df_for_viz(path: str) -> pd.DataFrame:
    """Đọc processed CSV multi-index, parse Date thành index."""
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = pd.to_datetime(df[("Date", "Unnamed: 1_level_1")])
    df = df.drop(columns=[("Date", "Unnamed: 1_level_1")])
    df.index.name = "Date"
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


def _dark_style(fig, axes_list):
    fig.patch.set_facecolor(STYLE_BG)
    for ax in axes_list:
        ax.set_facecolor(STYLE_PANEL)
        ax.tick_params(colors=STYLE_TEXT, labelsize=9)
        ax.xaxis.label.set_color(STYLE_TEXT)
        ax.yaxis.label.set_color(STYLE_TEXT)
        ax.title.set_color(STYLE_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE_GRID)
        ax.grid(color=STYLE_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


# ── Chart functions ───────────────────────────────────────────

def plot_price_volume(df: pd.DataFrame) -> plt.Figure:
    """Chart 1 — Price Trend + Volume Overlay (4 subplots)."""
    fig = plt.figure(figsize=(16, 10), facecolor=STYLE_BG)
    fig.suptitle(
        "Stock Price Trends with Volume Overlay\n(AAPL · MSFT · NVDA · TSLA — 2025–2026)",
        fontsize=15, fontweight="bold", color=STYLE_TEXT, y=0.98,
    )
    n  = len(TICKERS)
    gs = gridspec.GridSpec(n, 1, hspace=0.08, top=0.93, bottom=0.06,
                           left=0.07, right=0.96)

    for i, ticker in enumerate(TICKERS):
        ax_price = fig.add_subplot(gs[i])
        ax_vol   = ax_price.twinx()

        close  = df[("Adj Close", ticker)].dropna()
        volume = df[("Volume",    ticker)].dropna()

        ax_vol.bar(volume.index, volume.values, color=COLORS[ticker],
                   alpha=0.18, width=1.2, zorder=1)
        ax_vol.set_ylim(0, volume.max() * 4)
        ax_vol.tick_params(axis="y", colors=STYLE_SUB, labelsize=7.5)
        ax_vol.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax_vol.set_ylabel("Volume", color=STYLE_SUB, fontsize=7.5)
        ax_vol.spines["right"].set_edgecolor(STYLE_GRID)

        ax_price.plot(close.index, close.values, color=COLORS[ticker],
                      linewidth=1.6, zorder=3)
        ax_price.fill_between(close.index, close.values, close.min(),
                              color=COLORS[ticker], alpha=0.07, zorder=2)
        ax_price.set_ylabel(f"{ticker} (USD)", color=COLORS[ticker],
                            fontsize=9, fontweight="bold")
        ax_price.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
        _dark_style(fig, [ax_price])

        if i < n - 1:
            ax_price.set_xticklabels([])
        else:
            ax_price.xaxis.set_major_locator(mdates.MonthLocator())
            ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax_price.xaxis.get_majorticklabels(),
                     rotation=30, ha="right", color=STYLE_TEXT, fontsize=8)

        last_price = close.iloc[-1]
        ax_price.annotate(
            f"${last_price:.2f}",
            xy=(close.index[-1], last_price),
            xytext=(8, 0), textcoords="offset points",
            color=COLORS[ticker], fontsize=8, fontweight="bold", va="center",
        )

    fig.legend(
        handles=[Line2D([0], [0], color=COLORS[t], linewidth=2, label=t)
                 for t in TICKERS],
        loc="upper right", ncol=4, framealpha=0.15,
        labelcolor=STYLE_TEXT, facecolor=STYLE_PANEL, edgecolor=STYLE_GRID,
        fontsize=9, bbox_to_anchor=(0.96, 0.97),
    )
    plt.tight_layout()
    return fig


def plot_price_trend_comparison(df: pd.DataFrame) -> plt.Figure:
    """Chart 1A — Normalized Price Comparison (all on 1 axis)."""
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=STYLE_BG)
    fig.suptitle(
        "Stock Price Trend Comparison — Normalized (Start = 100)\n"
        "AAPL · MSFT · NVDA · TSLA — 2025–2026",
        fontsize=15, fontweight="bold", color=STYLE_TEXT, y=0.97,
    )
    for ticker in TICKERS:
        close = df[("Adj Close", ticker)].dropna()
        ax.plot(close.index, (close / close.iloc[0] * 100).values,
                label=ticker, color=COLORS[ticker], linewidth=2.2)

    ax.set_ylabel("Normalized Price Index (Start = 100)", color=STYLE_TEXT)
    ax.set_xlabel("Date", color=STYLE_TEXT)
    ax.grid(True, alpha=0.35, color=STYLE_GRID)
    ax.set_facecolor(STYLE_PANEL)
    ax.tick_params(axis="x", colors=STYLE_SUB, labelsize=9, rotation=30)
    ax.tick_params(axis="y", colors=STYLE_SUB, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(STYLE_GRID)
    legend = ax.legend(loc="upper left", ncol=4, frameon=True, fontsize=10)
    legend.get_frame().set_facecolor(STYLE_PANEL)
    legend.get_frame().set_edgecolor(STYLE_GRID)
    plt.tight_layout()
    return fig


def plot_rolling_stats(df: pd.DataFrame) -> plt.Figure:
    """Chart 2 — Moving Averages + Bollinger Bands (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=STYLE_BG)
    fig.suptitle("Rolling Statistics: Moving Averages & Bollinger Bands",
                 fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.99)
    axes = axes.flatten()

    for i, ticker in enumerate(TICKERS):
        ax    = axes[i]
        close = df[("Adj Close", ticker)].dropna()
        ma7   = df[("MA7",       ticker)].dropna()
        ma30  = df[("MA30",      ticker)].dropna()
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        c = COLORS[ticker]

        ax.fill_between(bb_mid.index, bb_mid + 2*bb_std, bb_mid - 2*bb_std,
                        alpha=0.12, color=c, label="Bollinger Band")
        ax.plot((bb_mid + 2*bb_std).index, (bb_mid + 2*bb_std).values,
                color=c, linewidth=0.6, linestyle=":", alpha=0.7)
        ax.plot((bb_mid - 2*bb_std).index, (bb_mid - 2*bb_std).values,
                color=c, linewidth=0.6, linestyle=":", alpha=0.7)
        ax.plot(close.index, close.values, color=c,         linewidth=1.2, alpha=0.85, label="Close")
        ax.plot(ma7.index,   ma7.values,   color="#f0e68c",  linewidth=1.4, linestyle="--", label="MA 7")
        ax.plot(ma30.index,  ma30.values,  color="#ff8c00",  linewidth=1.6, label="MA 30")

        ax.set_title(ticker, fontsize=12, fontweight="bold", color=c, pad=6)
        ax.set_ylabel("Price (USD)", fontsize=8.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.2,
                  labelcolor=STYLE_TEXT, facecolor=STYLE_PANEL, edgecolor=STYLE_GRID)

    _dark_style(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_return_distribution(df: pd.DataFrame) -> plt.Figure:
    """Chart 3 — Daily Return Distribution Histogram + KDE (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=STYLE_BG)
    fig.suptitle("Daily Return Distribution — Histogram + KDE",
                 fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.99)
    axes = axes.flatten()

    for i, ticker in enumerate(TICKERS):
        ax      = axes[i]
        returns = df[("Return", ticker)].dropna() * 100
        c       = COLORS[ticker]
        mu, std = returns.mean(), returns.std()

        ax.hist(returns, bins=35, color=c, alpha=0.45,
                density=True, edgecolor="none", label="Histogram")
        kde_x = np.linspace(returns.min(), returns.max(), 300)
        kde   = gaussian_kde(returns, bw_method=0.4)
        ax.plot(kde_x, kde(kde_x), color=c, linewidth=2.2, label="KDE")
        ax.plot(kde_x, norm.pdf(kde_x, mu, std), color="grey",
                linewidth=1.2, linestyle="--", alpha=0.7, label="Normal fit")
        ax.axvline(mu,       color="#555",    linewidth=1.2, alpha=0.8)
        ax.axvline(mu + std, color="#ffd700", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.axvline(mu - std, color="#ffd700", linewidth=0.9, linestyle="--", alpha=0.7)

        stats_txt = (f"μ = {mu:.3f}%\n"
                     f"σ = {std:.3f}%\n"
                     f"Skew = {returns.skew():.2f}\n"
                     f"Kurt = {returns.kurtosis():.2f}")
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=STYLE_TEXT,
                bbox=dict(facecolor=STYLE_BG, alpha=0.6,
                          edgecolor=STYLE_GRID, boxstyle="round,pad=0.4"))
        ax.set_title(ticker, fontsize=12, fontweight="bold", color=c, pad=6)
        ax.set_xlabel("Daily Return (%)", fontsize=8.5)
        ax.set_ylabel("Density", fontsize=8.5)
        ax.legend(fontsize=7.5, framealpha=0.2, labelcolor=STYLE_TEXT,
                  facecolor=STYLE_PANEL, edgecolor=STYLE_GRID)

    _dark_style(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Chart 4 — Correlation Heatmap + Scatter."""
    returns_df = pd.DataFrame({t: df[("Return", t)] for t in TICKERS}).dropna()
    corr = returns_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=STYLE_BG)
    fig.suptitle("Correlation Analysis — Daily Returns",
                 fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.99)

    ax1 = axes[0]
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, ax=ax1, annot=True, fmt=".3f", cmap=cmap,
        vmin=-1, vmax=1, linewidths=1.5, linecolor=STYLE_BG,
        annot_kws={"size": 13, "weight": "bold", "color": STYLE_TEXT},
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax1.set_title("Return Correlation Matrix", fontsize=11,
                  fontweight="bold", color=STYLE_TEXT, pad=10)
    ax1.set_facecolor(STYLE_PANEL)
    ax1.tick_params(colors=STYLE_TEXT, labelsize=10)
    ax1.set_xticklabels(TICKERS, rotation=0,  color=STYLE_TEXT, fontsize=11)
    ax1.set_yticklabels(TICKERS, rotation=90, color=STYLE_TEXT, fontsize=11, va="center")
    cbar = ax1.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(STYLE_TEXT)
    cbar.ax.tick_params(colors=STYLE_TEXT)

    ax2 = axes[1]
    ax2.set_facecolor(STYLE_PANEL)
    ax2.set_title("Return Scatter: AAPL vs Others", fontsize=11,
                  fontweight="bold", color=STYLE_TEXT, pad=10)
    markers = ["o", "s", "^"]
    for other, mrk in zip([t for t in TICKERS if t != "AAPL"], markers):
        x = returns_df["AAPL"] * 100
        y = returns_df[other]  * 100
        ax2.scatter(x, y, c=COLORS[other], alpha=0.45, s=18,
                    marker=mrk, label=f"AAPL vs {other}")
        m, b = np.polyfit(x.dropna(), y.dropna(), 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xl, m * xl + b, color=COLORS[other],
                 linewidth=1.5, linestyle="--", alpha=0.8)
    ax2.axhline(0, color=STYLE_GRID, linewidth=0.8)
    ax2.axvline(0, color=STYLE_GRID, linewidth=0.8)
    ax2.set_xlabel("AAPL Daily Return (%)", fontsize=9)
    ax2.set_ylabel("Other Asset Daily Return (%)", fontsize=9)
    ax2.legend(fontsize=8.5, framealpha=0.2, labelcolor=STYLE_TEXT,
               facecolor=STYLE_BG, edgecolor=STYLE_GRID)
    _dark_style(fig, [ax2])
    for spine in ax1.spines.values():
        spine.set_edgecolor(STYLE_GRID)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_candlestick_grid(df: pd.DataFrame, display_months: int = 3) -> plt.Figure:
    """Chart 5 (Bonus) — Candlestick 2×2 với Volume Overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=STYLE_BG)
    fig.suptitle(
        f"Candlestick Chart with Volume Overlay — Last {display_months} Months\n"
        "AAPL · MSFT · NVDA · TSLA",
        fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.99,
    )
    for idx, ticker in enumerate(TICKERS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        close  = df[("Adj Close", ticker)].dropna()
        volume = df[("Volume",    ticker)].reindex(close.index).fillna(0)

        # Estimate OHLC từ Close + Volatility
        np.random.seed(42)
        vol_pct = df[("Volatility", ticker)].reindex(close.index).ffill().fillna(5) / 100
        open_   = close.shift(1).fillna(close)
        spread  = close * vol_pct * 0.5
        rng     = np.random.uniform(0.4, 0.9, len(close))
        high    = pd.concat([close + spread * rng, close, open_], axis=1).max(axis=1)
        low_    = pd.concat([close - spread * rng, close, open_], axis=1).min(axis=1)

        cutoff = close.index[-1] - pd.DateOffset(months=display_months)
        mask   = close.index >= cutoff
        dates  = close.index[mask]
        o, h, l, c_ = (open_[mask].values, high[mask].values,
                       low_[mask].values,  close[mask].values)
        vol    = volume[mask].values
        dnums  = mdates.date2num(dates)
        width  = (dnums[1] - dnums[0]) * 0.6 if len(dnums) > 1 else 0.6

        ax_v = ax.twinx()
        colors_v = [COLOR_UP if c_[i] >= o[i] else COLOR_DOWN for i in range(len(dates))]
        ax_v.bar(dnums, vol, width=width, color=colors_v, alpha=0.18, zorder=1)
        ax_v.set_ylim(0, vol.max() * 4)
        ax_v.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax_v.set_ylabel("Volume", color=STYLE_SUB, fontsize=7.5)
        ax_v.tick_params(axis="y", colors=STYLE_SUB, labelsize=7.5)
        ax_v.spines["right"].set_edgecolor(STYLE_GRID)

        for i in range(len(dates)):
            color = COLOR_UP if c_[i] >= o[i] else COLOR_DOWN
            ax.plot([dnums[i], dnums[i]], [l[i], h[i]],
                    color=color, linewidth=0.9, zorder=2)
            body_h = abs(c_[i] - o[i])
            if body_h > 0:
                ax.add_patch(mpatches.Rectangle(
                    (dnums[i] - width / 2, min(o[i], c_[i])),
                    width, body_h,
                    facecolor=color, edgecolor=color,
                    linewidth=0.3, zorder=3, alpha=0.9,
                ))

        ax.set_xlim(dnums[0] - 1, dnums[-1] + 1)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        tc = COLORS[ticker]
        ax.annotate(f"${c_[-1]:.2f}", xy=(dnums[-1], c_[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=tc, fontsize=9, fontweight="bold", va="center")
        ax.set_title(ticker, fontsize=12, fontweight="bold", color=tc, pad=6)
        ax.set_ylabel("Price (USD)", fontsize=8.5, color=STYLE_TEXT)
        ax.set_facecolor(STYLE_PANEL)
        ax.tick_params(colors=STYLE_TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE_GRID)
        ax.grid(color=STYLE_GRID, linewidth=0.5, linestyle="--", alpha=0.6, zorder=0)

    fig.legend(
        handles=[mpatches.Patch(color=COLOR_UP,   label="Bullish (Close ≥ Open)"),
                 mpatches.Patch(color=COLOR_DOWN, label="Bearish (Close < Open)")],
        loc="upper right", ncol=1, fontsize=9, framealpha=0.15,
        labelcolor=STYLE_TEXT, facecolor=STYLE_PANEL, edgecolor=STYLE_GRID,
        bbox_to_anchor=(0.99, 0.97),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_volatility_vs_vix(df: pd.DataFrame) -> plt.Figure:
    """Chart 6 — Stock Annualized Volatility (%) vs VIX Index Level."""
    fig, ax1 = plt.subplots(figsize=(16, 7), facecolor=STYLE_BG)
    fig.suptitle(
        "Annualized Volatility vs VIX  —  Comparing Realized vs Implied Market Fear",
        fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.97,
    )

    latest_vol = {}
    for ticker in TICKERS:
        ret = df[("Return", ticker)].dropna()
        ann_vol = ret.rolling(30).std() * np.sqrt(252) * 100
        ann_vol = ann_vol.dropna()
        latest_vol[ticker] = ann_vol.iloc[-1]

        ax1.plot(ann_vol.index, ann_vol.values,
                 color=COLORS[ticker], linewidth=1.8, alpha=0.88,
                 label=ticker, zorder=3)
        ax1.scatter(ann_vol.index[-1], ann_vol.iloc[-1],
                    color=COLORS[ticker], s=50, zorder=5)
        ax1.annotate(
            f"{ann_vol.iloc[-1]:.1f}%",
            xy=(ann_vol.index[-1], ann_vol.iloc[-1]),
            xytext=(8, 0), textcoords="offset points",
            color=COLORS[ticker], fontsize=8.5, fontweight="bold", va="center",
        )

    ax1.set_ylabel("Realized Volatility — Annualized (%)", fontsize=10, color=STYLE_TEXT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    ax2 = ax1.twinx()
    vix = df[("Adj Close", "VIX")].dropna()
    ax2.fill_between(vix.index, vix.values, vix.min(),
                     color="#6b7280", alpha=0.10, zorder=1)
    ax2.plot(vix.index, vix.values,
             color="#374151", linewidth=2.5, linestyle="-",
             alpha=0.80, label="VIX", zorder=2)
    ax2.scatter(vix.index[-1], vix.iloc[-1], color="#374151", s=60, zorder=6)
    ax2.annotate(
        f"VIX {vix.iloc[-1]:.1f}",
        xy=(vix.index[-1], vix.iloc[-1]),
        xytext=(10, 3), textcoords="offset points",
        fontsize=9, fontweight="bold", color="#374151", va="bottom",
    )
    ax2.set_ylabel("VIX Index (Implied Volatility %)", fontsize=10, color=STYLE_SUB)
    ax2.tick_params(axis="y", colors=STYLE_SUB)
    ax2.spines["right"].set_edgecolor(STYLE_GRID)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Regime bands
    for start, end, vix_val in zip(vix.index[:-1], vix.index[1:], vix.values[:-1]):
        if vix_val < 20:
            color_bg = "#d1fae5"
        elif vix_val < 30:
            color_bg = "#fef3c7"
        else:
            color_bg = "#fee2e2"
        ax1.axvspan(start, end, color=color_bg, alpha=0.18, linewidth=0, zorder=0)

    from matplotlib.patches import Patch
    regime_legend = [
        Patch(facecolor="#d1fae5", alpha=0.7, label="VIX < 20  (Calm)"),
        Patch(facecolor="#fef3c7", alpha=0.7, label="VIX 20–30 (Elevated)"),
        Patch(facecolor="#fee2e2", alpha=0.7, label="VIX > 30  (Fear)"),
    ]

    _dark_style(fig, [ax1])
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=8, color=STYLE_TEXT)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h2 + h1 + regime_legend,
        l2 + l1 + [p.get_label() for p in regime_legend],
        loc="upper left", fontsize=8.5, framealpha=0.25,
        labelcolor=STYLE_TEXT, facecolor=STYLE_PANEL, edgecolor=STYLE_GRID,
        ncol=2,
    )

    most_volatile  = max(latest_vol, key=latest_vol.get)
    least_volatile = min(latest_vol, key=latest_vol.get)
    ax1.text(
        0.99, 0.97,
        f"Most volatile: {most_volatile} ({latest_vol[most_volatile]:.1f}%)\n"
        f"Least volatile: {least_volatile} ({latest_vol[least_volatile]:.1f}%)\n"
        f"Current VIX: {vix.iloc[-1]:.1f}",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=8, color=STYLE_TEXT,
        bbox=dict(facecolor=STYLE_BG, edgecolor=STYLE_GRID,
                  boxstyle="round,pad=0.4", alpha=0.8),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_rolling_correlation_nasdaq(df: pd.DataFrame, window: int = 30) -> plt.Figure:
    """Chart 7 — Rolling 30-Day Correlation of Stock Returns vs NASDAQ."""
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=STYLE_BG)
    fig.suptitle(
        f"Rolling {window}-Day Correlation of Daily Returns vs NASDAQ Composite",
        fontsize=14, fontweight="bold", color=STYLE_TEXT, y=0.97,
    )

    band_configs = [
        (0.70,  1.00, "#dcfce7", "High sensitivity (r > 0.7)"),
        (0.30,  0.70, "#fef9c3", "Moderate (0.3–0.7)"),
        (-1.00, 0.30, "#fee2e2", "Low / divergent (r < 0.3)"),
    ]
    for y_lo, y_hi, color, _ in band_configs:
        ax.axhspan(y_lo, y_hi, color=color, alpha=0.20, zorder=0, linewidth=0)

    for ref_y, label_text, x_offset in [
        (0.70, "r = 0.70", 0.01),
        (0.30, "r = 0.30", 0.01),
        (0.00, "r = 0",    0.01),
    ]:
        ax.axhline(ref_y, color=STYLE_SUB, linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=1)
        ax.text(
            ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 0.01,
            ref_y + 0.02, label_text,
            fontsize=7.5, color=STYLE_SUB, alpha=0.8,
            transform=ax.get_yaxis_transform(),
            va="bottom",
        )

    nasdaq_ret  = df[("Return", "NASDAQ")]
    latest_corr = {}

    for ticker in TICKERS:
        stock_ret    = df[("Return", ticker)]
        rolling_corr = stock_ret.rolling(window).corr(nasdaq_ret).dropna()
        latest_corr[ticker] = rolling_corr.iloc[-1]

        ax.plot(rolling_corr.index, rolling_corr.values,
                color=COLORS[ticker], linewidth=2.0, alpha=0.90,
                label=ticker, zorder=3)
        ax.scatter(rolling_corr.index[-1], rolling_corr.iloc[-1],
                   color=COLORS[ticker], s=55, zorder=5)
        ax.annotate(
            f"{rolling_corr.iloc[-1]:.2f}",
            xy=(rolling_corr.index[-1], rolling_corr.iloc[-1]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, fontweight="bold",
            color=COLORS[ticker], va="center",
        )

    ax.set_ylim(-0.5, 1.05)
    ax.set_ylabel("Rolling Correlation Coefficient (r)", fontsize=10, color=STYLE_TEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=8, color=STYLE_TEXT)

    _dark_style(fig, [ax])

    from matplotlib.patches import Patch
    regime_patches = [
        Patch(facecolor=c, alpha=0.5, label=lbl)
        for _, _, c, lbl in band_configs
    ]
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(
        h1 + regime_patches,
        l1 + [p.get_label() for p in regime_patches],
        loc="lower left", fontsize=8.5, framealpha=0.25,
        labelcolor=STYLE_TEXT, facecolor=STYLE_PANEL, edgecolor=STYLE_GRID,
        ncol=2,
    )

    highest = max(latest_corr, key=latest_corr.get)
    lowest  = min(latest_corr, key=latest_corr.get)
    ax.text(
        0.99, 0.97,
        f"Most correlated with NASDAQ: {highest} (r = {latest_corr[highest]:.2f})\n"
        f"Least correlated:  {lowest} (r = {latest_corr[lowest]:.2f})\n"
        f"Window: {window}-day rolling",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=STYLE_TEXT,
        bbox=dict(facecolor=STYLE_BG, edgecolor=STYLE_GRID,
                  boxstyle="round,pad=0.4", alpha=0.8),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Runner ────────────────────────────────────────────────────

def run_visualization():
    """Chạy toàn bộ visualization module, lưu chart ra file."""
    log.info("=== MODULE 3: VISUALIZATION ===")
    os.makedirs(CHART_DIR, exist_ok=True)

    df = _load_df_for_viz(PROCESSED_CSV)
    log.info(f"Data loaded: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")

    charts = [
        ("01_price_volume.png",         plot_price_volume(df)),
        ("01a_price_comparison.png",    plot_price_trend_comparison(df)),
        ("02_rolling_statistics.png",   plot_rolling_stats(df)),
        ("03_returns_distribution.png", plot_return_distribution(df)),
        ("04_correlation_heatmap.png",  plot_correlation_heatmap(df)),
        ("05_candlestick.png",          plot_candlestick_grid(df)),
        ("06_volatility_vs_vix.png",    plot_volatility_vs_vix(df)),
        ("07_corr_with_nasdaq.png",     plot_rolling_correlation_nasdaq(df)),
    ]
    for fname, fig in charts:
        path = os.path.join(CHART_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE_BG)
        plt.close(fig)
        log.info(f"[✓] Saved: {path}")
