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
    ]
    for fname, fig in charts:
        path = os.path.join(CHART_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE_BG)
        plt.close(fig)
        log.info(f"[✓] Saved: {path}")
