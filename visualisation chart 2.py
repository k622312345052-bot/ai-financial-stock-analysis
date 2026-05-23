"""
visualization.py
================
Module 3 — Visualization Chart 2
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



# ── Runner ────────────────────────────────────────────────────

def run_visualization():
    """Chạy toàn bộ visualization module, lưu chart ra file."""
    log.info("=== MODULE 3: VISUALIZATION ===")
    os.makedirs(CHART_DIR, exist_ok=True)

    df = _load_df_for_viz(PROCESSED_CSV)
    log.info(f"Data loaded: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")

    charts = [
        ("04_correlation_heatmap.png",  plot_correlation_heatmap(df)),
        ("05_candlestick.png",          plot_candlestick_grid(df))
    ]
    for fname, fig in charts:
        path = os.path.join(CHART_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE_BG)
        plt.close(fig)
        log.info(f"[✓] Saved: {path}")
