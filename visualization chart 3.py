"""
visualization.py
================
Module 3 — Visualization Chart 3
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
        ("06_volatility_vs_vix.png",    plot_volatility_vs_vix(df)),
        ("07_corr_with_nasdaq.png",     plot_rolling_correlation_nasdaq(df))
    ]
    for fname, fig in charts:
        path = os.path.join(CHART_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE_BG)
        plt.close(fig)
        log.info(f"[✓] Saved: {path}")
