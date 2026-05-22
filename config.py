"""
config.py
=========
Shared constants used across all FinAgent modules.
"""

import os
from datetime import datetime

# ── Tickers ───────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
MACRO_TICKERS = {
    "VIX": "^VIX",
    "NASDAQ": "^IXIC",
}

# ── Date range ────────────────────────────────────────────────
START_DATE = "2025-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# ── File paths ────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "output"
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
AI_DIR     = os.path.join(OUTPUT_DIR, "ai")

RAW_CSV       = os.path.join(DATA_DIR, "data_raw.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "data_processed.csv")

# ── Gemini ────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Chart styling ─────────────────────────────────────────────
COLORS = {
    "AAPL": "#1f77b4",
    "MSFT": "#2ca02c",
    "NVDA": "#9467bd",
    "TSLA": "#d62728",
}
STYLE_BG    = "#ffffff"
STYLE_PANEL = "#f9fafb"
STYLE_GRID  = "#e5e7eb"
STYLE_TEXT  = "#111827"
STYLE_SUB   = "#6b7280"
COLOR_UP    = "#16a34a"
COLOR_DOWN  = "#dc2626"
