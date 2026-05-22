"""
data_collection.py
==================
Module 1 — Data Collection
Tải dữ liệu OHLCV từ Yahoo Finance cho 4 tickers + macro indicators.
"""

import os
import logging

import pandas as pd
import yfinance as yf

from config import TICKERS, MACRO_TICKERS, START_DATE, END_DATE, DATA_DIR, RAW_CSV

log = logging.getLogger(__name__)


def collect_data() -> pd.DataFrame:
    """
    Tải dữ liệu OHLCV từ Yahoo Finance cho 4 tickers.
    Trả về DataFrame wide-format với MultiIndex columns.
    Lưu raw data vào data/data_raw.csv.
    """
    log.info("=== MODULE 1: DATA COLLECTION ===")
    log.info(f"Tickers : {TICKERS}")
    log.info(f"Period  : {START_DATE} → {END_DATE}")

    all_tickers = TICKERS + list(MACRO_TICKERS.values())
    raw = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=False,
    )

    adj_close = raw["Adj Close"].copy()
    adj_close = adj_close.rename(columns={
        "^VIX":  "VIX",
        "^IXIC": "NASDAQ",
    })
    volume = raw["Volume"][TICKERS].copy()

    df = pd.concat(
        {
            "Adj Close": adj_close,
            "Volume":    volume,
        },
        axis=1,
    )

    df = df.reset_index()
    df.columns = pd.MultiIndex.from_tuples([
        ("Date", "") if col == "Date" else col
        for col in df.columns
    ])

    # ── Cleaning ──────────────────────────────────────────────
    missing = df.isna().sum().sum()
    log.info(f"Rows loaded    : {len(df)}")
    log.info(f"Missing values : {missing}")
    if missing > 0:
        df = df.dropna()
        log.info(f"Rows after dropna: {len(df)}")

    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Duplicates removed: {before - len(df)}")

    df = df.sort_values(("Date", ""))

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_CSV)
    log.info(f"Raw data saved → {RAW_CSV}")

    return df
