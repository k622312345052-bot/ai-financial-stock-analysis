"""
data_processing.py
==================
Module 2 — Data Processing
Feature engineering: Daily Return, MA7, MA30, Annualized Volatility.
"""

import os
import logging

import numpy as np
import pandas as pd

from config import TICKERS, MACRO_TICKERS, DATA_DIR, PROCESSED_CSV

log = logging.getLogger(__name__)


def process_data(df: pd.DataFrame):
    """
    Feature engineering trên raw DataFrame.

    Trả về:
      df_wide : DataFrame wide-format với MultiIndex columns (dùng cho visualization)
      df_long : DataFrame long-format                         (dùng cho AI analysis)
    """
    log.info("=== MODULE 2: DATA PROCESSING ===")

    ALL_ANALYSIS_TICKERS = TICKERS + list(MACRO_TICKERS.keys())

    for ticker in ALL_ANALYSIS_TICKERS:
        daily_return = df["Adj Close", ticker].pct_change()

        df["Return", ticker] = daily_return

        df["MA7", ticker] = (
            df["Adj Close", ticker]
            .rolling(window=7)
            .mean()
        )

        df["MA30", ticker] = (
            df["Adj Close", ticker]
            .rolling(window=30)
            .mean()
        )

        # 30-day rolling annualized volatility
        df["Volatility", ticker] = (
            daily_return
            .rolling(window=30)
            .std()
            * np.sqrt(252)
        )

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # ── Save wide-format ──────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_CSV)
    log.info(f"Processed data saved → {PROCESSED_CSV}")
    log.info(f"Shape: {df.shape}")

    # ── Build long-format for AI module ──────────────────────
    all_rows = []
    for ticker in ALL_ANALYSIS_TICKERS:
        if ticker in df["Volume"].columns:
            volume_values = df[("Volume", ticker)].values
        else:
            volume_values = np.nan

        temp = pd.DataFrame({
            "Date":       pd.to_datetime(df[("Date", "")]),
            "Ticker":     ticker,
            "Close":      df[("Adj Close",  ticker)].values,
            "Volume":     volume_values,
            "Return":     df[("Return",     ticker)].values,
            "MA7":        df[("MA7",        ticker)].values,
            "MA30":       df[("MA30",       ticker)].values,
            "Volatility": df[("Volatility", ticker)].values,
        })
        all_rows.append(temp)

    df_long = pd.concat(all_rows, ignore_index=True)
    df_long = df_long.dropna(subset=["Return", "Volatility"])
    log.info(f"Long-format rows: {len(df_long)}")

    return df, df_long
