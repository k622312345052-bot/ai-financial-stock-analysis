# FinAgent — AI-Powered US Tech Stock Analysis

> **Course:** IT Application in Banking and Finance  
> **Assessment:** Midterm Hackathon-Like Project  
> **Deadline:** 27th May 2026

An end-to-end AI financial data agent that autonomously collects stock market data, cleans and processes it, generates professional visualizations, and delivers AI-powered analytical summaries using the Gemini API.

---

## Overview

FinAgent tracks **4 major US tech stocks** — `AAPL`, `MSFT`, `NVDA`, `TSLA` — alongside macro indicators (`VIX`, `NASDAQ`) over a configurable date range. The pipeline runs fully automated from data collection to AI-written analysis in a single command.

---

## Project Structure

```
FinAgent/
├── main.py               # Full pipeline entry point
├── key.env               # API key config (not committed)
├── .env.example          # Template for API key setup
├── requirements.txt      # Python dependencies
├── README.md
├── data/
│   ├── data_raw.csv      # Raw downloaded data
│   └── data_processed.csv
└── output/
    ├── charts/           # All saved chart images
    │   ├── 01_price_volume.png
    │   ├── 01a_price_comparison.png
    │   ├── 02_rolling_statistics.png
    │   ├── 03_returns_distribution.png
    │   ├── 04_correlation_heatmap.png
    │   ├── 05_candlestick.png
    │   ├── 06_volatility_vs_vix.png
    │   └── 07_corr_with_nasdaq.png
    └── ai/
        ├── summary_stats.txt
        └── ai_analysis_latest.txt
```

---

## Pipeline Modules

### Module 1 — Data Collection

Fetches OHLCV data from Yahoo Finance via `yfinance` for the 4 tickers plus macro indicators (VIX, NASDAQ Composite). Handles missing values via `dropna()` with logging, removes duplicates, and saves raw data to `data/data_raw.csv`.

**Data sources:** Yahoo Finance (free, no API key required)

### Module 2 — Data Processing

Computes feature engineering on top of the raw adjusted close prices:

- **Daily Returns** — percentage change day-over-day
- **MA7 / MA30** — 7-day and 30-day rolling moving averages
- **Annualized Volatility** — 30-day rolling standard deviation of returns × √252

Outputs both a wide-format DataFrame (for visualization) and a long-format DataFrame (for AI analysis). Saved to `data/data_processed.csv`.

### Module 3 — Visualization

Generates 8 charts saved to `output/charts/`:

| File | Description |
|---|---|
| `01_price_volume.png` | Price trend + volume overlay (4-panel) |
| `01a_price_comparison.png` | Normalized price comparison (base = 100) |
| `02_rolling_statistics.png` | Moving averages + Bollinger Bands (2×2 grid) |
| `03_returns_distribution.png` | Daily return histogram + KDE + Normal fit |
| `04_correlation_heatmap.png` | Correlation heatmap + AAPL scatter plot |
| `05_candlestick.png` | *(Bonus)* Candlestick with volume overlay (last 3 months) |
| `06_volatility_vs_vix.png` | Realized volatility vs VIX with regime bands |
| `07_corr_with_nasdaq.png` | Rolling 30-day correlation vs NASDAQ |

### Module 4 — AI Analysis (Gemini)

Sends structured summary statistics and correlation data to **Google Gemini 2.5 Flash** via the `google-generativeai` SDK. The model produces a 5-section analysis report:

1. Trend Analysis
2. Risk Commentary
3. Correlation & Relationship
4. Macro & Market Environment
5. Comparison Summary

All outputs are saved to `output/ai/` with a timestamped filename and a `ai_analysis_latest.txt` copy.

---

## Setup & Installation

### 1. Install dependencies

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy google-generativeai python-dotenv
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Copy the example env file and fill in your Gemini API key:

```bash
cp .env.example key.env
```

Edit `key.env`:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

**Never commit `key.env` to Git.** It is listed in `.gitignore`.

### 3. Run the pipeline

```bash
python main.py
```

The full pipeline — data collection → processing → visualization → AI analysis — runs end-to-end and prints the AI analysis to the console on completion.

---

## Configuration

Edit the constants at the top of `main.py` to customize the pipeline:

```python
TICKERS    = ["AAPL", "MSFT", "NVDA", "TSLA"]   # Stocks to track
START_DATE = "2025-01-01"                          # Historical start date
GEMINI_MODEL = "gemini-2.5-flash"                  # Gemini model to use
```

---

## Requirements

```
yfinance
pandas
numpy
matplotlib
seaborn
scipy
google-generativeai
python-dotenv
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (required for Module 4) |

See `.env.example` for the template. Load via `key.env` in the project root.

---

## Notes

- Yahoo Finance data may have a slight delay for real-time prices.
- Candlestick OHLC values for Chart 5 are **estimated** from Adj Close + rolling volatility, since `yfinance` intraday OHLC and adjusted prices can differ; treat as indicative only.
- Gemini API free tier has rate limits — for large date ranges, occasional retries may occur.
- The pipeline requires an internet connection for both data collection (yfinance) and AI analysis (Gemini API).
