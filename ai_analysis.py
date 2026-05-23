"""
ai_analysis.py
==============
Module 4 — AI Analysis (Gemini API)
Nhận long-format DataFrame, gọi Gemini API, trả về báo cáo phân tích.
"""

import os
import logging
from datetime import datetime

import pandas as pd
import google.generativeai as genai

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative financial analyst writing a structured analysis report.
You will be given precise statistical data computed from historical stock price data.
Your job is to write a clear, professional analysis that MUST:
- Reference specific numbers from the data provided (prices, percentages, correlations)
- Never fabricate or estimate figures not present in the data
- Use plain financial language — avoid jargon without explanation
- Be concise but insightful

Structure your response with exactly these five sections:
1. TREND ANALYSIS
2. RISK COMMENTARY
3. CORRELATION & RELATIONSHIP
4. MACRO & MARKET ENVIRONMENT
5. COMPARISON SUMMARY
Each section should be 2-4 paragraphs."""


def _build_user_prompt(stats_text: str) -> str:
    return f"""Analyze the following statistical data for US tech stocks (AAPL, MSFT, TSLA, NVDA)
along with macro indicators:
- VIX = CBOE Volatility Index
- NASDAQ = Nasdaq Composite Index

The dataset ALREADY includes VIX and NASDAQ statistics and correlations.
Use them directly in your macro analysis.

{stats_text}

Write a comprehensive financial analysis covering:
1. TREND ANALYSIS — Which stocks are trending up/down? What do MA7 vs MA30 tell us? Reference specific price levels.
2. RISK COMMENTARY — Which stock is riskiest? Compare annualized volatility figures. Flag any stocks with unusual outlier days.
3. CORRELATION & RELATIONSHIP — Do these stocks move together? What does the correlation matrix reveal? Any surprising pairs?
4. MACRO & MARKET ENVIRONMENT —
    Use the provided VIX and NASDAQ statistics/correlations directly.
    Discuss:
    - Whether periods of elevated VIX coincided with increased volatility in AAPL, MSFT, NVDA, and TSLA
    - Which stocks appeared most sensitive to changes in market sentiment or macro uncertainty
    - Whether the four tech stocks generally moved together with the NASDAQ Composite
    - Any notable divergence between individual stock performance and broader market performance
    Reference specific correlation coefficients, volatility metrics, return trends, and relative performance figures.
    Do NOT mention missing macro data or introduce unsupported external assumptions.
5. COMPARISON SUMMARY — If forced to rank these four stocks by risk-adjusted attractiveness based solely on this data, what would the ranking be and why?

Ground every claim in the exact numbers provided. Do not introduce external market knowledge."""


# ─────────────────────────────────────────────────────────────
# STATS BUILDERS
# ─────────────────────────────────────────────────────────────

def _build_summary_stats(df: pd.DataFrame) -> dict:
    """Tính summary statistics cho từng ticker từ long-format DataFrame."""
    stats = {}
    for ticker in df["Ticker"].unique():
        sub    = df[df["Ticker"] == ticker].dropna(subset=["Return", "Volatility"])
        latest = sub.sort_values("Date").iloc[-1]
        first  = sub.sort_values("Date").iloc[0]
        total_return = ((latest["Close"] - first["Close"]) / first["Close"]) * 100
        recent = sub.tail(30)
        prior  = sub.iloc[-60:-30] if len(sub) >= 60 else sub.head(30)
        stats[ticker] = {
            "current_price":             round(float(latest["Close"]), 2),
            "total_return_pct":          round(total_return, 2),
            "avg_daily_return_pct":      round(float(sub["Return"].mean()), 3),
            "return_std_pct":            round(float(sub["Return"].std()), 3),
            "annualized_volatility_pct": round(float(sub["Volatility"].mean()), 2),
            "latest_volatility_pct":     round(float(latest["Volatility"]), 2),
            "ma7":                       round(float(latest["MA7"]), 2),
            "ma30":                      round(float(latest["MA30"]), 2),
            "price_vs_ma30":             "above" if latest["Close"] > latest["MA30"] else "below",
            "recent_30d_avg_return":     round(float(recent["Return"].mean()), 3),
            "prior_30d_avg_return":      round(float(prior["Return"].mean()), 3),
            "date_range": f"{sub['Date'].min().date()} to {sub['Date'].max().date()}",
        }
    return stats


def _build_correlation_matrix(df: pd.DataFrame) -> dict:
    """Tính correlation matrix của daily returns."""
    pivot = df.pivot_table(index="Date", columns="Ticker", values="Return")
    return pivot.corr().round(3).to_dict()


def _format_stats_for_prompt(stats: dict, corr: dict) -> str:
    """Format stats + correlation thành text để đưa vào prompt."""
    lines = ["=== STOCK SUMMARY STATISTICS ===\n"]
    for ticker, s in stats.items():
        lines += [
            f"[{ticker}]",
            f"  Current Price:          ${s['current_price']}",
            f"  Total Return (period):  {s['total_return_pct']}%",
            f"  Avg Daily Return:       {s['avg_daily_return_pct']}%",
            f"  Return Std Dev:         {s['return_std_pct']}%",
            f"  Annualized Volatility:  {s['annualized_volatility_pct']}%",
            f"  Latest Volatility:      {s['latest_volatility_pct']}%",
            f"  MA7:                    ${s['ma7']}",
            f"  MA30:                   ${s['ma30']}",
            f"  Price vs MA30:          {s['price_vs_ma30']}",
            f"  Recent 30d Avg Return:  {s['recent_30d_avg_return']}%",
            f"  Prior 30d Avg Return:   {s['prior_30d_avg_return']}%",
            f"  Date Range:             {s['date_range']}",
            "",
        ]
    lines.append("=== RETURN CORRELATIONS ===")
    macro_assets = ["VIX", "NASDAQ"]
    for t1 in corr:
        for t2 in corr[t1]:
            val      = corr[t1][t2]
            relation = " (MACRO RELATIONSHIP)" if t1 in macro_assets or t2 in macro_assets else ""
            lines.append(f"Correlation({t1}, {t2}) = {val}{relation}")
    lines.append("\n=== MACRO INDICATORS INCLUDED IN DATASET ===")
    lines.append("VIX = CBOE Volatility Index")
    lines.append("NASDAQ = Nasdaq Composite Index")
    lines.append("These indicators are part of the dataset and should be used in macro analysis.")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────

def run_ai_analysis(df_long: pd.DataFrame, gemini_model: str, ai_dir: str) -> str:
    """
    Chạy AI analysis với Gemini.

    Parameters
    ----------
    df_long      : long-format DataFrame (Date, Ticker, Close, Return, MA7, MA30, Volatility)
    gemini_model : tên model Gemini (vd: "gemini-2.5-flash")
    ai_dir       : thư mục lưu output (vd: "output/ai")

    Returns
    -------
    analysis_text : str — nội dung báo cáo AI
    """
    log.info("=== MODULE 4: AI ANALYSIS (GEMINI) ===")

    # Build stats & prompt
    stats      = _build_summary_stats(df_long)
    corr       = _build_correlation_matrix(df_long)
    stats_text = _format_stats_for_prompt(stats, corr)

    # Lưu summary stats
    os.makedirs(ai_dir, exist_ok=True)
    stats_path = os.path.join(ai_dir, "summary_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    log.info(f"Summary stats saved → {stats_path}")

    # Kiểm tra API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY không tìm thấy.\n"
            "Tạo file key.env với nội dung: GEMINI_API_KEY=your-key-here"
        )

    # Gọi Gemini API
    genai.configure(api_key=api_key)
    model       = genai.GenerativeModel(gemini_model)
    full_prompt = SYSTEM_PROMPT + "\n\n" + _build_user_prompt(stats_text)

    log.info(f"Calling Gemini API (model: {gemini_model}) ...")
    response      = model.generate_content(full_prompt)
    analysis_text = response.text
    log.info(f"AI analysis received ({len(analysis_text)} chars).")

    # Lưu output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(ai_dir, f"ai_analysis_{timestamp}.txt")
    latest    = os.path.join(ai_dir, "ai_analysis_latest.txt")

    for path in (out_path, latest):
        with open(path, "w", encoding="utf-8") as f:
            if path == out_path:
                f.write(f"=== AI FINANCIAL ANALYSIS ===\n")
                f.write(f"Generated: {timestamp}\nModel: {gemini_model}\n\n")
            f.write(analysis_text)

    log.info(f"AI analysis saved → {out_path}")
    return analysis_text
