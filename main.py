"""
main.py
=======
FinAgent — AI-Powered US Tech Stock Analysis
Chạy toàn bộ pipeline từ đầu đến cuối:
  1. Data Collection   → data_collection.py
  2. Data Processing   → data_processing.py
  3. Visualization     → visualization.py
  4. AI Analysis       → ai_analysis.py

Cách chạy:
  pip install -r requirements.txt
  Tạo file key.env với nội dung:  GEMINI_API_KEY=your-key-here
  python main.py
"""

import logging
import warnings
from dotenv import load_dotenv

# ── Import từ các module riêng biệt ──────────────────────────
from config import DATA_DIR, CHART_DIR, AI_DIR, GEMINI_MODEL
from data_collection import collect_data
from data_processing import process_data
from visualization import run_visualization
from ai_analysis import run_ai_analysis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  FinAgent — US Tech Stock Analysis Pipeline")
    print("=" * 60 + "\n")

    # Load API key từ key.env
    load_dotenv("key.env")

    # 1. Data Collection
    df_raw = collect_data()

    # 2. Data Processing
    df_wide, df_long = process_data(df_raw)

    # 3. Visualization
    run_visualization()

    # 4. AI Analysis
    analysis = run_ai_analysis(df_long, GEMINI_MODEL, AI_DIR)

    print("\n" + "=" * 60)
    print("  === AI ANALYSIS OUTPUT ===")
    print("=" * 60)
    print(analysis)

    print("\n" + "=" * 60)
    print("  Pipeline hoàn thành!")
    print(f"  Data  : {DATA_DIR}/")
    print(f"  Charts: {CHART_DIR}/")
    print(f"  AI    : {AI_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
