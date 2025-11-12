"""
İZİNSİZ DOKUNMA!!!!!!!!!!
VERİ SETİNİN İSMİNİ Default_Data yap
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
VISUALIZE_DIR = BASE_DIR / "visualize"
REPORTS_DIR = BASE_DIR / "reports"

for folder in [DATA_DIR, VISUALIZE_DIR, REPORTS_DIR]:
    folder.mkdir(exist_ok=True)

METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = VISUALIZE_DIR / "figure"

METRICS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

RAW_DATA_FILE = DATA_DIR / "Default_Data.csv"
PROCESSED_DATA_FILE = DATA_DIR / "Processed_Data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2