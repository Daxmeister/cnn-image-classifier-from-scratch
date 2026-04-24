from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
DATA = ROOT / "data"

RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"

TEST_UTILS = ROOT / "tests" / "utils"