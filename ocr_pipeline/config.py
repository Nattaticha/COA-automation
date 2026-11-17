"""Central settings. Edit here instead of inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"                # put historical COA / RD Spec PDFs here (sub-folders OK)
LABELED = DATA / "labeled"        # PPOCRLabel output for real scanned pages
TO_LABEL = DATA / "to_label"      # scanned pages exported for labeling
DATASET = DATA / "dataset"        # generated: rec/, tables/, pages/, split.json
RESULTS = ROOT / "results"        # compare runs
MODELS = ROOT / "models"          # pretrained + fine-tuned weights

DPI = 300
NATIVE_MIN_CHARS = 50             # page with >= this many embedded chars = native PDF
SPLIT = (0.70, 0.15, 0.15)        # train / val / test, split by SUPPLIER
SEED = 42
MAX_LINE_CHARS = 40               # longer text lines are chunked for rec training
SCAN_VARIANTS = 2                 # simulated-scan copies per native crop

_TESS = [Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
         Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"]   # winget user install
TESSERACT_CMD = str(next((p for p in _TESS if p.exists()), _TESS[0]))
TESSERACT_LANG = "eng"            # COA content to OCR is English (user, 2026-11-16)
EASYOCR_LANGS = ("en",)

# PaddleOCR 3.x model names (mobile = fast on CPU). Fine-tune starts from PADDLE_REC_MODEL.
PADDLE_DET_MODEL = "PP-OCRv5_mobile_det"
PADDLE_REC_MODEL = "PP-OCRv5_mobile_rec"
PADDLE_LAYOUT_MODEL = "PP-DocLayout-M"
PADDLE_FT_BASE = "th_PP-OCRv5_mobile_rec"   # best baseline on English COA lines (2026-11-16) -> fine-tune this
PADDLE_FT_DIR = MODELS / "paddle_rec_ft" / "inference"
