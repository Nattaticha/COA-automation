"""Document Layout Analysis: split each page into TEXT / TITLE / TABLE / FIGURE zones.

Native PDF : PyMuPDF text blocks + page.find_tables()      (no OCR)
Scanned PDF: layout detector (PaddleOCR PP-DocLayout or OpenCV grid lines)
             -> text zones go to an OCR engine, table zones go to a table reader
All bboxes are pixels at config.DPI: [x0, y0, x1, y1].
"""
from bisect import bisect_right
from dataclasses import asdict, dataclass, field

import cv2
import pymupdf as fitz
import numpy as np

from .config import DPI
from .engines import PADDLE_CPU, Engine, Line, _get, group_rows, join_lines, mean_conf
from .preprocess import to_gray


@dataclass
class Zone:
    page: int
    kind: str                      # text | title | table | figure
    bbox: tuple
    text: str = ""
    cells: list | None = None      # table: 2D list of strings
    conf: float | None = None
    source: str = "native"         # native | ocr:<engine>
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def pt2px(rect, dpi=DPI):
    s = dpi / 72
    return tuple(int(round(v * s)) for v in rect)


def overlap_ratio(a, b) -> float:
    """intersection / area(a)"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _is_red(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return r > 150 and g < 100 and b < 100


# ============================================================ native PDF
def native_tables(page: fitz.Page) -> list[tuple[fitz.Rect, list[list[str]]]]:
    try:
        # use_layout=False: PyMuPDF 2.x's onnx layout model crashes (access violation) on this PC
        try:
            tabs = page.find_tables(use_layout=False)
        except TypeError:  # PyMuPDF < 2.0 has no use_layout
            tabs = page.find_tables()
    except Exception:
        return []
    out = []
    for t in tabs.tables:
        rows = [[(c or "").strip() for c in r] for r in t.extract()]
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            out.append((fitz.Rect(t.bbox), rows))
    return out


def zones_native(page: fitz.Page, page_no: int, dpi: int = DPI) -> list[Zone]:
    tables = native_tables(page)
    zones = [Zone(page_no, "table", pt2px(r, dpi), cells=cells) for r, cells in tables]
    t_rects = [tuple(r) for r, _ in tables]
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        rect = tuple(b["bbox"])
        if any(overlap_ratio(rect, t) > 0.5 for t in t_rects):
            continue
        lines, sizes = [], []
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if not _is_red(s["color"])]   # drop red watermark text
            txt = "".join(s["text"] for s in spans).strip()
            if txt:
                lines.append(txt)
                sizes += [s["size"] for s in spans]
        if lines:
            kind = "title" if sizes and max(sizes) >= 13 else "text"
            zones.append(Zone(page_no, kind, pt2px(rect, dpi), text="\n".join(lines)))
    return sorted(zones, key=lambda z: (z.bbox[1], z.bbox[0]))


# ============================================================ layout detectors (scanned)
def _line_masks(gray, h_div=30, v_div=40, min_len=80):
    """Ruling-line masks. min_len (px @300 DPI) must exceed letter height, otherwise letter stems
    (P, L, d ~30 px) are taken for vertical lines."""
    inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    h, w = gray.shape
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(min_len, w // h_div), 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(min_len, h // v_div))))
    return horiz, vert


class OpenCVLayout:
    """Finds bordered tables from ruling lines; everything between tables is a text band.
    Cannot see borderless tables – that is exactly what compare_models measures."""
    name = "opencv"

    def detect(self, img) -> list[tuple[str, float, tuple]]:
        gray = to_gray(img)
        h, w = gray.shape
        horiz, vert = _line_masks(gray)
        mask = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tables = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 0.3 * w and bh > 0.03 * h:
                n_hlines = len(_runs(np.where(horiz[y:y + bh, x:x + bw].sum(1) / 255 > 0.5 * bw)[0]))
                if n_hlines >= 2:
                    tables.append((x, y, x + bw, y + bh))
        tables.sort(key=lambda b: b[1])
        out = [("table", 1.0, t) for t in tables]
        y = 0
        for t in tables + [(0, h, w, h)]:           # text bands between tables
            if t[1] - y > 20:
                out.append(("text", 1.0, (0, y, w, t[1])))
            y = max(y, t[3])
        return out


PADDLE_LABEL_MAP = {
    "table": "table",
    "doc_title": "title", "paragraph_title": "title", "table_title": "title", "figure_title": "title",
    "chart_title": "title",
    "image": "figure", "chart": "figure", "seal": "figure", "header_image": "figure", "footer_image": "figure",
    "formula": "figure",
}


class PaddleLayout:
    name = "paddle_layout"

    def __init__(self, model_name=None):
        from paddleocr import LayoutDetection
        from .config import PADDLE_LAYOUT_MODEL
        self.model_name = model_name or PADDLE_LAYOUT_MODEL
        self.m = LayoutDetection(model_name=self.model_name, **PADDLE_CPU)

    def detect(self, img):
        out = []
        for res in self.m.predict(img, batch_size=1):
            for b in _get(res, "boxes", []) or []:
                out.append((str(b["label"]), float(b["score"]), tuple(int(v) for v in b["coordinate"])))
        return out


def get_layout(name: str):
    return OpenCVLayout() if name == "opencv" else PaddleLayout()


# ============================================================ table readers (scanned)
def _runs(idx: np.ndarray) -> list[float]:
    """Cluster consecutive indices -> centre of each run."""
    if len(idx) == 0:
        return []
    groups, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 2:
            groups.append((start + a) / 2)
            start = b
    groups.append((start + idx[-1]) / 2)
    return groups


def grid_separators(img):
    gray = to_gray(img)
    h, w = gray.shape
    horiz, vert = _line_masks(gray, h_div=15, v_div=15)
    rows = _runs(np.where(horiz.sum(1) / 255 > 0.5 * w)[0])
    cols = _runs(np.where(vert.sum(0) / 255 > 0.5 * h)[0])
    return rows, cols


def column_gaps(lines: list[Line], width: int) -> list[float]:
    """Borderless table: columns = vertical white gaps not covered by any text segment."""
    cover = np.zeros(width + 1, bool)
    for l in lines:
        cover[max(0, l.bbox[0]):min(width, l.bbox[2]) + 1] = True
    min_gap = max(8, int(np.median([l.bbox[3] - l.bbox[1] for l in lines])))
    seps, x = [0.0], 0
    while x <= width:
        if not cover[x]:
            s = x
            while x <= width and not cover[x]:
                x += 1
            if s > 0 and x <= width and x - s >= min_gap:
                seps.append((s + x) / 2)
        x += 1
    seps.append(float(width))
    return seps


def _assign(v, seps):
    return min(max(bisect_right(seps, v) - 1, 0), len(seps) - 2)


def ink_column_gaps(img, min_gap: int) -> list[float]:
    """Column separators from the IMAGE: vertical white bands crossing every text row.
    Works even when the OCR engine returns a whole row as one segment (PaddleOCR does)."""
    gray = to_gray(img)
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    horiz, vert = _line_masks(gray, h_div=15, v_div=15)
    ink = cv2.subtract(cv2.subtract(inv, horiz), vert)           # text only, no ruling lines
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop scan speckle
    h, w = ink.shape
    cover = (ink > 0).sum(0) > max(2, 0.004 * h)
    seps, x = [0.0], 0
    xs = np.where(cover)[0]
    if len(xs) == 0:
        return [0.0, float(w)]
    x0, x1 = xs[0], xs[-1]
    x = x0
    while x <= x1:
        if not cover[x]:
            s = x
            while x <= x1 and not cover[x]:
                x += 1
            if x - s >= min_gap:
                seps.append((s + x) / 2)
        x += 1
    seps.append(float(w))
    return seps


class GridTableReader:
    """Engine-agnostic table reader.
    rows    : visual text rows from the engine's detection boxes (ruling lines only if they match)
    columns : vertical ruling lines if present, else white gaps in the image
    cells   : each (row x column) crop is recognised separately with engine.recognize_line()."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.name = f"grid+{engine.name}"

    def read(self, img) -> tuple[list[list[str]], float]:
        from .preprocess import deskew
        img = deskew(img)   # a 1° tilt shifts a 10-row table by ~7 px and closes the column gaps
        lines = self.engine.recognize_lines(img)
        if not lines:
            return [], 0.0
        h, w = img.shape[:2]
        text_h = int(np.median([l.bbox[3] - l.bbox[1] for l in lines]))
        rseps, cseps = grid_separators(img)
        if len(cseps) < 3:   # word spaces ~0.3 x text height; column gaps are wider
            cseps = ink_column_gaps(img, min_gap=max(8, int(0.45 * text_h)))
        rows = group_rows(lines)
        bands = [(min(l.bbox[1] for l in r), max(l.bbox[3] for l in r)) for r in rows]
        if len(rseps) - 1 >= len(rows):        # fully ruled table: one ruling line per row
            bands = [(int(a), int(b)) for a, b in zip(rseps[:-1], rseps[1:]) if b - a > 0.6 * text_h]
        grid, confs = [], []
        pad = 4
        for y0, y1 in bands:
            row = []
            for x0, x1 in zip(cseps[:-1], cseps[1:]):
                # tight crop = detected text inside this cell (clipped to the cell), NOT the whole cell:
                # whole-cell crops carry ruling lines + white space that derail rec-only models
                segs = [l for l in lines if y0 - 2 <= (l.bbox[1] + l.bbox[3]) / 2 <= y1 + 2
                        and l.bbox[0] < x1 and l.bbox[2] > x0]
                if not segs:
                    row.append("")
                    continue
                cx0 = max(int(x0), min(l.bbox[0] for l in segs)) - pad
                cx1 = min(int(x1), max(l.bbox[2] for l in segs)) + pad
                cy0 = min(l.bbox[1] for l in segs) - pad
                cy1 = max(l.bbox[3] for l in segs) + pad
                cell = img[max(0, cy0):min(h, cy1), max(0, cx0):min(w, cx1)]
                if cell.size == 0:
                    row.append("")
                    continue
                txt, c = self.engine.recognize_line(cell)
                row.append(txt.strip())
                confs.append(c)
            grid.append(row)
        grid = [r for r in grid if any(r)]
        keep = [j for j in range(len(cseps) - 1) if any(r[j] for r in grid)]
        return [[r[j] for j in keep] for r in grid], (float(np.mean(confs)) if confs else 0.0)


class PaddleTableReader:
    """PaddleOCR table-structure model (SLANet) – an alternative table 'model' to compare."""
    name = "paddle_table"

    def __init__(self):
        from paddleocr import TableRecognitionPipelineV2
        try:
            self.p = TableRecognitionPipelineV2(**PADDLE_CPU, use_doc_orientation_classify=False, use_doc_unwarping=False,
                                                use_layout_detection=False)
        except TypeError:
            self.p = TableRecognitionPipelineV2(**PADDLE_CPU)

    def read(self, img):
        from io import StringIO
        import pandas as pd
        for res in self.p.predict(img):
            tl = _get(res, "table_res_list", []) or []
            if tl:
                html = tl[0]["pred_html"]
                df = pd.read_html(StringIO(html), header=None)[0].fillna("").astype(str)
                return df.values.tolist(), 1.0
        return [], 0.0


def get_table_reader(name: str, engine: Engine):
    return PaddleTableReader() if name == "paddle_table" else GridTableReader(engine)


# ============================================================ scanned page
def zones_scanned(img, page_no, layout, engine: Engine, table_reader) -> list[Zone]:
    zones = []
    for label, score, (x0, y0, x1, y1) in layout.detect(img):
        kind = PADDLE_LABEL_MAP.get(label, "text")
        crop = img[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        z = Zone(page_no, kind, (x0, y0, x1, y1), source=f"ocr:{engine.name}", meta={"label": label, "score": score})
        if kind == "table":
            z.cells, z.conf = table_reader.read(crop)
        elif kind != "figure":
            lines = engine.recognize_lines(crop)
            z.text, z.conf = join_lines(lines), mean_conf(lines)
        zones.append(z)
    return sorted(zones, key=lambda z: (z.bbox[1], z.bbox[0]))
