"""OCR engine adapters with one interface, so every model is compared the same way.

recognize_lines(img) -> list[Line]   text segments (word/phrase) with bbox + confidence (0..1)
recognize_line(img)  -> (text, conf) for a single cropped text line (used for CER/WER)
"""
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np

from . import config
from .preprocess import pad, to_bgr


@dataclass
class Line:
    text: str
    conf: float
    bbox: tuple  # x0, y0, x1, y1 (pixels)


def group_rows(lines: list[Line], tol: float = 0.6) -> list[list[Line]]:
    """Group segments into visual rows by y-centre, each row sorted left->right."""
    if not lines:
        return []
    med_h = float(np.median([max(1, l.bbox[3] - l.bbox[1]) for l in lines]))
    yc = lambda l: (l.bbox[1] + l.bbox[3]) / 2
    rows: list[list[Line]] = []
    for l in sorted(lines, key=yc):
        if rows and abs(yc(l) - np.mean([yc(p) for p in rows[-1]])) <= tol * med_h:
            rows[-1].append(l)
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l.bbox[0]) for r in rows]


def join_lines(lines: list[Line]) -> str:
    return "\n".join(" ".join(l.text for l in row) for row in group_rows(lines))


def mean_conf(lines: list[Line]) -> float:
    return float(np.mean([l.conf for l in lines])) if lines else 0.0


class Engine:
    name = "base"

    def recognize_lines(self, img) -> list[Line]:
        raise NotImplementedError

    def recognize_line(self, img) -> tuple[str, float]:
        lines = self.recognize_lines(pad(img))
        return " ".join(l.text for l in sorted(lines, key=lambda l: l.bbox[0])), mean_conf(lines)

    def info(self) -> str:
        return self.name


# ---------------------------------------------------------------- Tesseract
class TesseractEngine(Engine):
    name = "tesseract"

    def __init__(self, lang: str = config.TESSERACT_LANG):
        import pytesseract
        cmd = config.TESSERACT_CMD if Path(config.TESSERACT_CMD).exists() else shutil.which("tesseract")
        if not cmd:
            raise RuntimeError("ไม่พบ tesseract.exe – ติดตั้ง Tesseract (UB Mannheim) หรือแก้ TESSERACT_CMD ใน config.py")
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.pt, self.lang = pytesseract, lang
        self.version = str(pytesseract.get_tesseract_version())

    def _words(self, img, psm: int) -> list[Line]:
        rgb = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB)
        d = self.pt.image_to_data(rgb, lang=self.lang, config=f"--psm {psm}", output_type=self.pt.Output.DICT)
        out = []
        for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
            if str(t).strip() and float(c) >= 0:
                out.append(Line(str(t).strip(), float(c) / 100, (x, y, x + w, y + h)))
        return out

    def recognize_lines(self, img):
        return self._words(img, psm=6)  # word-level: lets table reader split columns

    def recognize_line(self, img):
        words = self._words(pad(img), psm=7)
        return " ".join(w.text for w in words), mean_conf(words)

    def info(self):
        return f"tesseract {self.version} ({self.lang})"


# ---------------------------------------------------------------- EasyOCR
class EasyOCREngine(Engine):
    name = "easyocr"

    def __init__(self, langs=config.EASYOCR_LANGS):
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
        self.version = easyocr.__version__

    def recognize_lines(self, img):
        out = []
        for pts, text, conf in self.reader.readtext(cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB)):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(Line(text, float(conf), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
        return out

    def info(self):
        return f"easyocr {self.version}"


# ---------------------------------------------------------------- PaddleOCR 3.x
# PaddlePaddle 3.3 oneDNN bug on CPU: "ConvertPirAttribute2RuntimeAttribute not support" -> turn oneDNN off
PADDLE_CPU = {"device": "cpu", "enable_mkldnn": False}


def _get(res, key, default=None):
    try:
        return res[key]
    except Exception:
        try:
            return res.json["res"][key]
        except Exception:
            return default


class PaddleEngine(Engine):
    name = "paddle"

    def __init__(self, rec_model_dir: str | Path | None = None, rec_model_name: str = config.PADDLE_REC_MODEL,
                 det_model_name: str = config.PADDLE_DET_MODEL, name: str | None = None):
        from paddleocr import PaddleOCR
        kw = dict(**PADDLE_CPU, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                  text_detection_model_name=det_model_name, text_recognition_model_name=rec_model_name)
        if rec_model_dir:
            if not Path(rec_model_dir).exists():
                raise RuntimeError(f"ไม่พบโมเดล fine-tune ที่ {rec_model_dir} – รัน finetune_rec ก่อน")
            kw["text_recognition_model_dir"] = str(rec_model_dir)
        self.ocr = PaddleOCR(**kw)
        self.rec_model_name, self.rec_model_dir = rec_model_name, rec_model_dir
        self.name = name or ("paddle_ft" if rec_model_dir else "paddle")
        self._rec = None

    def recognize_lines(self, img):
        out = []
        for res in self.ocr.predict(to_bgr(img)):
            texts = _get(res, "rec_texts", []) or []
            scores = _get(res, "rec_scores", []) or []
            polys = _get(res, "rec_polys", []) or []
            for t, s, p in zip(texts, scores, polys):
                p = np.asarray(p)
                if str(t).strip():
                    out.append(Line(str(t), float(s), (int(p[:, 0].min()), int(p[:, 1].min()),
                                                       int(p[:, 0].max()), int(p[:, 1].max()))))
        return out

    def recognize_line(self, img):
        """Recognition-only on a line crop = fair test of the (fine-tuned) rec model."""
        if self._rec is None:
            try:
                from paddleocr import TextRecognition
                kw = {"model_name": self.rec_model_name, **PADDLE_CPU}
                if self.rec_model_dir:
                    kw["model_dir"] = str(self.rec_model_dir)
                self._rec = TextRecognition(**kw)
            except Exception:
                self._rec = False
        if not self._rec:
            return super().recognize_line(img)
        res = next(iter(self._rec.predict(pad(img, 4))))
        return str(_get(res, "rec_text", "")), float(_get(res, "rec_score", 0.0))

    def info(self):
        return f"paddleocr {self.rec_model_name}" + (f" ft:{self.rec_model_dir}" if self.rec_model_dir else "")


# ---------------------------------------------------------------- registry
ENGINE_NAMES = ("tesseract", "easyocr", "paddle", "paddle_th", "paddle_ft")


def get_engine(name: str) -> Engine:
    if name == "tesseract":
        return TesseractEngine()
    if name == "easyocr":
        return EasyOCREngine()
    if name == "paddle":
        return PaddleEngine()
    if name == "paddle_th":   # PP-OCRv5_mobile_rec cannot read Thai; this one is trained on Thai (+ Latin)
        return PaddleEngine(rec_model_name="th_PP-OCRv5_mobile_rec", name="paddle_th")
    if name == "paddle_ft":
        return PaddleEngine(rec_model_dir=config.PADDLE_FT_DIR)
    raise ValueError(f"unknown engine {name!r}; choose from {ENGINE_NAMES}")
