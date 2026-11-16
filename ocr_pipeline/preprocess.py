"""Page rendering, image clean-up (red removal, deskew, denoise, binarize) and scan simulation."""
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

from .config import DPI, NATIVE_MIN_CHARS


# ---------- unicode-safe image IO (cv2.imread/imwrite fail on Thai paths) ----------
def imread(path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


# ---------- PDF ----------
def render_page(page: fitz.Page, dpi: int = DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def is_native(page: fitz.Page, min_chars: int = NATIVE_MIN_CHARS) -> bool:
    return len(page.get_text("text").strip()) >= min_chars


# ---------- clean-up steps ----------
def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def remove_red(img):
    """Whiten red pixels (RD Spec watermark / red stamps)."""
    hsv = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    out = to_bgr(img).copy()
    out[mask > 0] = 255
    return out


def estimate_skew(gray, limit=5.0, step=0.25) -> float:
    """Projection-profile skew search; robust across OpenCV versions."""
    h, w = gray.shape
    s = min(1.0, 1000 / w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    inv = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    sh, sw = inv.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-limit, limit + 1e-9, step):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), a, 1.0)
        r = cv2.warpAffine(inv, m, (sw, sh), flags=cv2.INTER_NEAREST)
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def deskew(img):
    a = estimate_skew(to_gray(img))
    return img if abs(a) < 0.1 else rotate(img, a)


def denoise(img):
    return cv2.medianBlur(img, 3)


def binarize(img):
    g = to_gray(img)
    return to_bgr(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15))


def upscale(img, target_width=2480):
    """Low-res images (e.g. ~72 DPI web images) -> about A4 @300 DPI width; small text OCRs much better."""
    w = img.shape[1]
    return img if w >= target_width * 0.8 else cv2.resize(img, None, fx=target_width / w, fy=target_width / w,
                                                         interpolation=cv2.INTER_CUBIC)


STEPS = {"upscale": upscale, "red": remove_red, "deskew": deskew, "denoise": denoise, "binarize": binarize}


def parse_steps(spec: str | None) -> tuple[str, ...]:
    if not spec or spec.lower() == "none":
        return ()
    steps = tuple(s.strip() for s in spec.split(",") if s.strip())
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise ValueError(f"unknown preprocess step(s) {bad}; choose from {list(STEPS)}")
    return steps


def preprocess(img, steps=()):
    for s in steps:
        img = STEPS[s](img)
    return to_bgr(img)


# ---------- synthetic scan (for training data from native PDFs) ----------
def simulate_scan(img, rng: np.random.Generator):
    out = rotate(img, rng.uniform(-1.2, 1.2))
    k = int(rng.choice([1, 3]))
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.8, 1.1), beta=rng.uniform(-20, 15))
    noise = rng.normal(0, rng.uniform(3, 12), out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(rng.integers(35, 80))
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def pad(img, px=10):
    return cv2.copyMakeBorder(to_bgr(img), px, px, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
