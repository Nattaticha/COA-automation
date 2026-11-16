"""CER / WER (text), Cell Accuracy / TEDS (tables), table-zone detection P/R (layout)."""
import re
import unicodedata

from rapidfuzz.distance import Levenshtein


def normalize(s: str) -> str:
    """Same normalisation for ground truth and prediction. Case-sensitive (Lot numbers)."""
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _ops(ref, hyp) -> dict:
    s = d = i = 0
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "replace":
            s += 1
        elif op.tag == "delete":
            d += 1
        else:
            i += 1
    n = len(ref)
    return {"S": s, "D": d, "I": i, "N": n, "rate": (s + d + i) / n if n else float(s + d + i > 0)}


def cer(ref: str, hyp: str) -> dict:
    """(S + D + I) / N at character level."""
    return _ops(normalize(ref), normalize(hyp))


def wer(ref: str, hyp: str) -> dict:
    """(S + D + I) / N at word level (whitespace tokens).
    Note: Thai has no spaces between words, so WER is only meaningful for English/number text."""
    return _ops(normalize(ref).split(), normalize(hyp).split())


def cer_nospace(ref: str, hyp: str) -> dict:
    """CER after removing all whitespace: spacing errors ('≤ 5.0' vs '≤5.0') do not change a COA value."""
    return _ops(re.sub(r"\s+", "", normalize(ref)), re.sub(r"\s+", "", normalize(hyp)))


def digits_ok(ref: str, hyp: str) -> bool | None:
    """All digits identical and in order (catches 250 -> 25o, ISO -> IS0). None if ref has no digit."""
    d = lambda s: re.sub(r"[^0-9]", "", normalize(s))
    return None if not d(ref) else d(ref) == d(hyp)


def corpus_rate(rows, prefix: str) -> float:
    """Micro-average over many samples: sum(edits) / sum(N)."""
    e = sum(r[f"{prefix}_S"] + r[f"{prefix}_D"] + r[f"{prefix}_I"] for r in rows)
    n = sum(r[f"{prefix}_N"] for r in rows)
    return e / n if n else float("nan")


# ---------------------------------------------------------------- tables
def _cell(s):
    return normalize(s).lower()


def cell_accuracy(gt: list[list[str]], pred: list[list[str]]) -> float:
    """% of ground-truth cells whose text matches the prediction at the same (row, col)."""
    total = sum(len(r) for r in gt)
    if not total:
        return float("nan")
    ok = 0
    for i, row in enumerate(gt):
        for j, c in enumerate(row):
            p = pred[i][j] if i < len(pred) and j < len(pred[i]) else ""
            ok += _cell(c) == _cell(p)
    return ok / total


def structure_match(gt, pred) -> bool:
    return len(gt) == len(pred) and all(len(a) == len(b) for a, b in zip(gt, pred))


try:
    from apted import APTED, Config

    class _Node:
        def __init__(self, tag, text="", children=None):
            self.tag, self.text, self.children = tag, text, children or []

    class _TedsConfig(Config):
        def rename(self, a, b):
            if a.tag != b.tag:
                return 1.0
            return Levenshtein.normalized_distance(a.text, b.text) if a.tag == "td" else 0.0

        def children(self, n):
            return n.children

    def _tree(grid):
        return _Node("table", children=[_Node("tr", children=[_Node("td", _cell(c)) for c in r]) for r in grid])

    def _count(n):
        return 1 + sum(_count(c) for c in n.children)

    def teds(gt, pred) -> float:
        """Tree-Edit-Distance-based Similarity (1 = identical). Grid version (no row/col spans)."""
        a, b = _tree(gt), _tree(pred)
        dist = APTED(a, b, _TedsConfig()).compute_edit_distance()
        return 1 - dist / max(_count(a), _count(b))
except ImportError:  # pragma: no cover
    def teds(gt, pred) -> float:
        return float("nan")


# ---------------------------------------------------------------- layout
def detection_pr(gt_boxes, pred_boxes, iou_thr=0.5):
    from .zoning import iou
    used, tp = set(), 0
    for g in gt_boxes:
        best, bj = 0.0, None
        for j, p in enumerate(pred_boxes):
            if j not in used and (v := iou(g, p)) > best:
                best, bj = v, j
        if best >= iou_thr:
            used.add(bj)
            tp += 1
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp   # TP, FP, FN
