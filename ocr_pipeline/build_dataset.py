"""Build the OCR dataset (ground truth) from historical COA / RD Spec PDFs.

    python -m ocr_pipeline.build_dataset [--src data/raw] [--limit-pdfs N]

Native PDF  -> embedded text is the ground truth (no typing). Each text chunk is cropped
               from the rendered page, saved clean + SCAN_VARIANTS simulated scans.
               Tables from page.find_tables() become table ground truth (crop + cells).
Scanned PDF -> page images exported to data/to_label/<supplier>/ for PPOCRLabel;
               once labeled (data/labeled/**/Label.txt) they are imported as variant=real_scan.

Split is by SUPPLIER and stored in data/dataset/split.json so the TEST SET NEVER CHANGES;
new suppliers found later go to train.

Output (data/dataset/):
  rec/{train,val,test}.txt        PaddleOCR rec label files: "<split>/images/x.png\\t<text>"
  rec/<split>/images/*.png
  rec/meta.csv                    id, split, supplier, pdf, page, group(text|table), variant, text
  tables/<split>/<id>.png|.json   table crop + ground-truth cells
  pages/<split>/<id>.png|.json    simulated-scan page + ground-truth zones (for layout eval)
  charset.txt                     every character in the labels
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz
import numpy as np

from . import config
from .preprocess import imread, imwrite, is_native, render_page, simulate_scan
from .zoning import native_tables, pt2px, zones_native

SUP_RE = re.compile(r"^(\d{3,})_")


def supplier_of(pdf: Path, root: Path) -> str:
    """COA/<supplier code>_<name>/... -> supplier code; else first sub-folder; else file stem."""
    parts = pdf.relative_to(root).parts[:-1]
    for p in parts:
        if m := SUP_RE.match(p):
            return m.group(1)
    return parts[0] if parts else pdf.stem


def load_or_make_split(suppliers: dict[str, int], path: Path, rng) -> dict[str, str]:
    split = json.loads(path.read_text("utf-8")) if path.exists() else {}
    new = [s for s in suppliers if s not in split]
    if not split:  # first build: allocate by file count
        if len(new) < 3:
            print(f"[warn] มี supplier แค่ {len(new)} ราย – แบ่ง split ตามไฟล์แทน (ควรมี ≥3 ราย)")
            return {}
        order = list(rng.permutation(new))
        total, acc = sum(suppliers.values()), 0
        for s in order:
            frac = acc / total
            split[s] = "train" if frac < config.SPLIT[0] else "val" if frac < config.SPLIT[0] + config.SPLIT[1] else "test"
            acc += suppliers[s]
        for need in ("val", "test"):   # guarantee non-empty val/test
            if need not in split.values():
                split[order[-1 if need == "test" else -2]] = need
    else:
        for s in new:
            split[s] = "train"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=1), "utf-8")
    return split


def text_chunks(page: fitz.Page, max_chars: int):
    """Group words of one PDF line into chunks; break at big horizontal gaps (= different cells)."""
    lines = defaultdict(list)
    for w in page.get_text("words"):
        lines[(w[5], w[6])].append(w)
    for ws in lines.values():
        ws.sort(key=lambda w: w[0])
        chunk = [ws[0]]
        for w in ws[1:]:
            h = chunk[-1][3] - chunk[-1][1]
            text_len = len(" ".join(c[4] for c in chunk)) + 1 + len(w[4])
            if w[0] - chunk[-1][2] > 1.5 * h or text_len > max_chars:
                yield chunk
                chunk = [w]
            else:
                chunk.append(w)
        yield chunk


def clean_label(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\t", " ")).strip()


def crop(img, bbox, padpx=6):
    x0, y0, x1, y1 = bbox
    h, w = img.shape[:2]
    return img[max(0, y0 - padpx):min(h, y1 + padpx), max(0, x0 - padpx):min(w, x1 + padpx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=config.RAW)
    ap.add_argument("--out", type=Path, default=config.DATASET)
    ap.add_argument("--limit-pdfs", type=int, default=0)
    ap.add_argument("--variants", type=int, default=config.SCAN_VARIANTS)
    a = ap.parse_args()

    rng = np.random.default_rng(config.SEED)
    pdfs = sorted(a.src.rglob("*.pdf"))
    if a.limit_pdfs:
        pdfs = pdfs[:a.limit_pdfs]
    if not pdfs:
        raise SystemExit(f"ไม่พบ PDF ใน {a.src} – คัดลอก COA/RD Spec ย้อนหลังมาไว้ที่นี่ (คงโครงโฟลเดอร์ supplier ไว้)")

    sup_of = {p: supplier_of(p, a.src) for p in pdfs}
    counts = defaultdict(int)
    for s in sup_of.values():
        counts[s] += 1
    split = load_or_make_split(counts, a.out / "split.json", rng)
    split_of = lambda pdf, idx: split.get(sup_of[pdf]) or ("train", "val", "test")[0 if idx % 7 < 5 else 1 if idx % 7 == 5 else 2]

    rec = a.out / "rec"
    meta, labels, charset = [], defaultdict(list), set()
    n_tables = n_pages = 0
    to_label = []

    for idx, pdf in enumerate(pdfs):
        sp, sup = split_of(pdf, idx), sup_of[pdf]
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"[skip] {pdf.name}: {e}")
            continue
        for pno, page in enumerate(doc):
            img = render_page(page)
            if not is_native(page):
                out = config.TO_LABEL / sup / f"{idx:05d}_{pdf.stem[:40]}_p{pno}.png"
                imwrite(out, img)
                to_label.append(out)
                continue
            base = f"{idx:05d}_p{pno}"
            tables = native_tables(page)
            t_rects = [pt2px(r) for r, _ in tables]
            # ---- text-line crops
            for k, ch in enumerate(text_chunks(page, config.MAX_LINE_CHARS)):
                text = clean_label(" ".join(w[4] for w in ch))
                bbox = pt2px((min(w[0] for w in ch), min(w[1] for w in ch), max(w[2] for w in ch), max(w[3] for w in ch)))
                if not text or bbox[3] - bbox[1] < 8:
                    continue
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                group = "table" if any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in t_rects) else "text"
                c = crop(img, bbox)
                variants = [("native_clean", c)] + [(f"scan_sim{v}", simulate_scan(c, rng)) for v in range(a.variants)]
                for vname, vimg in variants:
                    sid = f"{base}_{k:04d}_{vname}"
                    rel = f"{sp}/images/{sid}.png"
                    imwrite(rec / rel, vimg)
                    labels[sp].append(f"{rel}\t{text}")
                    meta.append({"id": sid, "image": rel, "split": sp, "supplier": sup, "pdf": str(pdf.relative_to(a.src)),
                                 "page": pno, "group": group, "variant": vname.rstrip("0123456789"), "text": text})
                charset.update(text)
            # ---- table ground truth
            for t, (r, cells) in enumerate(tables):
                bb = pt2px(r)
                for vname, timg in (("native_clean", crop(img, bb, 4)), ("scan_sim", simulate_scan(crop(img, bb, 4), rng))):
                    tid = f"{base}_t{t}_{vname}"
                    imwrite(a.out / "tables" / sp / f"{tid}.png", timg)
                    (a.out / "tables" / sp / f"{tid}.json").write_text(json.dumps(
                        {"cells": cells, "supplier": sup, "pdf": str(pdf.relative_to(a.src)), "page": pno, "variant": vname},
                        ensure_ascii=False), "utf-8")
                    n_tables += 1
            # ---- page-level zones for layout evaluation (simulated scan)
            zones = zones_native(page, pno)
            imwrite(a.out / "pages" / sp / f"{base}.png", simulate_scan(img, rng))
            (a.out / "pages" / sp / f"{base}.json").write_text(json.dumps(
                {"zones": [z.to_dict() for z in zones], "supplier": sup, "pdf": str(pdf.relative_to(a.src))},
                ensure_ascii=False), "utf-8")
            n_pages += 1
        doc.close()
        print(f"[{idx + 1}/{len(pdfs)}] {sp:5s} {sup:>10s}  {pdf.name}")

    meta += import_ppocrlabel(rec, labels, charset, split)
    for sp, rows in labels.items():
        (rec / f"{sp}.txt").write_text("\n".join(rows) + "\n", "utf-8")
    if meta:
        with open(rec / "meta.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0]))
            w.writeheader()
            w.writerows(meta)
    (a.out / "charset.txt").write_text("\n".join(sorted(charset)), "utf-8")

    print(f"\nline samples: { {k: len(v) for k, v in labels.items()} }  tables: {n_tables}  pages: {n_pages}")
    if to_label:
        print(f"scanned pages to label with PPOCRLabel: {len(to_label)} -> {config.TO_LABEL}")
    special = sorted(ch for ch in charset if ch in "≤≥±µμ°%/")
    print("special chars present:", "".join(special))


def import_ppocrlabel(rec: Path, labels, charset, split) -> list[dict]:
    """Label.txt lines: '<folder>/<img>.png\\t[{"transcription": "...", "points": [[x,y]x4]}, ...]'"""
    meta = []
    for lab in config.LABELED.rglob("Label.txt"):
        for line in lab.read_text("utf-8").splitlines():
            if "\t" not in line:
                continue
            rel_img, js = line.split("\t", 1)
            img = imread(lab.parent / Path(rel_img).name)   # Label.txt sits next to its images
            sup = lab.parent.name                             # folder = supplier (as exported to to_label/)
            sp = split.get(sup, "train")
            for k, item in enumerate(json.loads(js)):
                text = clean_label(item.get("transcription", ""))
                if not text or text == "###" or item.get("difficult"):
                    continue
                pts = np.array(item["points"])
                bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))
                sid = f"lab_{Path(rel_img).stem}_{k:04d}"
                rel = f"{sp}/images/{sid}.png"
                imwrite(rec / rel, crop(img, bbox, 3))
                labels[sp].append(f"{rel}\t{text}")
                charset.update(text)
                meta.append({"id": sid, "image": rel, "split": sp, "supplier": sup, "pdf": rel_img, "page": -1,
                             "group": item.get("key_cls", "text") if item.get("key_cls") in ("text", "table") else "text",
                             "variant": "real_scan", "text": text})
    return meta


if __name__ == "__main__":
    main()
