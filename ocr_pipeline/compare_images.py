"""Compare OCR models on a folder of page images / PDFs (e.g. ocr_pipeline/data/test).

    python -m ocr_pipeline.compare_images --src ocr_pipeline/data/test --engines easyocr,paddle
           [--layout paddle|opencv] [--preprocess upscale,deskew]

Each page goes through the same flow as production:  layout (zoning) -> text zones -> engine,
table zones -> grid table reader.  Per engine you get:
  <out>/<engine>/<page>_zones.png   zone boxes drawn on the page (green=text, blue=title, red=table)
  <out>/<engine>/<page>.xlsx        text zones + one sheet per table
  <out>/summary.xlsx                time, zones, tables, mean confidence per page x engine
                                    + side_by_side sheet: the same zone read by every engine
Ground truth is optional: put <page>.gt.txt (plain text of the page) next to the image and
page-level CER/WER is added. (Page CER also counts reading-order differences.)
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import pymupdf as fitz

from . import config
from .engines import get_engine
from .metrics import cer, wer
from .preprocess import imread, imwrite, parse_steps, preprocess, render_page
from .zoning import get_layout, get_table_reader, zones_scanned

COLORS = {"text": (0, 160, 0), "title": (255, 0, 0), "table": (0, 0, 255), "figure": (128, 128, 128)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_pages(src: Path):
    for f in sorted(src.iterdir()):
        if f.suffix.lower() in IMG_EXT:
            yield f.stem, imread(f)
        elif f.suffix.lower() == ".pdf":
            with fitz.open(f) as doc:
                for i, p in enumerate(doc):
                    yield f"{f.stem}_p{i}", render_page(p)


def draw(img, zones):
    vis = img.copy()
    for z in zones:
        c = COLORS.get(z.kind, (0, 0, 0))
        cv2.rectangle(vis, z.bbox[:2], z.bbox[2:], c, 4)
        cv2.putText(vis, z.kind, (z.bbox[0], max(30, z.bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
    return vis


def page_text(zones):
    parts = []
    for z in zones:
        if z.kind == "table" and z.cells:
            parts += [" ".join(r) for r in z.cells]
        elif z.text:
            parts.append(z.text)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=config.ROOT / "ocr_pipeline" / "data" / "test")
    ap.add_argument("--engines", default="easyocr,paddle")
    ap.add_argument("--layout", default="paddle")
    ap.add_argument("--preprocess", default="upscale,deskew")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    steps = parse_steps(a.preprocess)
    out = a.out or config.RESULTS / f"images_{datetime.now():%Y%m%d_%H%M%S}"
    pages = [(n, preprocess(img, steps)) for n, img in load_pages(a.src)]
    print(f"{len(pages)} pages from {a.src}  preprocess={steps or 'none'}")

    layout = get_layout(a.layout)
    t0 = time.perf_counter()
    layouts = {n: layout.detect(img) for n, img in pages}           # same zones for every engine = fair
    print(f"layout {layout.name}: {time.perf_counter() - t0:.1f}s")

    class _Fixed:                                                      # replay cached layout per page
        name = layout.name
        def __init__(self, dets): self.dets = dets
        def detect(self, img): return self.dets

    summary, side = [], {}
    for ename in a.engines.split(","):
        try:
            eng = get_engine(ename.strip())
        except Exception as e:
            print(f"[skip] {ename}: {e}")
            continue
        reader = get_table_reader("grid", eng)
        print(f"\n== {eng.info()}")
        for name, img in pages:
            t = time.perf_counter()
            zones = zones_scanned(img, 0, _Fixed(layouts[name]), eng, reader)
            dt = time.perf_counter() - t
            d = out / eng.name
            imwrite(d / f"{name}_zones.png", draw(img, zones))
            with pd.ExcelWriter(d / f"{name}.xlsx") as xw:
                pd.DataFrame([{"kind": z.kind, "bbox": str(z.bbox), "conf": z.conf, "text": z.text}
                              for z in zones if z.kind != "table"]).to_excel(xw, sheet_name="text_zones", index=False)
                for i, z in enumerate(z for z in zones if z.kind == "table"):
                    pd.DataFrame(z.cells or [[]]).to_excel(xw, sheet_name=f"table{i + 1}", index=False, header=False)
            confs = [z.conf for z in zones if z.conf is not None]
            row = {"page": name, "engine": eng.name, "seconds": round(dt, 1), "zones": len(zones),
                   "tables": sum(z.kind == "table" for z in zones),
                   "table_cells": sum(len(c) for z in zones if z.kind == "table" and z.cells for c in z.cells),
                   "mean_conf": round(sum(confs) / len(confs), 3) if confs else None}
            gt = a.src / f"{name}.gt.txt"
            if gt.exists():
                ref, hyp = gt.read_text("utf-8"), page_text(zones)
                row.update({"page_CER": round(cer(ref, hyp)["rate"], 4), "page_WER": round(wer(ref, hyp)["rate"], 4)})
            summary.append(row)
            for k, z in enumerate(zones):
                key = (name, k, z.kind, str(z.bbox))
                side.setdefault(key, {})[eng.name] = (z.text if z.kind != "table"
                                                      else "\n".join(" | ".join(r) for r in (z.cells or [])))
            print(f"  {name}: {dt:.1f}s zones={row['zones']} tables={row['tables']} conf={row['mean_conf']}")

    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "summary.xlsx") as xw:
        s = pd.DataFrame(summary)
        s.to_excel(xw, sheet_name="per_page", index=False)
        if not s.empty:
            s.groupby("engine").agg(pages=("page", "count"), sec_per_page=("seconds", "mean"),
                                    mean_conf=("mean_conf", "mean"), tables=("tables", "sum"),
                                    table_cells=("table_cells", "sum")).reset_index() \
             .to_excel(xw, sheet_name="per_engine", index=False)
        pd.DataFrame([{"page": k[0], "zone": k[1], "kind": k[2], "bbox": k[3], **v} for k, v in side.items()]) \
          .to_excel(xw, sheet_name="side_by_side", index=False)
    # keep every measurement in the shared history log
    import json
    from .compare_models import METRICS_LOG, append_metrics, metric_rows
    s = pd.DataFrame(summary)
    rows = []
    for eng_name, g in s.groupby("engine") if not s.empty else []:
        m = {"sec_per_page": g["seconds"].mean(), "mean_conf": g["mean_conf"].mean(), "tables": g["tables"].sum(),
             "table_cells": g["table_cells"].sum()}
        if "page_CER" in g:
            m.update(page_CER=g["page_CER"].mean(), page_WER=g["page_WER"].mean())
        rows += metric_rows("images", eng_name, f"layout={layout.name}", len(g), **m)
    cfg = {"run": out.name, "dataset": str(a.src), "split": "-", "preprocess": ",".join(steps) or "none", "note": "",
           "testset_hash": "", "models": {e: e for e in s["engine"].unique()} if not s.empty else {}}
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), "utf-8")
    append_metrics(rows, cfg)
    print(f"\nresults -> {out}\nall runs  -> {METRICS_LOG}")


if __name__ == "__main__":
    main()
