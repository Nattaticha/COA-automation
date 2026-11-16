"""Run zoning + OCR on one PDF and save the zones (text vs table) for inspection.

    python -m ocr_pipeline.extract_coa "path/to/COA.pdf" [--engine paddle] [--layout paddle|opencv]
           [--table-reader grid|paddle_table] [--preprocess red,deskew] [--out results/extract]

Native pages use the embedded text (no OCR). Scanned pages are rendered at 300 DPI,
cleaned, split into zones, then text zones -> OCR engine and table zones -> table reader.
Output: <name>.json (all zones with bbox) and <name>.xlsx (text sheet + one sheet per table).
"""
import argparse
import json
from pathlib import Path

import cv2
import pymupdf as fitz
import pandas as pd

from . import config
from .engines import get_engine
from .preprocess import imwrite, is_native, parse_steps, preprocess, render_page
from .zoning import get_layout, get_table_reader, zones_native, zones_scanned

COLORS = {"text": (0, 160, 0), "title": (255, 0, 0), "table": (0, 0, 255), "figure": (128, 128, 128)}


def extract_pdf(path: Path, engine_name="paddle", layout_name="paddle", table_reader_name="grid", steps=("red", "deskew"),
                debug_dir: Path | None = None):
    doc = fitz.open(path)
    zones, engine, layout, reader, kinds = [], None, None, None, []
    for pno, page in enumerate(doc):
        if is_native(page):
            pz = zones_native(page, pno)
            kinds.append("native")
            img = render_page(page) if debug_dir else None
        else:
            if engine is None:
                engine = get_engine(engine_name)
                layout = get_layout(layout_name)
                reader = get_table_reader(table_reader_name, engine)
            img = preprocess(render_page(page), steps)
            pz = zones_scanned(img, pno, layout, engine, reader)
            kinds.append("scanned")
        zones += pz
        if debug_dir is not None:   # draw zone boxes to check the layout visually
            vis = img.copy()
            for z in pz:
                cv2.rectangle(vis, z.bbox[:2], z.bbox[2:], COLORS.get(z.kind, (0, 0, 0)), 4)
                cv2.putText(vis, z.kind, (z.bbox[0], max(30, z.bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            COLORS.get(z.kind, (0, 0, 0)), 3)
            imwrite(debug_dir / f"{path.stem}_p{pno}_zones.png", vis)
    doc.close()
    return {"file": str(path), "pages": kinds, "zones": [z.to_dict() for z in zones]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--layout", default="paddle")
    ap.add_argument("--table-reader", default="grid")
    ap.add_argument("--preprocess", default="red,deskew")
    ap.add_argument("--out", type=Path, default=config.RESULTS / "extract")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    res = extract_pdf(a.pdf, a.engine, a.layout, a.table_reader, parse_steps(a.preprocess), debug_dir=a.out)
    stem = a.pdf.stem
    (a.out / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    with pd.ExcelWriter(a.out / f"{stem}.xlsx") as xw:
        text = [z for z in res["zones"] if z["kind"] in ("text", "title")]
        pd.DataFrame([{"page": z["page"], "kind": z["kind"], "bbox": str(z["bbox"]), "conf": z["conf"], "text": z["text"]}
                      for z in text]).to_excel(xw, sheet_name="text_zones", index=False)
        for i, z in enumerate(t for t in res["zones"] if t["kind"] == "table"):
            pd.DataFrame(z["cells"] or [[]]).to_excel(xw, sheet_name=f"table{i + 1}_p{z['page']}", index=False, header=False)
    n_t = sum(z["kind"] == "table" for z in res["zones"])
    print(f"pages: {res['pages']}  zones: {len(res['zones'])} (tables: {n_t})  -> {a.out / stem}.json/.xlsx + *_zones.png")


if __name__ == "__main__":
    main()
