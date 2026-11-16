"""Synthetic ENGLISH COA data with exact ground truth (no real COA needed).

    python -m ocr_pipeline.synth_coa [--lines 3000] [--tables 60] [--out data/synth_en]

Renders COA-style text in several Windows fonts:
  header lines  : Product name, Lot/Batch No., Mfg/Exp dates (many formats), Manufacturer, Quantity
  table cells   : Parameter / Specification (≤ ≥ ± < > ranges, units mg/kg, cfu/g, %) / Result / Method
Each crop is saved clean (native_clean) and as a simulated scan (scan_sim).
Output has the SAME layout as build_dataset (rec/meta.csv, rec/<split>.txt, tables/<split>/),
so compare_models and finetune_rec work on it directly:
    python -m ocr_pipeline.compare_models --dataset data/synth_en --split test --engines paddle,paddle_th
Split is by FONT (test uses fonts never seen in train) - analogous to "unseen supplier layout".
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .preprocess import imwrite, simulate_scan

FONTS = {  # font file -> split
    "arial.ttf": "train", "calibri.ttf": "train", "times.ttf": "train", "verdana.ttf": "train",
    "segoeui.ttf": "val", "cour.ttf": "val",
    "tahoma.ttf": "test", "georgia.ttf": "test", "cambria.ttc": "test",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PRODUCTS = ["Sucrose Refined Sugar", "Citric Acid Anhydrous", "Sodium Benzoate", "Ascorbic Acid", "Whey Protein Concentrate 80",
            "Maltodextrin DE 10-12", "Skim Milk Powder", "Potassium Sorbate", "Xanthan Gum", "Cocoa Powder 10-12%",
            "Vanilla Flavour Natural", "Soy Lecithin", "Glucose Syrup 42 DE", "Tricalcium Phosphate", "Vitamin Premix VP-218"]
MAKERS = ["Northwind Ingredients Co., Ltd.", "Blue Harbor Sugar Corp.", "Alpine Acids AG", "Contoso Nutrition B.V.",
          "Fabrikam Foods (Thailand) Co., Ltd.", "Southern Dairy Co-operative", "Litware Starch Co., Ltd.", "Adatum Chemicals PLC"]
# fictitious names on purpose: real supplier names must not appear in code that leaves the company
PARAMS = [  # name, unit, spec template, result generator, method
    ("Moisture", "%", "≤ {a}", lambda r: f"{r.uniform(0.02, 4.9):.2f}", "Karl Fischer"),
    ("Loss on drying", "%", "≤ {a}", lambda r: f"{r.uniform(0.1, 0.5):.2f}", "USP <731>"),
    ("Assay (on dry basis)", "%", "{lo} - {hi}", lambda r: f"{r.uniform(99.0, 100.9):.1f}", "HPLC"),
    ("pH (10% solution)", "", "{lo} - {hi}", lambda r: f"{r.uniform(5.0, 7.5):.2f}", "pH meter"),
    ("Total Plate Count", "cfu/g", "< {n}", lambda r: f"{r.choice([10, 20, 50, 100, 150, 300])}", "ISO 4833-1"),
    ("Yeast & Mold", "cfu/g", "< {n2}", lambda r: f"< {r.choice([10, 20])}", "ISO 21527"),
    ("Coliforms", "MPN/g", "< 3", lambda r: "< 3", "BAM Ch.4"),
    ("E. coli", "/10 g", "Negative", lambda r: "Negative", "ISO 16649-2"),
    ("Salmonella spp.", "/25 g", "Absent", lambda r: "Absent", "ISO 6579-1"),
    ("Lead (Pb)", "mg/kg", "≤ {a}", lambda r: f"{r.uniform(0.01, 0.4):.3f}", "ICP-MS"),
    ("Arsenic (As)", "mg/kg", "≤ {a}", lambda r: f"< {r.choice(['0.01', '0.05'])}", "ICP-MS"),
    ("Sulphated ash", "%", "≤ {a}", lambda r: f"{r.uniform(0.01, 0.09):.3f}", "Ph. Eur. 2.4.14"),
    ("Colour (ICUMSA)", "IU", "≤ {n}", lambda r: f"{r.randint(15, 45)}", "ICUMSA GS2/3-9"),
    ("Particle size (< 250 µm)", "%", "≥ {p}", lambda r: f"{r.uniform(90, 99.9):.1f}", "Sieve"),
    ("Optical rotation", "°", "+{lo} to +{hi}", lambda r: f"+{r.uniform(20.5, 21.5):.2f}", "Polarimeter"),
    ("Temperature at loading", "°C", "{lo} ± 2", lambda r: f"{r.uniform(20, 26):.1f}", "Thermometer"),
    ("Appearance", "", "White crystalline powder", lambda r: "Complies", "Visual"),
    ("Odour", "", "Characteristic", lambda r: r.choice(["Conforms", "Complies"]), "Organoleptic"),
    ("Protein (N x 6.38)", "%", "≥ {p}", lambda r: f"{r.uniform(80, 83):.2f}", "Kjeldahl"),
]


def rand_date(r: random.Random) -> str:
    d, m, y = r.randint(1, 28), r.randint(1, 12), r.choice([2025, 2026, 2027, 2028])
    return r.choice([f"{d:02d}.{m:02d}.{y}", f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{y}/{m:02d}/{d:02d}",
                     f"{MONTHS[m - 1]} {d}, {y}", f"{d:02d}-{MONTHS[m - 1]}-{y}", f"{d} {MONTHS[m - 1].upper()} {y}"])


def rand_lot(r: random.Random) -> str:
    al = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return r.choice([f"{r.randint(10, 99)}{r.randint(1, 12):02d}{r.randint(100, 999)}",
                     f"{r.choice(al)}{r.choice(al)}{r.randint(100000, 999999)}",
                     f"L{r.randint(2024, 2027)}-{r.randint(1, 999):03d}",
                     f"{r.randint(1000, 9999)}-{r.randint(10, 99)}{r.choice(al)}",
                     f"{r.choice(al)}{r.randint(10, 99)}/{r.randint(1000, 9999)}"])


def spec_text(tmpl: str, r: random.Random) -> str:
    lo = round(r.uniform(4.0, 99.0), 1)
    return tmpl.format(a=r.choice(["0.5", "1.0", "0.05", "5.0", "0.1", "4.0"]), lo=lo, hi=round(lo + r.uniform(0.5, 2.0), 1),
                        n=r.choice([100, 1000, 5000, 10000]), n2=r.choice([10, 50, 100]), p=r.choice([80.0, 90, 95, 99]))


def header_line(r: random.Random) -> str:
    return r.choice([
        f"Product Name: {r.choice(PRODUCTS)}", f"Batch No.: {rand_lot(r)}", f"Lot No. {rand_lot(r)}",
        f"Manufacturing Date: {rand_date(r)}", f"Mfg. Date: {rand_date(r)}", f"Expiry Date: {rand_date(r)}",
        f"Best Before: {rand_date(r)}", f"Manufacturer: {r.choice(MAKERS)}", f"Quantity: {r.randint(1, 40) * 25} kg",
        f"Material Code: {r.randint(40000000, 40099999)}", f"PO No. {r.randint(4500000000, 4509999999)}",
        f"Shelf life: {r.choice([12, 18, 24, 36])} months", "CERTIFICATE OF ANALYSIS", f"Date of Analysis: {rand_date(r)}",
    ])


def cell_text(r: random.Random) -> str:
    name, unit, tmpl, res, method = r.choice(PARAMS)
    return r.choice([name, f"{spec_text(tmpl, r)} {unit}".strip(), res(r), method, unit or "-",
                     f"{name} ({unit})" if unit else name])


def render(text: str, font: ImageFont.FreeTypeFont, r: random.Random) -> np.ndarray:
    l, t, rgt, b = font.getbbox(text)
    padx, pady = r.randint(4, 14), r.randint(3, 10)
    img = Image.new("RGB", (rgt - l + 2 * padx, b - t + 2 * pady), "white")
    ImageDraw.Draw(img).text((padx - l, pady - t), text, font=font, fill=(r.randint(0, 40),) * 3)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render_table(rows, font, r: random.Random) -> np.ndarray:
    widths = [max(font.getlength(row[j]) for row in rows) + 30 for j in range(len(rows[0]))]
    h = int(font.size * 1.9)
    W, H = int(sum(widths)) + 2, h * len(rows) + 2
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    bordered = r.random() < 0.7
    x = 0
    for j, w in enumerate(widths):
        for i, row in enumerate(rows):
            d.text((x + 15, i * h + (h - font.size) // 2), row[j], font=font, fill="black")
        if bordered:
            d.line([(x, 0), (x, H)], fill="black", width=2)
        x += w
    for i in range(len(rows) + 1):
        if bordered or i in (0, 1, len(rows)):
            d.line([(0, i * h), (W, i * h)], fill="black", width=2)
    if bordered:
        d.line([(W - 1, 0), (W - 1, H)], fill="black", width=2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=3000)
    ap.add_argument("--tables", type=int, default=60)
    ap.add_argument("--out", type=Path, default=config.DATA / "synth_en")
    a = ap.parse_args()

    r, rng = random.Random(config.SEED), np.random.default_rng(config.SEED)
    fonts = [(f, sp) for f, sp in FONTS.items() if (Path(r"C:\Windows\Fonts") / f).exists()]
    rec, meta, labels = a.out / "rec", [], {}
    for k in range(a.lines):
        fname, sp = fonts[k % len(fonts)]
        font = ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / fname), r.choice([30, 34, 38, 42, 48]))  # ~9-14pt @300dpi
        group = "text" if r.random() < 0.45 else "table"
        text = header_line(r) if group == "text" else cell_text(r)
        clean = render(text, font, r)
        for vname, img in (("native_clean", clean), ("scan_sim", simulate_scan(clean, rng))):
            sid = f"syn{k:05d}_{vname}"
            rel = f"{sp}/images/{sid}.png"
            imwrite(rec / rel, img)
            labels.setdefault(sp, []).append(f"{rel}\t{text}")
            meta.append({"id": sid, "image": rel, "split": sp, "supplier": fname, "pdf": "synthetic", "page": 0,
                         "group": group, "variant": vname, "text": text})
    for sp, rows in labels.items():
        (rec / f"{sp}.txt").write_text("\n".join(rows) + "\n", "utf-8")
    with open(rec / "meta.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)

    for t in range(a.tables):
        fname, sp = fonts[t % len(fonts)]
        font = ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / fname), r.choice([30, 34, 38]))
        rows = [["Parameter", "Specification", "Result", "Method"]]
        for name, unit, tmpl, res, method in r.sample(PARAMS, r.randint(5, 12)):
            rows.append([f"{name} ({unit})" if unit else name, spec_text(tmpl, r), res(r), method])
        clean = render_table(rows, font, r)
        for vname, img in (("native_clean", clean), ("scan_sim", simulate_scan(clean, rng))):
            tid = f"syntab{t:03d}_{vname}"
            imwrite(a.out / "tables" / sp / f"{tid}.png", img)
            (a.out / "tables" / sp / f"{tid}.json").write_text(json.dumps(
                {"cells": rows, "supplier": fname, "pdf": "synthetic", "page": 0, "variant": vname}, ensure_ascii=False), "utf-8")
    (a.out / "charset.txt").write_text("\n".join(sorted({c for m in meta for c in m["text"]})), "utf-8")
    print({sp: len(v) for sp, v in labels.items()}, f"tables={a.tables * 2}", "->", a.out)


if __name__ == "__main__":
    main()
