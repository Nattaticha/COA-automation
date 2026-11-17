"""Compare OCR models on the SAME fixed test set.

    python -m ocr_pipeline.compare_models --engines tesseract,easyocr,paddle [--split test]
           [--preprocess none|red,deskew,binarize] [--limit 300] [--tasks lines,tables,layout]
           [--table-readers grid,paddle_table] [--layouts opencv,paddle]

Tasks
  lines  : CER / WER per engine, broken down by group (text zone vs table zone) and
           variant (native_clean / scan_sim / real_scan)  + latency per line
  tables : Cell Accuracy, TEDS, structure match per table reader (grid+<engine>, paddle_table)
  layout : table-zone detection Precision / Recall @IoU0.5 per layout detector + latency

Writes results/<timestamp>/ {lines.csv, tables.csv, layout.csv, summary.xlsx} and appends
one row per engine to results/runs.csv (model version + test-set hash + scores).
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .engines import get_engine
from .metrics import cell_accuracy, cell_accuracy_nospace, cer, cer_nospace, corpus_rate, digits_ok, detection_pr, structure_match, teds, wer
from .preprocess import imread, parse_steps, preprocess
from .zoning import get_layout, get_table_reader

DATASET = config.DATASET


def testset_hash(paths) -> str:
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        h.update(p.encode())
    return h.hexdigest()[:10]


def script_of(text: str) -> str:
    thai = any("฀" <= ch <= "๿" for ch in text)
    latin = any(ch.isascii() and ch.isalnum() for ch in text)
    return "mixed" if thai and latin else "thai" if thai else "latin"


def eval_lines(engine, meta: pd.DataFrame, steps) -> list[dict]:
    rows = []
    for i, r in enumerate(meta.itertuples(), 1):
        img = preprocess(imread(DATASET / "rec" / r.image), steps)
        t0 = time.perf_counter()
        try:
            hyp, conf = engine.recognize_line(img)
        except Exception as e:
            hyp, conf = "", 0.0
            print(f"  [err] {r.id}: {e}")
        dt = time.perf_counter() - t0
        c, w, ns = cer(r.text, hyp), wer(r.text, hyp), cer_nospace(r.text, hyp)
        rows.append({"engine": engine.name, "id": r.id, "group": r.group, "variant": r.variant, "script": script_of(r.text),
                     "supplier": r.supplier,
                     "gt": r.text, "pred": hyp, "conf": round(conf, 4), "latency_s": round(dt, 4),
                     "cer": c["rate"], "wer": w["rate"],
                     "digits_ok": digits_ok(r.text, hyp),
                     **{f"c_{k}": c[k] for k in "SDIN"}, **{f"w_{k}": w[k] for k in "SDIN"},
                     **{f"ns_{k}": ns[k] for k in "SDIN"}})
        if i % 100 == 0:
            print(f"  {engine.name}: {i}/{len(meta)}  running CER={corpus_rate(rows, 'c'):.4f}")
    return rows


def summarize_lines(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for by in (["engine"], ["engine", "script"], ["engine", "group"], ["engine", "variant"], ["engine", "group", "variant"]):
        for key, g in df.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            recs = g.to_dict("records")
            out.append({**dict(zip(by, key)), "n": len(g),
                        "CER": corpus_rate(recs, "c"), "WER": corpus_rate(recs, "w"),
                        "line_exact_pct": 100 * (g["cer"] == 0).mean(),
                        "CER_nospace": corpus_rate(recs, "ns"),
                        "digits_ok_pct": 100 * g["digits_ok"].dropna().astype(bool).mean() if g["digits_ok"].notna().any() else float("nan"),
                        "latency_ms": 1000 * g["latency_s"].mean()})
    s = pd.DataFrame(out)
    s[["script", "group", "variant"]] = s[["script", "group", "variant"]].fillna("ALL")
    return s


def eval_tables(reader, files, steps) -> list[dict]:
    rows = []
    for png in files:
        gt = json.loads(png.with_suffix(".json").read_text("utf-8"))
        img = preprocess(imread(png), steps)
        t0 = time.perf_counter()
        try:
            pred, conf = reader.read(img)
        except Exception as e:
            pred, conf = [], 0.0
            print(f"  [err] {png.name}: {e}")
        dt = time.perf_counter() - t0
        rows.append({"reader": reader.name, "id": png.stem, "variant": gt["variant"], "supplier": gt["supplier"],
                     "gt_rows": len(gt["cells"]), "pred_rows": len(pred),
                     "cell_acc": cell_accuracy(gt["cells"], pred),
                     "cell_acc_nospace": cell_accuracy_nospace(gt["cells"], pred), "teds": teds(gt["cells"], pred),
                     "structure_ok": structure_match(gt["cells"], pred), "latency_s": round(dt, 3),
                     "pred_json": json.dumps(pred, ensure_ascii=False)})
    return rows


def eval_layout(layout, files, steps) -> list[dict]:
    rows = []
    for png in files:
        gt = json.loads(png.with_suffix(".json").read_text("utf-8"))
        gt_t = [z["bbox"] for z in gt["zones"] if z["kind"] == "table"]
        img = preprocess(imread(png), steps)
        t0 = time.perf_counter()
        dets = layout.detect(img)
        dt = time.perf_counter() - t0
        pred_t = [b for lab, s, b in dets if lab == "table"]
        tp, fp, fn = detection_pr(gt_t, pred_t)
        rows.append({"layout": layout.name, "id": png.stem, "tp": tp, "fp": fp, "fn": fn, "latency_s": round(dt, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="easyocr,paddle")
    ap.add_argument("--split", default="test")
    ap.add_argument("--preprocess", default="none")
    ap.add_argument("--limit", type=int, default=0, help="max samples per task (0 = all)")
    ap.add_argument("--tasks", default="lines,tables,layout")
    ap.add_argument("--table-readers", default="grid")
    ap.add_argument("--layouts", default="opencv,paddle")
    ap.add_argument("--note", default="")
    ap.add_argument("--dataset", type=Path, default=config.DATASET)
    a = ap.parse_args()

    global DATASET
    DATASET = a.dataset
    steps = parse_steps(a.preprocess)
    tasks = set(a.tasks.split(","))
    out = config.RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sheets, runlog, testset_hash_val = {}, [], ""

    engines = {}
    for name in a.engines.split(","):
        try:
            engines[name] = get_engine(name.strip())
            print(f"[ok] {engines[name].info()}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    # ---------------- lines: CER / WER
    if "lines" in tasks and engines:
        meta = pd.read_csv(DATASET / "rec" / "meta.csv", encoding="utf-8-sig", dtype=str)
        meta = meta[meta["split"] == a.split]
        if a.limit and len(meta) > a.limit:
            meta = meta.sample(a.limit, random_state=config.SEED)
        th = testset_hash(meta["id"])
        print(f"\nLINES: {len(meta)} samples (split={a.split}, hash={th}, preprocess={steps or 'none'})")
        rows = []
        for e in engines.values():
            rows += eval_lines(e, meta, steps)
        pd.DataFrame(rows).to_csv(out / "lines.csv", index=False, encoding="utf-8-sig")
        summ = summarize_lines(rows)
        sheets["lines_summary"] = summ
        sheets["lines_worst"] = pd.DataFrame(rows).sort_values("cer", ascending=False).head(300)
        print(summ[summ["script"].eq("ALL") & summ["group"].eq("ALL") & summ["variant"].eq("ALL")].to_string(index=False))
        testset_hash_val = th
        for s in summ.itertuples():
            subset = "/".join(f"{k}={getattr(s, k)}" for k in ("script", "group", "variant") if getattr(s, k) != "ALL") or "ALL"
            runlog += metric_rows("lines", s.engine, subset, s.n, CER=s.CER, WER=s.WER,
                                  line_exact_pct=s.line_exact_pct, CER_nospace=s.CER_nospace,
                                  digits_ok_pct=s.digits_ok_pct, latency_ms=s.latency_ms)

    # ---------------- tables: Cell Accuracy / TEDS
    if "tables" in tasks:
        files = sorted((DATASET / "tables" / a.split).glob("*.png"))[: a.limit or None]
        rows = []
        for rname in a.table_readers.split(","):
            targets = engines.values() if rname == "grid" else [next(iter(engines.values()), None)]
            for e in targets:
                try:
                    reader = get_table_reader(rname, e)
                except Exception as ex:
                    print(f"[skip] table reader {rname}: {ex}")
                    continue
                print(f"\nTABLES: {reader.name} on {len(files)} tables")
                rows += eval_tables(reader, files, steps)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / "tables.csv", index=False, encoding="utf-8-sig")
            summ = df.groupby(["reader", "variant"]).agg(n=("id", "count"), cell_acc=("cell_acc", "mean"),
                                                          cell_acc_nospace=("cell_acc_nospace", "mean"),
                                                          TEDS=("teds", "mean"), structure_ok=("structure_ok", "mean"),
                                                          latency_s=("latency_s", "mean")).reset_index()
            sheets["tables_summary"] = summ
            print(summ.to_string(index=False))
            for s in summ.itertuples():
                runlog += metric_rows("tables", s.reader, f"variant={s.variant}", s.n, cell_acc=s.cell_acc, cell_acc_nospace=s.cell_acc_nospace, TEDS=s.TEDS,
                                      structure_ok=s.structure_ok, latency_s=s.latency_s)
            testset_hash_val = testset_hash_val or testset_hash(df["id"])

    # ---------------- layout: table-zone detection
    if "layout" in tasks:
        files = sorted((DATASET / "pages" / a.split).glob("*.png"))[: a.limit or None]
        rows = []
        for lname in a.layouts.split(","):
            try:
                lay = get_layout(lname)
            except Exception as ex:
                print(f"[skip] layout {lname}: {ex}")
                continue
            print(f"\nLAYOUT: {lay.name} on {len(files)} pages")
            rows += eval_layout(lay, files, steps)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / "layout.csv", index=False)
            g = df.groupby("layout")[["tp", "fp", "fn", "latency_s"]].agg({"tp": "sum", "fp": "sum", "fn": "sum", "latency_s": "mean"})
            g["precision"] = g.tp / (g.tp + g.fp).clip(lower=1)
            g["recall"] = g.tp / (g.tp + g.fn).clip(lower=1)
            sheets["layout_summary"] = g.reset_index()
            print(g.to_string())
            for name, s in g.iterrows():
                runlog += metric_rows("layout", name, "table_zones", s.tp + s.fn, precision=s.precision,
                                      recall=s.recall, latency_s=s.latency_s)

    if sheets:
        with pd.ExcelWriter(out / "summary.xlsx") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
    run_cfg = {"run": out.name, "dataset": str(DATASET), "split": a.split, "preprocess": ",".join(steps) or "none",
               "limit": a.limit, "tasks": sorted(tasks), "note": a.note,
               "models": {n: e.info() for n, e in engines.items()}, "testset_hash": testset_hash_val}
    (out / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=1), "utf-8")
    append_metrics(runlog, run_cfg)
    print(f"\nresults -> {out}\nall runs  -> {METRICS_LOG}")


METRICS_LOG = config.RESULTS / "model_metrics.csv"
METRIC_FIELDS = ["run", "dataset", "split", "testset_hash", "preprocess", "task", "model", "model_info",
                 "subset", "n", "metric", "value", "note"]


def append_metrics(rows: list[dict], cfg: dict) -> None:
    """Long-format history of EVERY measurement (one row per metric) – never overwritten."""
    if not rows:
        return
    new = not METRICS_LOG.exists()
    METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"run": cfg["run"], "dataset": Path(cfg["dataset"]).name, "split": cfg["split"],
                        "testset_hash": cfg["testset_hash"], "preprocess": cfg["preprocess"], "note": cfg["note"],
                        "model_info": cfg["models"].get(r["model"].split("+")[-1], r["model"]), **r})


def metric_rows(task, model, subset, n, **metrics):
    return [{"task": task, "model": model, "subset": subset, "n": int(n), "metric": k,
             "value": round(float(v), 6)} for k, v in metrics.items() if pd.notna(v)]


if __name__ == "__main__":
    main()
