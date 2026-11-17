"""Record a finished fine-tune run so it can be read later.

    python -m ocr_pipeline.log_finetune [--save-dir models/paddle_rec_ft/train] [--note "..."]

Parses PaddleOCR's <save_dir>/train.log and writes
  results/finetune_runs.csv            one row per fine-tune run (append-only)
  results/finetune_<run>.json          full record: settings, data, per-epoch val scores, timing, files
  results/model_metrics.csv            task=finetune rows (val acc / norm_edit_dis per epoch + best)
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from . import config
from .compare_models import append_metrics, metric_rows

TS = re.compile(r"^\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]")
KV = re.compile(r"(\w+): ([-\d.e]+)")


def parse_train_log(path: Path) -> dict:
    lines = path.read_text("utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.rstrip().endswith("INFO: Architecture :")]
    lines = lines[starts[-1]:] if starts else lines        # train.log appends runs -> keep the LAST run only
    times, args, steps, evals, best = [], {}, [], [], {}
    epoch, section = None, ""
    for ln in lines:
        if m := re.search(r"INFO: (\w+) :\s*$", ln):         # top-level config section (Global/Train/Eval/...)
            section = m.group(1)
        if m := TS.match(ln):
            times.append(datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S"))
        if "epoch: [" in ln and "global_step" in ln:
            epoch = int(re.search(r"epoch: \[(\d+)/", ln).group(1))
            kv = dict(KV.findall(ln.split("epoch:")[1]))
            steps.append({"epoch": epoch, **{k: float(v) for k, v in kv.items() if k in
                                             ("global_step", "lr", "acc", "norm_edit_dis", "loss", "avg_batch_cost")}})
        elif "cur metric" in ln:
            kv = {k: float(v) for k, v in KV.findall(ln.split("cur metric")[1])}
            step = int(steps[-1]["global_step"]) if steps else 0   # eval runs every N steps, not per epoch
            evals.append({"epoch": epoch, "step": step, **{k: kv[k] for k in ("acc", "norm_edit_dis") if k in kv}})
        elif "best metric" in ln:
            best = {k: float(v) for k, v in KV.findall(ln.split("best metric")[1])}
        for key in ("epoch_num", "pretrained_model", "learning_rate", "batch_size_per_card", "label_file_list",
                    "character_dict_path", "max_text_length", "eval_batch_step"):
            k = f"{section}.{key}" if key in ("label_file_list", "batch_size_per_card") else key
            if re.search(rf"\b{key} :", ln) and k not in args:
                args[k] = ln.split(f"{key} :", 1)[1].strip()
    return {"start": times[0].isoformat() if times else None, "end": times[-1].isoformat() if times else None,
            "minutes": round((times[-1] - times[0]).total_seconds() / 60, 1) if len(times) > 1 else None,
            "args": args, "evals": evals, "best": best, "last_step": steps[-1] if steps else {},
            "sec_per_batch": round(sum(s.get("avg_batch_cost", 0) for s in steps) / max(1, len(steps)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", type=Path, default=config.MODELS / "paddle_rec_ft" / "train")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    log = a.save_dir / "train.log"
    if not log.exists():
        raise SystemExit(f"ไม่พบ {log} – ยังไม่ได้ train หรือ save_dir ผิด")
    rec = parse_train_log(log)
    run = "ft_" + (rec["end"] or datetime.now().isoformat()).replace("-", "").replace(":", "").replace("T", "_")[:15]
    exported = config.PADDLE_FT_DIR
    record = {"run": run, "base_model": config.PADDLE_FT_BASE, "note": a.note, **rec,
              "train_log": str(log), "exported_model": str(exported) if exported.exists() else None,
              "console_log": str(config.RESULTS / "finetune_log.txt")}
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / f"finetune_{run}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")

    row = {"run": run, "base_model": config.PADDLE_FT_BASE, "start": rec["start"], "minutes": rec["minutes"],
           "epochs": rec["args"].get("epoch_num"), "lr": rec["args"].get("learning_rate"),
           "batch": rec["args"].get("Train.batch_size_per_card"), "train_labels": rec["args"].get("Train.label_file_list"),
           "val_labels": rec["args"].get("Eval.label_file_list"),
           "sec_per_batch": rec["sec_per_batch"], "best_val_acc": rec["best"].get("acc"),
           "best_val_norm_edit_dis": rec["best"].get("norm_edit_dis"), "best_epoch": rec["best"].get("best_epoch"),
           "val_acc_per_eval": " | ".join(f"e{e['epoch']}/s{e['step']}:{e.get('acc', float('nan')):.4f}"
                                          for e in rec["evals"]),
           "exported_model": record["exported_model"], "note": a.note}
    out = config.RESULTS / "finetune_runs.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    rows = []
    for e in rec["evals"]:
        rows += metric_rows("finetune", "paddle_ft", f"val/epoch={e['epoch']}/step={e['step']}", 0,
                            val_acc=e.get("acc", float("nan")), val_norm_edit_dis=e.get("norm_edit_dis", float("nan")))
    if rec["best"]:
        rows += metric_rows("finetune", "paddle_ft", "val/best", 0, val_acc=rec["best"].get("acc", float("nan")),
                            val_norm_edit_dis=rec["best"].get("norm_edit_dis", float("nan")),
                            train_minutes=rec["minutes"] or float("nan"))
    append_metrics(rows, {"run": run, "dataset": "synth_en", "split": "val", "testset_hash": "", "preprocess": "none",
                          "note": a.note, "models": {"paddle_ft": f"{config.PADDLE_FT_BASE} fine-tuned"}})
    print(json.dumps(row, ensure_ascii=False, indent=1))
    print(f"\n-> {out}\n-> {config.RESULTS / f'finetune_{run}.json'}\n-> {config.RESULTS / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
