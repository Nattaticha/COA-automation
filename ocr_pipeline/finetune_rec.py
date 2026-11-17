"""Fine-tune the PaddleOCR recognition model (PP-OCRv5 mobile rec) on COA line crops – CPU, offline.

One-time setup (in the project venv):
    git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git third_party/PaddleOCR
      (no git? download the repo ZIP from GitHub and unzip to third_party/PaddleOCR)
    pip install -r third_party/PaddleOCR/requirements.txt

Run:
    python -m ocr_pipeline.finetune_rec --epochs 10 --max-train 3000          # prints commands only
    python -m ocr_pipeline.finetune_rec --epochs 10 --max-train 3000 --run    # train + export

Then compare:  python -m ocr_pipeline.compare_models --engines paddle,paddle_ft
The test split is NEVER used for training (train -> train.txt, model selection -> val.txt).
"""
import argparse
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import config

SPECIAL = set("≤≥±µμ°%<>/")
PRETRAINED_BASE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"


def subset_labels(src: Path, dst: Path, max_n: int) -> int:
    """Keep every line with special chars (≤ ≥ ± µ ° %), fill the rest randomly."""
    rows = [r for r in src.read_text("utf-8").splitlines() if "\t" in r]
    if max_n and len(rows) > max_n:
        special = [r for r in rows if SPECIAL & set(r.split("\t", 1)[1])]
        other = [r for r in rows if r not in set(special)]
        random.Random(config.SEED).shuffle(other)
        rows = special[:max_n] + other[: max(0, max_n - len(special))]
    dst.write_text("\n".join(rows) + "\n", "utf-8")
    return len(rows)


def find_config(repo: Path, base: str) -> Path | None:
    root = repo / "configs" / "rec"
    hits = sorted(root.rglob(f"{base}.yml")) + sorted(root.rglob(f"{base}.yaml"))
    return hits[0] if hits else None


def dict_of(cfg: Path) -> str | None:
    import re as _re
    m = _re.search(r"character_dict_path:\s*(\S+)", cfg.read_text("utf-8"))
    return m.group(1) if m else None


def check_dict(repo: Path, dict_rel: str | None, dataset: Path) -> None:
    d = repo / dict_rel if dict_rel else None
    cs = dataset / "charset.txt"
    if d is None:
        return
    if d.exists() and cs.exists():
        known = set(d.read_text("utf-8").splitlines())
        missing = [c for c in cs.read_text("utf-8").splitlines() if c and c not in known and not c.isspace()]
        if missing:
            print(f"[warn] ตัวอักษรที่ dictionary ของโมเดลไม่มี (โมเดลจะอ่านไม่ได้): {''.join(missing)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=config.ROOT / "third_party" / "PaddleOCR")
    ap.add_argument("--base", default=config.PADDLE_FT_BASE, help="pretrained rec model to start from")
    ap.add_argument("--config", default=None, help="repo-relative yml; default: configs/rec/**/<base>.yml")
    ap.add_argument("--dataset", type=Path, default=config.DATASET, help="e.g. data/synth_en")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--max-train", type=int, default=3000, help="cap train lines to fit CPU time (0 = all)")
    ap.add_argument("--max-val", type=int, default=400, help="cap val lines: eval on CPU is slow (0 = all)")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    a.dataset = a.dataset.resolve()          # training runs with cwd=repo -> paths must be absolute
    rec = a.dataset / "rec"
    for f in ("train.txt", "val.txt"):
        if not (rec / f).exists():
            raise SystemExit(f"ไม่พบ {rec / f} – รัน build_dataset ก่อน")
    cfg = (a.repo / a.config) if a.config else find_config(a.repo, a.base)
    if not cfg or not cfg.exists():
        raise SystemExit(f"ไม่พบ config ของ {a.base} ใน {a.repo} – clone PaddleOCR ก่อน (ดู docstring) หรือระบุ --config")
    dict_rel = dict_of(cfg)
    print(f"config: {cfg}\ndict  : {dict_rel}")
    check_dict(a.repo, dict_rel, a.dataset)

    n = subset_labels(rec / "train.txt", rec / "train_ft.txt", a.max_train)
    n_val = subset_labels(rec / "val.txt", rec / "val_ft.txt", a.max_val)
    iters = max(1, n // a.batch)   # evaluate (and keep best_accuracy) once per epoch
    pre_dir = config.MODELS / "pretrained"
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre = pre_dir / f"{a.base}_pretrained.pdparams"
    if not pre.exists():
        print(f"downloading pretrained weights -> {pre}")
        urllib.request.urlretrieve(f"{PRETRAINED_BASE}{a.base}_pretrained.pdparams", pre)

    save_dir = config.MODELS / "paddle_rec_ft" / "train"
    common = ["Global.use_gpu=False"]   # keep the base model's own character dict (from its yml)
    tool = str(Path(__file__).with_name("_paddle_tool.py"))
    train = [sys.executable, tool, "tools/train.py", "-c", str(cfg), "-o", *common,
             f"Global.pretrained_model={pre.with_suffix('')}",
             f"Global.save_model_dir={save_dir}",
             f"Global.epoch_num={a.epochs}", "Global.print_batch_step=10", "Global.save_epoch_step=1",
             f"Global.eval_batch_step=[0,{iters}]", "Global.cal_metric_during_train=True",
             f"Optimizer.lr.learning_rate={a.lr}", "Optimizer.lr.warmup_epoch=1",
             f"Train.dataset.data_dir={rec.as_posix()}", f"Train.dataset.label_file_list=['{(rec / 'train_ft.txt').as_posix()}']",
             f"Train.loader.batch_size_per_card={a.batch}", f"Train.sampler.first_bs={a.batch}",
             "Train.loader.num_workers=0",
             f"Eval.dataset.data_dir={rec.as_posix()}", f"Eval.dataset.label_file_list=['{(rec / 'val_ft.txt').as_posix()}']",
             f"Eval.loader.batch_size_per_card={a.batch}", "Eval.loader.num_workers=0"]
    export = [sys.executable, tool, "tools/export_model.py", "-c", str(cfg), "-o", *common,
              f"Global.pretrained_model={save_dir / 'best_accuracy'}",
              f"Global.save_inference_dir={config.PADDLE_FT_DIR}"]

    print(f"train lines: {n}  val lines: {n_val}  ~{iters} iters/epoch\n")
    print("TRAIN :", subprocess.list2cmdline(train), "\n")
    print("EXPORT:", subprocess.list2cmdline(export), "\n")
    if a.run:
        subprocess.run(train, cwd=a.repo, check=True)
        subprocess.run(export, cwd=a.repo, check=True)
        print(f"fine-tuned model -> {config.PADDLE_FT_DIR}\nnext: python -m ocr_pipeline.compare_models --dataset {a.dataset} --split test --engines paddle_th,paddle_ft")


if __name__ == "__main__":
    main()
