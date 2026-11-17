"""Run a PaddleOCR repo tool (tools/train.py, tools/export_model.py) with torch imported FIRST.

On Windows, paddle and torch ship conflicting DLLs; albumentations (used by PaddleOCR's
augmentation) imports torch after paddle and fails with "WinError 127". Loading torch before
paddle avoids it.   usage: python _paddle_tool.py tools/train.py -c ... -o ...
"""
import runpy
import sys

try:
    import torch  # noqa: F401  (must come before paddle)
except Exception:
    pass

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, ".")
runpy.run_path(script, run_name="__main__")
