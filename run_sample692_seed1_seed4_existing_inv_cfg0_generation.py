#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import runpy
import sys

_ROOT = Path(__file__).resolve().parent
_EXPERIMENTS = _ROOT / "experiments"
_TARGET = _EXPERIMENTS / Path(__file__).name
sys.path[:0] = [str(_EXPERIMENTS), str(_ROOT)]

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    _spec = importlib.util.spec_from_file_location(f"_experiments_{Path(__file__).stem}", _TARGET)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    globals().update({k: v for k, v in _module.__dict__.items() if not k.startswith("__")})
