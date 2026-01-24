#!/usr/bin/env python3
"""
Repair a broken `python-dotenv` installation where dist-info exists but the `dotenv/` package is missing.

This happens sometimes in constrained environments (partial installs). It breaks:
- importing `dotenv`
- `chromadb` startup (via Pydantic settings)
- `verify.py` retrieval checks

Strategy:
- If `dotenv` is importable: do nothing.
- Else try to copy `dotenv/` from the Conda base prefix into the active environment's site-packages.
  (This mirrors the approach used in `repair_dateutil.py`.)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _env_site_packages() -> Path | None:
    # Typical on Windows/Conda: <prefix>/Lib/site-packages
    prefix = Path(sys.prefix)
    p = prefix / "Lib" / "site-packages"
    return p if p.exists() else None


def _conda_base_prefix() -> Path | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        return None
    exe = Path(conda_exe)
    if not exe.exists():
        return None
    # .../anaconda3/Scripts/conda.exe -> base prefix = .../anaconda3
    try:
        return exe.resolve().parents[1]
    except Exception:
        return None


def main() -> int:
    try:
        import dotenv  # type: ignore

        print(f"[OK] dotenv importable: {dotenv.__file__}")
        return 0
    except Exception as e:
        print(f"[WARN] dotenv not importable in this environment: {e}")

    env_sp = _env_site_packages()
    if env_sp is None:
        print("[FAIL] Could not locate this environment's site-packages.")
        return 2

    base_prefix = _conda_base_prefix()
    if base_prefix is None:
        print("[FAIL] Could not locate Conda base prefix (CONDA_EXE not set).")
        return 2

    src = base_prefix / "Lib" / "site-packages" / "dotenv"
    dst = env_sp / "dotenv"

    if not src.exists():
        print(f"[FAIL] Base dotenv package not found at: {src}")
        print("       Fix by reinstalling: python -m pip install --force-reinstall python-dotenv")
        return 2

    if dst.exists():
        print(f"[INFO] Removing existing dst folder: {dst}")
        try:
            shutil.rmtree(dst)
        except Exception as e:
            print(f"[FAIL] Could not remove existing {dst}: {e}")
            return 2

    print(f"[INFO] Copying {src} -> {dst}")
    shutil.copytree(src, dst)

    try:
        import dotenv  # type: ignore

        print(f"[OK] Repaired. dotenv importable: {dotenv.__file__}")
        return 0
    except Exception as e:
        print(f"[FAIL] Copy completed but import still fails: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

