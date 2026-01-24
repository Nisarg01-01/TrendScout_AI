#!/usr/bin/env python3
"""
Repair a broken python-dateutil install inside the current environment.

Symptom:
- `pip show python-dateutil` shows installed, but `import dateutil` fails.
Cause:
- Only the *.dist-info metadata exists; the actual `dateutil/` package folder is missing.

Fix strategy:
1) Prefer reinstall: `python -m pip install --force-reinstall --no-deps python-dateutil`
2) If network is blocked, copy `dateutil/` from the conda base environment if available.
"""

import os
import shutil
import sys


def _print(status: str, msg: str):
    print(f"[{status}] {msg}")


def find_conda_base_prefix(env_prefix: str) -> str | None:
    parts = env_prefix.split(os.sep)
    # ...\anaconda3\envs\<name>
    if len(parts) >= 2 and "envs" in parts:
        idx = parts.index("envs")
        return os.sep.join(parts[:idx])
    return None


def main():
    try:
        import dateutil  # noqa: F401
        _print("OK", "dateutil is already importable")
        return 0
    except Exception as e:
        _print("WARN", f"dateutil not importable: {e}")

    env_prefix = sys.prefix
    base_prefix = find_conda_base_prefix(env_prefix)
    if not base_prefix:
        _print("FAIL", f"Could not infer conda base prefix from sys.prefix: {env_prefix}")
        _print("INFO", "Try: python -m pip install --force-reinstall --no-deps python-dateutil")
        return 1

    src = os.path.join(base_prefix, "Lib", "site-packages", "dateutil")
    dst = os.path.join(env_prefix, "Lib", "site-packages", "dateutil")

    if not os.path.isdir(src):
        _print("FAIL", f"Source dateutil folder not found: {src}")
        _print("INFO", "Try: python -m pip install --force-reinstall --no-deps python-dateutil")
        return 1

    if os.path.isdir(dst):
        _print("OK", f"Destination already exists: {dst}")
    else:
        _print("INFO", f"Copying {src} -> {dst}")
        shutil.copytree(src, dst)
        _print("OK", "Copied dateutil package folder")

    try:
        import dateutil  # noqa: F401
        _print("OK", "dateutil import now works")
        return 0
    except Exception as e:
        _print("FAIL", f"dateutil still not importable after copy: {e}")
        _print("INFO", "Try: python -m pip install --force-reinstall --no-deps python-dateutil")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

