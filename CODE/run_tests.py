#!/usr/bin/env python3
"""
Run the repo's unittest suite reliably from any working directory.

Why:
- `python -m unittest discover -s tests` must be run from the repo root (where `tests/` exists).
- Many users run commands from `CODE/`, which makes discovery fail.
"""

from __future__ import annotations

import os
import sys
import unittest


def main(argv: list[str] | None = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(repo_root, "tests")

    if not os.path.isdir(tests_dir):
        print(f"[FAIL] tests directory not found: {tests_dir}")
        return 2

    # Ensure `tests` is importable for unittest discovery when run from `CODE/`.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    suite = unittest.defaultTestLoader.discover(
        start_dir=tests_dir,
        pattern="test*.py",
        top_level_dir=repo_root,
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
