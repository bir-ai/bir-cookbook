#!/usr/bin/env python3
"""Discover and smoke-run every cookbook recipe.

A recipe is any directory under ``recipes/`` that contains a ``main.py`` and
whose name does not start with ``_`` (so ``recipes/_template`` is ignored). This
module has no third-party dependencies so it can run as a bare CI step.

Usage
-----
    python scripts/smoke.py            # run every recipe's `main.py --smoke`
    python scripts/smoke.py --list     # print recipe names as a JSON array
    python scripts/smoke.py <name>...  # run only the named recipe(s)

``--list`` feeds the GitHub Actions matrix (see .github/workflows/ci.yml); the
default run mode is a local convenience that assumes each recipe's dependencies
are importable in the current environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "recipes"


def discover_recipes() -> list[str]:
    """Return sorted recipe directory names that ship a runnable ``main.py``."""

    if not RECIPES_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in RECIPES_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith("_") and (entry / "main.py").is_file()
    )


def run_recipe(name: str) -> bool:
    """Run one recipe's offline smoke path; return True on success."""

    recipe_dir = RECIPES_DIR / name
    main_py = recipe_dir / "main.py"
    if not main_py.is_file():
        print(f"[smoke] SKIP {name}: no main.py", file=sys.stderr)
        return False

    print(f"[smoke] running {name} …")
    completed = subprocess.run(
        [sys.executable, "main.py", "--smoke"],
        cwd=recipe_dir,
    )
    ok = completed.returncode == 0
    print(f"[smoke] {'PASS' if ok else 'FAIL'} {name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and smoke-run cookbook recipes.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered recipe names as a JSON array (for the CI matrix) and exit.",
    )
    parser.add_argument(
        "recipes",
        nargs="*",
        help="Specific recipe name(s) to run. Defaults to every discovered recipe.",
    )
    args = parser.parse_args()

    discovered = discover_recipes()

    if args.list:
        print(json.dumps(discovered))
        return 0

    selected = args.recipes or discovered
    if not selected:
        print("[smoke] no recipes found under recipes/", file=sys.stderr)
        return 1

    unknown = [name for name in selected if name not in discovered]
    if unknown:
        print(f"[smoke] unknown recipe(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    results = {name: run_recipe(name) for name in selected}
    failed = [name for name, ok in results.items() if not ok]
    print(f"\n[smoke] {len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print(f"[smoke] failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
