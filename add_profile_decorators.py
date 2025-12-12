#!/usr/bin/env python
"""Add @profile decorators to functions in a Python file for line_profiler.

- Inserts a fallback profile() if missing so the file still runs without kernprof.
- Adds @profile above every function definition that is not already decorated.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


def _ensure_profile_stub(lines: List[str]) -> List[str]:
    """Ensure a profile() stub exists so normal execution keeps working."""
    if any(re.match(r"\s*def\s+profile\s*\(", ln) for ln in lines):
        return lines

    stub = [
        "try:",
        "    profile",
        "except NameError:",
        "    def profile(func):",
        "        return func",
        "",
    ]

    insert_idx = 0
    last_import = -1
    for idx, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = idx
            insert_idx = idx + 1
        elif stripped == "":
            continue
        else:
            break
    if last_import == -1:
        insert_idx = 0

    return lines[:insert_idx] + stub + lines[insert_idx:]


def _add_profile_decorators(lines: List[str]) -> List[str]:
    """Insert @profile ahead of every function definition unless already present."""
    out: List[str] = []
    prev_nonempty = ""
    pattern = re.compile(r"^(\s*)def\s+([A-Za-z0-9_]+)\s*\(")

    for ln in lines:
        m = pattern.match(ln)
        if m and m.group(2) != "profile":
            if prev_nonempty.lstrip().startswith("@profile") or prev_nonempty.lstrip().startswith("@profile("):
                out.append(ln)
            elif prev_nonempty.lstrip().startswith("@"):
                out.append(ln)
            else:
                indent = m.group(1)
                out.append(f"{indent}@profile")
                out.append(ln)
        else:
            out.append(ln)

        if ln.strip():
            prev_nonempty = ln

    return out


def process_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8").splitlines()
    text = _ensure_profile_stub(text)
    text = _add_profile_decorators(text)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add @profile decorators to a Python file.")
    parser.add_argument("target", nargs="?", default="image_processor_hough.py", help="Path to the Python file to update")
    args = parser.parse_args()

    target_path = Path(args.target).expanduser().resolve()
    if not target_path.exists():
        raise SystemExit(f"Target file not found: {target_path}")

    process_file(target_path)
    print(f"Updated {target_path}")


if __name__ == "__main__":
    main()
