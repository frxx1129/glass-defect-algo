"""Summarize line_profiler output and save to txt.

Usage examples (PowerShell):
    .\.venv\Scripts\python.exe read_profile_hough.py              # defaults
    .\.venv\Scripts\python.exe read_profile_hough.py my.prof -n 15 -o summary.txt
"""
import argparse
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

DEFAULT_PROFILE = 'profile_hough.prof'
DEFAULT_OUTPUT = 'profile_hough_summary.txt'


@dataclass
class ProfileData:
    timings: Dict[Tuple[str, int, str], Any]
    unit: float


def _load_stats(path: Path) -> ProfileData:
    """Load a kernprof/line_profiler .prof file; avoid hard dependency on LineStats."""
    with path.open('rb') as f:
        obj = pickle.load(f)

    # Typical format: (timings_dict, unit_float)
    if isinstance(obj, tuple) and len(obj) == 2:
        timings, unit = obj
        return ProfileData(timings=timings, unit=unit)

    # Some versions pickle a LineStats-like object with .timings/.unit attributes
    if hasattr(obj, 'timings') and hasattr(obj, 'unit'):
        return ProfileData(timings=obj.timings, unit=obj.unit)

    raise TypeError(f"Unexpected profile format: {type(obj)}")


def _summarize(stats: ProfileData, top_n: int) -> str:
    lines = []
    entries: Dict[Tuple[str, int, str], Any] = stats.timings

    summary = []
    for (fn, lineno, funcname), records in entries.items():
        total_ticks = sum(r[2] for r in records)
        total_time_s = total_ticks * stats.unit
        total_hits = sum(r[1] for r in records)
        summary.append((total_time_s, total_hits, fn, lineno, funcname))

    summary.sort(key=lambda t: t[0], reverse=True)
    top = summary[:top_n]

    lines.append(f"Top {len(top)} functions by total time (line_profiler):")
    lines.append(f"{'Rank':>4} {'Total(ms)':>12} {'Calls':>8}  Location")
    for idx, (tot_s, calls, fn, lineno, func) in enumerate(top, 1):
        lines.append(
            f"{idx:>4} {tot_s*1e3:12.2f} {calls:8d}  {fn}:{lineno} :: {func}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize line_profiler results and save to txt.")
    parser.add_argument("profile", nargs="?", default=DEFAULT_PROFILE, help="Path to .prof file (from kernprof -l)")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Where to save the text report")
    parser.add_argument("-n", "--top", type=int, default=50, help="How many functions to show")
    parser.add_argument("--full", action="store_true", help="Also append full line_profiler table")
    args = parser.parse_args()

    prof_path = Path(args.profile)
    if not prof_path.exists():
        raise SystemExit(f"Profile file not found: {prof_path}")

    stats = _load_stats(prof_path)
    report = _summarize(stats, args.top)

    if args.full:
        # Reuse line_profiler CLI output for full detail
        cmd = [sys.executable, '-m', 'line_profiler', str(prof_path)]
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        report += "\nFull line_profiler output:\n" + cp.stdout

    out_path = Path(args.output)
    out_path.write_text(report, encoding='utf-8')
    print(f"Wrote report to {out_path}")


if __name__ == '__main__':
    main()
