"""bench.py - benchmark of the optimiser.

  1. static:  IR and PTX instruction counts for examples/*.kern at -O0, -O1, -O1 --fma
  2. dynamic: IR instructions actually EXECUTED (summed over all threads) for every
              tests/exec/*.kern launch, at -O0 vs -O1. This is where loop-invariant
              code motion shows up, because it shrinks loop bodies, not code size.

    python tools/bench.py                       # print a markdown table
    python tools/bench.py --out docs/BENCHMARKS.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerntools as kt  # noqa: E402


def ptx_instructions(text: str) -> int:
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s.endswith(";") or s.startswith((".", "//")):
            continue
        if s.startswith("ld.param") or s.startswith("cvta.to.global"):
            continue  # kernel prologue, identical at every level
        n += 1
    return n


def measure(kern: Path, opts: list, tmp: str):
    ptx = os.path.join(tmp, kern.stem + ".ptx")
    code, so, se = kt.run([kt.find_kernc(), "compile", str(kern), "-o", ptx, "--stats"] + opts)
    if code != 0:
        raise RuntimeError(se)
    ir = sum(int(m.group(1)) for m in re.finditer(r"ir-instructions \S+ (\d+)", se))
    return ir, ptx_instructions(Path(ptx).read_text())


def dynamic(tmp: str):
    import run_tests
    rows = []
    for k in sorted((kt.ROOT / "tests" / "exec").glob("*.kern")):
        src = k.read_text()
        launch = run_tests.build_launch(src, k)
        lp = os.path.join(tmp, k.stem + ".launch")
        Path(lp).write_text(launch.text())
        counts = []
        for opt in ("-O0", "-O1"):
            code, so, se = kt.run([kt.find_kernc(), "run", str(k), "--launch", lp, "--engine", "ir", opt, "--stats"])
            m = re.search(r"ir-executed (\d+)", se)
            counts.append(int(m.group(1)) if m else 0)
        rows.append((k.name, *counts))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the table to this markdown file")
    a = ap.parse_args()
    rows, tot = [], [0, 0, 0, 0, 0]
    with tempfile.TemporaryDirectory() as tmp:
        for k in sorted((kt.ROOT / "examples").glob("*.kern")):
            ir0, p0 = measure(k, ["-O0"], tmp)
            ir1, p1 = measure(k, ["-O1"], tmp)
            _, pf = measure(k, ["-O1", "--fma"], tmp)
            rows.append((k.name, ir0, ir1, p0, p1, pf))
            for i, v in enumerate((ir0, ir1, p0, p1, pf)):
                tot[i] += v
    pct = lambda a, b: f"{100.0 * (a - b) / a:.0f}%"  # noqa: E731
    lines = ["| kernel | IR -O0 | IR -O1 | PTX -O0 | PTX -O1 | PTX -O1 --fma | PTX reduction |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, ir0, ir1, p0, p1, pf in rows:
        lines.append(f"| {name} | {ir0} | {ir1} | {p0} | {p1} | {pf} | {pct(p0, p1)} |")
    lines.append(f"| **total** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{tot[3]}** | **{tot[4]}** | "
                 f"**{pct(tot[2], tot[3])}** |")
    with tempfile.TemporaryDirectory() as tmp:
        dyn = dynamic(tmp)
    d0 = sum(r[1] for r in dyn)
    d1 = sum(r[2] for r in dyn)
    lines += ["", "Executed IR instructions (all threads, CPU IR interpreter):", "",
              "| test | -O0 | -O1 | reduction |", "|---|---:|---:|---:|"]
    for name, a0, a1 in dyn:
        lines.append(f"| {name} | {a0:,} | {a1:,} | {pct(a0, a1) if a0 else '-'} |")
    lines.append(f"| **total** | **{d0:,}** | **{d1:,}** | **{pct(d0, d1)}** |")
    table = "\n".join(lines)
    print(table)
    if a.out:
        Path(a.out).write_text("# Optimiser benchmark (static instruction counts)\n\n"
                               "Static counts: PTX excludes the parameter-loading prologue, identical at every level.\n\n"
                               + table + "\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
