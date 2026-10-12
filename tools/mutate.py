"""mutate.py - mutation testing: how good is the verification suite?

The compiler contains deliberate bugs ("mutants", listed by `kernc mutants`)
that are switched on one at a time with KERNC_MUTATE=<id>. For each mutant
we run the test suite, then (if it survived) a fuzzing campaign. A mutant is
KILLED if anything fails. The mutation score = killed / total.

A surviving mutant is not a failure of this script - it is a precise
description of a gap in the tests, i.e. the next test to write.

    python tools/mutate.py                 # all mutants, 200 fuzz programs each
    python tools/mutate.py --fuzz 500 --only 3 4
    python tools/mutate.py --no-tests      # measure the fuzzer alone
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuzz  # noqa: E402
import kerntools as kt  # noqa: E402
import run_tests  # noqa: E402


def list_mutants():
    code, so, se = kt.run([kt.find_kernc(), "mutants"])
    out = []
    for line in so.splitlines():
        mid, name, desc = line.split("\t")
        out.append((int(mid), name, desc))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuzz", type=int, default=200, help="fuzz programs per surviving mutant")
    ap.add_argument("--only", type=int, nargs="*", help="mutant ids to run")
    ap.add_argument("--no-tests", action="store_true", help="skip the test suite (fuzzer only)")
    ap.add_argument("--no-fuzz", action="store_true", help="skip fuzzing (test suite only)")
    ap.add_argument("--gpu", action="store_true", help="include GPU execution in tests and fuzzing")
    ap.add_argument("--report", default=str(kt.ROOT / "MUTATION_REPORT.md"))
    a = ap.parse_args()

    # sanity: the unmutated compiler must be clean, otherwise every mutant looks "killed"
    print("baseline (no mutant) ...", end=" ", flush=True)
    base = run_tests.run_all(gpu=a.gpu, quiet=True)
    if base:
        sys.exit(f"FAILED - fix the test suite first: {base[0]}")
    print("clean\n")

    rows = []
    t0 = time.time()
    for mid, name, desc in list_mutants():
        if a.only and mid not in a.only:
            continue
        env = {"KERNC_MUTATE": str(mid)}
        killer, detail = None, ""
        if not a.no_tests:
            fails = run_tests.run_all(gpu=a.gpu, env=env, quiet=True, stop_early=True)
            if fails:
                layer, test, msg = fails[0]
                killer, detail = f"tests ({layer})", f"{layer}/{test}: {msg}"
        if not killer and not a.no_fuzz:
            fails = fuzz.campaign(a.fuzz, 1, a.gpu, env, reduce=False, stop_on_first=True, quiet=True)
            real = [f for f in fails if f[1] != "invalid"]
            if real:
                seed, kind, msg = real[0]
                killer, detail = f"fuzzer ({kind})", f"seed {seed}: {msg}"
        status = "KILLED" if killer else "survived"
        print(f"  #{mid:<2} {name:<26} {status:<9} {killer or ''}")
        if detail:
            print(f"       {detail[:150]}")
        rows.append((mid, name, desc, status, killer or "-", detail))

    killed = sum(1 for r in rows if r[3] == "KILLED")
    score = 100.0 * killed / max(1, len(rows))
    print(f"\nmutation score: {killed}/{len(rows)} = {score:.0f}%   ({time.time() - t0:.0f}s)")

    lines = ["# Mutation testing report", "",
             f"Mutation score: **{killed}/{len(rows)} ({score:.0f}%)**  ",
             f"Fuzz budget per surviving mutant: {a.fuzz} programs; GPU: {'yes' if a.gpu else 'no'}", "",
             "| # | mutant | injected bug | result | killed by |", "|---|---|---|---|---|"]
    for mid, name, desc, status, killer, _ in rows:
        lines.append(f"| {mid} | `{name}` | {desc} | {status} | {killer} |")
    lines += ["", "## First failure seen for each killed mutant", ""]
    for mid, name, _, status, _, detail in rows:
        if detail:
            lines.append(f"- **#{mid} {name}**: {detail[:300]}")
    Path(a.report).write_text("\n".join(lines) + "\n")
    print(f"report written to {kt.rel(a.report)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
