"""fuzz.py - differential fuzzing campaign with automatic test-case reduction.

For every random (UB-free) program from gen.py:
  1. AST interpreter          -> the reference result (oracle)
  2. IR interpreter at -O0    -> must match bit-for-bit   (catches lowering bugs)
  3. IR interpreter at -O1    -> must match bit-for-bit   (catches optimiser bugs)
  4. PTX at -O0/-O1/--fma     -> must be accepted by ptxas (catches codegen bugs)
  5. GPU at -O0/-O1 (--gpu)   -> must match bit-for-bit   (catches codegen + hardware-semantics bugs)
Failing programs are saved to fuzz_failures/ and shrunk with delta debugging.

    python tools/fuzz.py -n 500                  # 500 programs, CPU only
    python tools/fuzz.py -n 200 --gpu            # also run every program on the GPU
    python tools/fuzz.py -n 300 --seed 1000 --no-reduce
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
import kerntools as kt  # noqa: E402

FAIL_DIR = kt.ROOT / "fuzz_failures"


def check_program(src: str, launch: kt.Launch, work: str, gpu=False, env=None):
    """Returns (kind, detail) for the first problem found, or (None, None) if all engines agree.

    kind is one of: invalid (generator produced a bad program), ir-O0, ir-O1, fma,
    compile, ptxas, gpu-O0, gpu-O1.
    """
    kern = os.path.join(work, "prog.kern")
    lpath = os.path.join(work, "prog.launch")
    Path(kern).write_text(src)
    Path(lpath).write_text(launch.text())

    code, so, se = kt.run([kt.find_kernc(), "check", kern], env)
    if code != 0:
        return "invalid", (se or so).strip()
    ref, err = kt.run_cpu(kern, lpath, "ast", ["-O0"], env)
    if err:
        return "invalid", "AST interpreter: " + err
    if not kt.parse_result(ref)[0].startswith("ok"):
        return "invalid", "reference run is not UB-free: " + kt.parse_result(ref)[0]

    for tag, opts in (("ir-O0", ["-O0"]), ("ir-O1", ["-O1"])):
        got, err = kt.run_cpu(kern, lpath, "ir", opts, env)
        if err:
            return tag, err
        diff = kt.compare_results(ref, got)
        if diff:
            return tag, diff
    got, err = kt.run_cpu(kern, lpath, "ir", ["-O1", "--fma"], env)
    if err or not kt.parse_result(got)[0].startswith("ok"):
        return "fma", err or kt.parse_result(got)[0]

    for opts in (["-O0"], ["-O1"], ["-O1", "--fma"]):
        ptx = os.path.join(work, "prog" + "".join(opts) + ".ptx")
        code, so, se = kt.compile_ptx(kern, ptx, opts, env)
        if code != 0:
            return "compile", (se or so).strip()
        perr = kt.ptxas_check(ptx)
        if perr:
            return "ptxas", perr.strip()[:500]

    if gpu:
        import gpu_run
        for tag, opts in (("gpu-O0", ["-O0"]), ("gpu-O1", ["-O1"])):
            got, err = gpu_run.compile_and_run(kern, launch, opts, work, env)
            if err:
                return tag, err
            diff = kt.compare_results(ref, got)
            if diff:
                return tag, diff
    return None, None


# --------------------------------------------------------------------------- reduction
def reduce_program(src: str, launch: kt.Launch, kind: str, gpu=False, env=None, log=print) -> str:
    """Delta debugging (ddmin) over lines, plus whole-block removal.

    A candidate is kept only if it still compiles, still runs UB-free on the
    oracle, and still fails in the SAME way (same `kind`).
    """
    work = tempfile.mkdtemp(prefix="kernc_reduce_")
    tries = 0

    def interesting(lines):
        nonlocal tries
        tries += 1
        k, _ = check_program("\n".join(lines) + "\n", launch, work, gpu, env)
        return k == kind

    lines = src.rstrip("\n").split("\n")
    changed = True
    while changed:
        changed = False
        # pass 1: remove a whole `{ ... }` block together with its header line
        i = 0
        while i < len(lines):
            if lines[i].rstrip().endswith("{") and i > 0:
                depth, j = 0, i
                while j < len(lines):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if depth <= 0 and j > i:
                        break
                    j += 1
                if j < len(lines) and lines[j].strip() == "}":
                    cand = lines[:i] + lines[j + 1:]
                    if interesting(cand):
                        lines, changed = cand, True
                        continue
            i += 1
        # pass 2: classic ddmin over single lines / chunks
        n = 2
        while len(lines) >= 2:
            chunk = max(1, len(lines) // n)
            removed = False
            for start in range(0, len(lines), chunk):
                cand = lines[:start] + lines[start + chunk:]
                if cand and interesting(cand):
                    lines, removed, changed = cand, True, True
                    n = max(n - 1, 2)
                    break
            if not removed:
                if chunk == 1:
                    break
                n = min(n * 2, len(lines))
    shutil.rmtree(work, ignore_errors=True)
    log(f"    reduced to {len(lines)} lines in {tries} attempts")
    return "\n".join(lines) + "\n"


def save_failure(seed: int, src: str, launch: kt.Launch, kind: str, detail: str, reduced: str | None) -> Path:
    d = FAIL_DIR / f"seed_{seed}_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prog.kern").write_text(src)
    (d / "prog.launch").write_text(launch.text())
    (d / "failure.txt").write_text(f"kind: {kind}\nseed: {seed}\n\n{detail}\n")
    if reduced:
        (d / "reduced.kern").write_text(reduced)
    return d


# --------------------------------------------------------------------------- campaign
def fuzz_one(seed: int, gpu=False, env=None, stmts=14):
    src, launch = gen.generate(seed, stmts)
    work = tempfile.mkdtemp(prefix="kernc_fuzz_")
    try:
        kind, detail = check_program(src, launch, work, gpu, env)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return seed, src, launch, kind, detail


def campaign(n: int, seed0: int, gpu=False, env=None, reduce=True, stop_on_first=False, quiet=False, stmts=14):
    """Runs n programs. Returns list of (seed, kind, detail)."""
    failures = []
    t0 = time.time()
    workers = 1 if gpu else max(2, os.cpu_count() or 2)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fuzz_one, s, gpu, env, stmts) for s in range(seed0, seed0 + n)]
        for done, f in enumerate(futs, 1):
            seed, src, launch, kind, detail = f.result()
            if kind:
                failures.append((seed, kind, detail))
                if not quiet:
                    print(f"  [{kind}] seed {seed}: {detail[:200]}")
                    reduced = None
                    if reduce and kind != "invalid":
                        reduced = reduce_program(src, launch, kind, gpu, env)
                    path = save_failure(seed, src, launch, kind, detail, reduced)
                    print(f"    saved to {kt.rel(path)}")
                if stop_on_first:
                    for g in futs:
                        g.cancel()
                    break
            if not quiet and done % 50 == 0:
                print(f"  {done}/{n} programs, {len(failures)} failures, {time.time() - t0:.0f}s")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=300, help="number of programs")
    ap.add_argument("--seed", type=int, default=1, help="first seed")
    ap.add_argument("--stmts", type=int, default=14, help="statement budget per program")
    ap.add_argument("--gpu", action="store_true", help="also execute on the GPU (needs CuPy)")
    ap.add_argument("--no-reduce", action="store_true")
    a = ap.parse_args()
    if a.gpu and not kt.gpu_available():
        sys.exit("--gpu given but CuPy cannot see a CUDA GPU")
    print(f"fuzzing {a.n} programs (seeds {a.seed}..{a.seed + a.n - 1}), "
          f"ptxas: {'yes' if kt.find_ptxas() else 'no'}, gpu: {'yes' if a.gpu else 'no'}")
    t0 = time.time()
    fails = campaign(a.n, a.seed, a.gpu, reduce=not a.no_reduce, stmts=a.stmts)
    kinds = {}
    for _, k, _ in fails:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"\n{a.n} programs in {time.time() - t0:.1f}s: {len(fails)} failures {kinds if kinds else ''}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
