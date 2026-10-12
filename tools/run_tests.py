"""run_tests.py - KernC verification suite.

Layers (each is a folder under tests/):
  unit     C++ unit tests (build/bin/kernc_unit_tests)
  errors   programs that MUST be rejected, with the expected message and position
  codegen  FileCheck-style pattern checks on the generated PTX (+ ptxas validation)
  exec     differential execution: AST oracle vs IR -O0 vs IR -O1 (vs GPU with --gpu)
  edge     like exec, for corner cases: NaN, -0.0, INT_MIN, overflow, saturation, UB

    python tools/run_tests.py            # everything that runs without a GPU
    python tools/run_tests.py --gpu      # also execute every exec/edge test on the GPU
    python tools/run_tests.py -v --only exec
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerntools as kt  # noqa: E402

TESTS = kt.ROOT / "tests"


def directives(src: str, name: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(rf"^\s*//\s*{re.escape(name)}:(.*)$", src, re.M)]


# --------------------------------------------------------------------------- errors
def run_error_test(path: Path, env=None) -> str | None:
    src = path.read_text()
    want = directives(src, "ERROR")
    at = directives(src, "ERROR-AT")
    if not want:
        return "test has no // ERROR: directive"
    code, so, se = kt.run([kt.find_kernc(), "check", str(path)], env)
    if code != 1:
        return f"expected a compile error (exit 1), got exit {code}: {(se or so).strip()[:200]}"
    for w in want:
        if w not in se:
            return f"error message does not contain '{w}'. Got: {se.strip()}"
    if at:
        m = re.search(r":(\d+):(\d+): error:", se)
        if not m or f"{m.group(1)}:{m.group(2)}" != at[0]:
            return f"expected error at {at[0]}, got {m.group(0) if m else se.strip()}"
    return None


# --------------------------------------------------------------------------- codegen (FileCheck)
def pattern_to_regex(p: str) -> re.Pattern:
    """Literal text, except {{...}} which is a regular expression (LLVM FileCheck style)."""
    out, pos = "", 0
    for m in re.finditer(r"\{\{(.*?)\}\}", p):
        out += re.escape(p[pos:m.start()]) + "(?:" + m.group(1) + ")"
        pos = m.end()
    out += re.escape(p[pos:])
    return re.compile(out)


def filecheck(text: str, src: str) -> str | None:
    checks = [(m.group(1), m.group(2).strip())
              for m in re.finditer(r"^\s*//\s*(CHECK|CHECK-NOT|CHECK-NEXT):(.*)$", src, re.M)]
    if not checks:
        return "test has no CHECK directives"
    lines = text.splitlines()
    pos = 0                 # next line to search from
    pending_not = []        # CHECK-NOT patterns waiting for the next CHECK
    last_match = -1
    for kind, pat in checks:
        rx = pattern_to_regex(pat)
        if kind == "CHECK-NOT":
            pending_not.append((pat, rx))
            continue
        if kind == "CHECK-NEXT":
            idx = last_match + 1
            if idx >= len(lines) or not rx.search(lines[idx]):
                got = lines[idx] if idx < len(lines) else "<end of file>"
                return f"CHECK-NEXT: '{pat}' not on the line after the previous match (got: {got.strip()})"
            found = idx
        else:
            found = next((i for i in range(pos, len(lines)) if rx.search(lines[i])), None)
            if found is None:
                return f"CHECK: '{pat}' not found after line {pos + 1}"
        for npat, nrx in pending_not:
            for i in range(pos, found):
                if nrx.search(lines[i]):
                    return f"CHECK-NOT: '{npat}' matched line {i + 1}: {lines[i].strip()}"
        pending_not = []
        pos, last_match = found + 1, found
    for npat, nrx in pending_not:
        for i in range(pos, len(lines)):
            if nrx.search(lines[i]):
                return f"CHECK-NOT: '{npat}' matched line {i + 1}: {lines[i].strip()}"
    return None


def run_codegen_test(path: Path, tmp: str, env=None) -> str | None:
    src = path.read_text()
    opts = (directives(src, "RUN") or [""])[0].split()
    ptx = os.path.join(tmp, path.stem + ".ptx")
    code, so, se = kt.compile_ptx(str(path), ptx, opts, env)
    if code != 0:
        return f"compile failed: {se.strip()}"
    err = filecheck(Path(ptx).read_text(), src)
    if err:
        return err
    err = kt.ptxas_check(ptx)
    return f"ptxas rejected the PTX: {err.strip()[:300]}" if err else None


# --------------------------------------------------------------------------- exec / edge
def build_launch(src: str, path: Path) -> kt.Launch:
    spec = directives(src, "LAUNCH")
    if not spec:
        raise ValueError("test has no // LAUNCH: directive")
    kv = dict(x.split("=", 1) for x in spec[0].split())
    rng = __import__("random").Random(zlib.crc32(path.name.encode()))
    args = []
    for a in directives(src, "ARG"):
        parts = a.split(None, 2)
        kind = parts[0]
        if kind.endswith("buf"):
            n = int(parts[1])
            args.append(kt.Arg(kind, kt.gen_buffer(kind, n, parts[2] if len(parts) > 2 else "zeros", rng)))
        else:
            args.append(kt.Arg(kind, [kt.parse_scalar(kind, parts[1])]))
    return kt.Launch(kv["kernel"], int(kv.get("grid", 1)), int(kv.get("block", 1)), args)


def check_expects(src: str, result: str) -> str | None:
    status, bufs = kt.parse_result(result)
    want_status = directives(src, "EXPECT-STATUS")
    if want_status and not (status or "").startswith(want_status[0]):
        return f"expected status '{want_status[0]}', got '{status}'"
    if not want_status and not (status or "").startswith("ok"):
        return f"reference run reported '{status}'"
    for e in directives(src, "EXPECT"):
        m = re.fullmatch(r"arg(\d+)\[(\*|\d+(?:\.\.\d+)?)\]\s*==\s*(\S+)", e)
        if not m:
            return f"bad EXPECT directive: {e}"
        idx, sel, val = int(m.group(1)), m.group(2), m.group(3)
        kind, words = bufs[idx]
        if sel == "*":
            rng = range(len(words))
        elif ".." in sel:
            a, b = sel.split("..")
            rng = range(int(a), int(b) + 1)
        else:
            rng = [int(sel)]
        want = kt.parse_scalar(kind, val)
        for i in rng:
            got = words[i]
            ok = (got == want) or (kind == "f32" and kt.is_nan_bits(got) and kt.is_nan_bits(want))
            if not ok:
                shown = kt.b2f(got) if kind == "f32" else kt.b2i(got)
                return f"EXPECT {e}: element {i} is {shown!r} (0x{got:08x})"
    return None


def run_exec_test(path: Path, tmp: str, gpu: bool, env=None) -> str | None:
    src = path.read_text()
    try:
        launch = build_launch(src, path)
    except Exception as e:
        return f"bad test directives: {e}"
    lpath = os.path.join(tmp, path.stem + ".launch")
    Path(lpath).write_text(launch.text())
    kern = str(path)

    # 1. the oracle: AST interpreter at -O0 (the AST is never optimised)
    ref, err = kt.run_cpu(kern, lpath, "ast", ["-O0"], env)
    if err:
        return f"AST interpreter failed: {err}"
    err = check_expects(src, ref)
    if err:
        return "[ast] " + err

    # 2. IR interpreter before and after optimisation must agree bit-for-bit
    for engine, opts in [("ir", ["-O0"]), ("ir", ["-O1"])]:
        got, err = kt.run_cpu(kern, lpath, engine, opts, env)
        if err:
            return f"[{engine} {' '.join(opts)}] failed: {err}"
        diff = kt.compare_results(ref, got)
        if diff:
            return f"[{engine} {' '.join(opts)}] differs from AST oracle: {diff}"

    # 3. FMA contraction changes rounding: compare with a tolerance only when the test opts in
    fma_tol = directives(src, "FMA-ULP")
    got, err = kt.run_cpu(kern, lpath, "ir", ["-O1", "--fma"], env)
    if err:
        return f"[ir -O1 --fma] failed: {err}"
    if fma_tol:
        diff = kt.compare_results(ref, got, max_ulp=int(fma_tol[0]))
        if diff:
            return f"[ir -O1 --fma] beyond {fma_tol[0]} ulp: {diff}"

    # 4. generated PTX must be accepted by ptxas at every optimisation level
    for opts in (["-O0"], ["-O1"], ["-O1", "--fma"]):
        ptx = os.path.join(tmp, f"{path.stem}{''.join(opts)}.ptx")
        code, so, se = kt.compile_ptx(kern, ptx, opts, env)
        if code != 0:
            return f"compile {' '.join(opts)} failed: {se.strip()}"
        perr = kt.ptxas_check(ptx)
        if perr:
            return f"ptxas rejected {' '.join(opts)} PTX: {perr.strip()[:300]}"

    # 5. real hardware
    if gpu:
        import gpu_run
        status, _ = kt.parse_result(ref)
        for opts in (["-O0"], ["-O1"]):
            if status.startswith("ub"):
                break  # undefined behaviour: the GPU may do anything (or fault)
            got, err = gpu_run.compile_and_run(kern, launch, opts, tmp, env)
            if err:
                return f"[gpu {' '.join(opts)}] {err}"
            diff = kt.compare_results(ref, got)
            if diff:
                return f"[gpu {' '.join(opts)}] differs from AST oracle: {diff}"
        if fma_tol and not status.startswith("ub"):
            got, err = gpu_run.compile_and_run(kern, launch, ["-O1", "--fma"], tmp, env)
            if err:
                return f"[gpu -O1 --fma] {err}"
            diff = kt.compare_results(ref, got, max_ulp=int(fma_tol[0]))
            if diff:
                return f"[gpu -O1 --fma] beyond {fma_tol[0]} ulp: {diff}"
    return None


# --------------------------------------------------------------------------- driver
def collect(only: set | None):
    jobs = []
    for layer in ("errors", "codegen", "exec", "edge"):
        if only and layer not in only:
            continue
        for p in sorted((TESTS / layer).glob("*.kern")):
            jobs.append((layer, p))
    return jobs


def run_one(layer: str, path: Path, gpu: bool, env) -> str | None:
    with tempfile.TemporaryDirectory(prefix="kernc_t_") as tmp:
        try:
            if layer == "errors":
                return run_error_test(path, env)
            if layer == "codegen":
                return run_codegen_test(path, tmp, env)
            return run_exec_test(path, tmp, gpu, env)
        except Exception as e:  # a crash in the harness is a failure, not a pass
            return f"harness exception: {type(e).__name__}: {e}"


def run_unit(env=None) -> str | None:
    exe = kt.find_unit_tests()
    if not exe:
        return "kernc_unit_tests not built"
    code, so, se = kt.run([exe], env, timeout=300)
    if code != 0:
        fails = [l for l in (so + se).splitlines() if "FAIL" in l or "EXCEPTION" in l]
        return "unit tests failed: " + "; ".join(fails[:5])
    return None


def run_all(only=None, gpu=False, env=None, verbose=False, quiet=False, stop_early=False):
    """Returns a list of (layer, name, message) failures."""
    failures = []
    if not only or "unit" in only:
        err = run_unit(env)
        if err:
            failures.append(("unit", "kernc_unit_tests", err))
            if stop_early:
                return failures
        elif verbose:
            print("  PASS  unit/kernc_unit_tests")
    jobs = collect(only)
    workers = 1 if gpu else max(2, (os.cpu_count() or 2))
    with ThreadPoolExecutor(workers) as ex:
        futs = [(layer, p, ex.submit(run_one, layer, p, gpu, env)) for layer, p in jobs]
        for layer, p, f in futs:
            err = f.result()
            name = f"{layer}/{p.name}"
            if err:
                failures.append((layer, p.name, err))
                if not quiet:
                    print(f"  FAIL  {name}\n        {err}")
            elif verbose:
                print(f"  PASS  {name}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also run exec/edge tests on the GPU (needs CuPy)")
    ap.add_argument("--only", action="append", choices=["unit", "errors", "codegen", "exec", "edge"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.gpu and not kt.gpu_available():
        sys.exit("--gpu given but CuPy cannot see a CUDA GPU (pip install cupy-cuda12x, or use Colab)")
    only = set(a.only) if a.only else None
    print(f"kernc : {kt.rel(kt.find_kernc())}")
    print(f"ptxas : {kt.find_ptxas() or 'not found (PTX validation skipped)'}")
    print(f"gpu   : {'yes' if a.gpu else 'no (use --gpu)'}\n")
    failures = run_all(only, a.gpu, verbose=a.verbose)
    total = len(collect(only)) + (1 if not only or "unit" in only else 0)
    print(f"\n{total - len(failures)}/{total} passed" + (f", {len(failures)} FAILED" if failures else " - all good"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
