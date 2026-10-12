"""kerntools.py - shared helpers for the KernC verification tools.

Finds the kernc / ptxas binaries, builds launch files, runs the CPU engines
and the GPU, and compares results bit-for-bit. Standard library only;
NumPy + CuPy are needed only for real-GPU runs (gpu_run.py).
"""
from __future__ import annotations

import glob
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if os.name == "nt" else ""


# --------------------------------------------------------------------------- binaries
def find_kernc() -> str:
    env = os.environ.get("KERNC_BIN")
    if env:
        return env
    candidates = [
        ROOT / "build" / "bin" / f"kernc{EXE}",
        ROOT / "build" / "bin" / "Release" / f"kernc{EXE}",
        ROOT / "build" / "Release" / f"kernc{EXE}",
        ROOT / "build" / f"kernc{EXE}",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    sys.exit("error: kernc executable not found. Build it first:\n"
             "  cmake -S . -B build\n  cmake --build build --config Release\n"
             "(or set KERNC_BIN to its path)")


def find_unit_tests() -> str | None:
    for c in [ROOT / "build" / "bin" / f"kernc_unit_tests{EXE}",
              ROOT / "build" / "Release" / f"kernc_unit_tests{EXE}"]:
        if c.exists():
            return str(c)
    return None


def find_ptxas() -> str | None:
    """ptxas validates PTX without needing a GPU. Looks in PATH, CUDA_PATH and pip's nvidia-cuda-nvcc."""
    env = os.environ.get("PTXAS")
    if env:
        return env
    p = shutil.which("ptxas")
    if p:
        return p
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda and (Path(cuda) / "bin" / f"ptxas{EXE}").exists():
        return str(Path(cuda) / "bin" / f"ptxas{EXE}")
    for sp in sys.path:
        hits = glob.glob(os.path.join(sp, "nvidia", "cuda_nvcc", "bin", f"ptxas{EXE}"))
        if hits:
            return hits[0]
    return None


def gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# --------------------------------------------------------------------------- float bits
def f2b(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def b2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def i2b(x: int) -> int:
    return x & 0xFFFFFFFF


def b2i(u: int) -> int:
    return u - (1 << 32) if u & 0x80000000 else u


def is_nan_bits(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF) != 0


def ulp_distance(a: int, b: int) -> int:
    """Distance between two f32 bit patterns in units in the last place."""
    def key(u):  # map to a monotonically ordered integer line
        return (0x80000000 - (u & 0x7FFFFFFF)) if u & 0x80000000 else (0x80000000 + u)
    return abs(key(a) - key(b))


# --------------------------------------------------------------------------- launches
@dataclass
class Arg:
    kind: str            # "i32", "f32", "i32buf", "f32buf"
    words: list = field(default_factory=list)  # buffer words, or [scalar]


@dataclass
class Launch:
    kernel: str
    grid: int
    block: int
    args: list

    def text(self) -> str:
        lines = [f"kernel {self.kernel}", f"grid {self.grid}", f"block {self.block}"]
        for a in self.args:
            if a.kind.endswith("buf"):
                lines.append(f"{a.kind} {len(a.words)} " + " ".join(f"{w:08x}" for w in a.words))
            else:
                lines.append(f"{a.kind} {a.words[0]:08x}")
        return "\n".join(lines) + "\n"


def parse_launch(text: str) -> Launch:
    l = Launch("", 1, 1, [])
    for line in text.splitlines():
        line = line.split("#")[0].split()
        if not line:
            continue
        k = line[0]
        if k == "kernel":
            l.kernel = line[1]
        elif k == "grid":
            l.grid = int(line[1])
        elif k == "block":
            l.block = int(line[1])
        elif k in ("i32", "f32"):
            l.args.append(Arg(k, [int(line[1], 16)]))
        else:
            n = int(line[1])
            l.args.append(Arg(k, [int(w, 16) for w in line[2:2 + n]]))
    return l


SPECIAL_F32 = [0.0, -0.0, 1.0, -1.0, 0.5, 3.0e38, -3.0e38, 1.0e-40, float("inf"), float("-inf"),
               float("nan"), 16777217.0, 2147483648.0, -2147483904.0]
SPECIAL_I32 = [0, 1, -1, 2, 7, -7, 2147483647, -2147483648, 65536, 46341]


def parse_scalar(kind: str, s: str) -> int:
    s = s.strip()
    if s.startswith("0x"):
        return int(s, 16) & 0xFFFFFFFF
    if kind == "i32":
        return i2b(int(s))
    return f2b(float(s))


def gen_buffer(kind: str, n: int, spec: str, rng: random.Random) -> list:
    """spec: zeros | range | random | random(lo,hi) | special | values(a,b,...)"""
    is_f = kind.startswith("f32")
    m = re.fullmatch(r"(\w+)(?:\((.*)\))?", spec.strip())
    name, params = m.group(1), (m.group(2) or "")
    if name == "zeros":
        return [0] * n
    if name == "range":
        return [f2b(float(i)) if is_f else i2b(i) for i in range(n)]
    if name == "values":
        vals = [parse_scalar("f32" if is_f else "i32", v) for v in params.split(",")]
        return [vals[i % len(vals)] for i in range(n)]
    if name == "special":
        pool = SPECIAL_F32 if is_f else SPECIAL_I32
        return [f2b(rng.choice(pool)) if is_f else i2b(rng.choice(pool)) for _ in range(n)]
    if name == "random":
        if params:
            lo, hi = (float(x) for x in params.split(","))
        else:
            lo, hi = (-100.0, 100.0) if is_f else (-1000, 1000)
        if is_f:
            return [f2b(rng.uniform(lo, hi)) for _ in range(n)]
        return [i2b(rng.randint(int(lo), int(hi))) for _ in range(n)]
    raise ValueError(f"unknown buffer generator '{spec}'")


# --------------------------------------------------------------------------- running
def run(cmd: list, env_extra: dict | None = None, input_text: str | None = None, timeout: int = 60):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, input=input_text, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -999, "", "timeout"


def compile_ptx(kern: str, out: str, opts: list, env_extra=None):
    return run([find_kernc(), "compile", kern, "-o", out] + opts, env_extra)


def ptxas_check(ptx: str, arch: str = "sm_75"):
    """Returns None if ptxas accepts the file (or ptxas is unavailable), else the error text."""
    ptxas = find_ptxas()
    if not ptxas:
        return None
    out = ptx + ".cubin"
    code, so, se = run([ptxas, f"-arch={arch}", ptx, "-o", out])
    try:
        os.remove(out)
    except OSError:
        pass
    return None if code == 0 else (se or so or f"ptxas exit {code}")


def run_cpu(kern: str, launch_path: str, engine: str, opts: list, env_extra=None):
    """Returns (result_text, error_text). result_text is None on failure."""
    code, so, se = run([find_kernc(), "run", kern, "--launch", launch_path, "--engine", engine] + opts, env_extra)
    if code != 0:
        return None, (se or so).strip() or f"exit code {code}"
    return so, None


def parse_result(text: str):
    status, bufs = None, {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "status":
            status = " ".join(parts[1:])
        elif parts[0] == "buf":
            idx, kind, n = int(parts[1]), parts[2], int(parts[3])
            bufs[idx] = (kind, [int(w, 16) for w in parts[4:4 + n]])
    return status, bufs


def compare_results(ref_text: str, got_text: str, max_ulp: int = 0, rel_tol: float = 0.0, max_report: int = 3):
    """Bit-exact comparison (all NaNs are considered equal). Returns None if equal, else a message."""
    rs, rb = parse_result(ref_text)
    gs, gb = parse_result(got_text)
    rs_kind = (rs or "").split()[0] if rs else rs
    gs_kind = (gs or "").split()[0] if gs else gs
    if rs_kind != gs_kind:
        return f"status differs: reference '{rs}' vs '{gs}'"
    if rs_kind == "ub":
        return None  # behaviour is undefined: any outcome is allowed
    problems = []
    for idx, (kind, words) in rb.items():
        if idx not in gb:
            return f"buffer {idx} missing"
        gwords = gb[idx][1]
        for j, (a, b) in enumerate(zip(words, gwords)):
            if a == b:
                continue
            if kind == "f32":
                if is_nan_bits(a) and is_nan_bits(b):
                    continue
                if max_ulp and not (is_nan_bits(a) or is_nan_bits(b)) and ulp_distance(a, b) <= max_ulp:
                    continue
                fa, fb = b2f(a), b2f(b)
                if rel_tol and math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= rel_tol * max(abs(fa), abs(fb), 1e-30):
                    continue
                problems.append(f"arg{idx}[{j}]: expected {fa!r} (0x{a:08x}) got {fb!r} (0x{b:08x})")
            else:
                problems.append(f"arg{idx}[{j}]: expected {b2i(a)} got {b2i(b)}")
            if len(problems) >= max_report:
                break
    return "; ".join(problems) if problems else None


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)
