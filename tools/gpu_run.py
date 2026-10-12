"""gpu_run.py - run kernc-generated PTX on a real NVIDIA GPU via CuPy.

Reads the same launch format as `kernc run` and prints results in the same
format, so GPU output can be diffed bit-for-bit against the CPU interpreters.

    python tools/gpu_run.py tests/exec/saxpy_golden.kern            # launch taken from // LAUNCH: / // ARG:
    python tools/gpu_run.py examples/saxpy.kern my.launch -O1       # explicit launch file

Needs: an NVIDIA GPU + driver, and `pip install cupy-cuda12x` (or cupy-cuda11x).
Google Colab (Runtime -> T4 GPU) has everything preinstalled.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerntools as kt  # noqa: E402


def device_arch() -> str:
    """PTX target for the current GPU. sm_75 PTX JIT-compiles on every newer GPU."""
    import cupy as cp
    cc = int(cp.cuda.Device().compute_capability)  # e.g. 75, 86, 89
    return f"sm_{min(cc, 75)}"


def run_gpu(ptx_path: str, launch: kt.Launch) -> str:
    """Load PTX, launch the kernel, return result text in kernc's format."""
    import cupy as cp
    import numpy as np

    module = cp.RawModule(path=str(ptx_path))
    fn = module.get_function(launch.kernel)
    dev_args, keep = [], []
    for a in launch.args:
        if a.kind.endswith("buf"):
            # Keep raw 32-bit words: the kernel only sees a pointer, and bits stay exact.
            arr = cp.asarray(np.array(a.words if a.words else [0], dtype=np.uint32))
            keep.append((a, arr))
            dev_args.append(arr)
        elif a.kind == "i32":
            dev_args.append(np.array([a.words[0]], dtype=np.uint32).view(np.int32)[0])
        else:
            dev_args.append(np.array([a.words[0]], dtype=np.uint32).view(np.float32)[0])
    fn((launch.grid,), (launch.block,), tuple(dev_args))
    cp.cuda.Device().synchronize()

    lines = ["status ok"]
    for i, a in enumerate(launch.args):
        if a.kind.endswith("buf"):
            arr = next(x for (aa, x) in keep if aa is a)
            words = cp.asnumpy(arr).tolist()[: len(a.words)]
            kind = "f32" if a.kind == "f32buf" else "i32"
            lines.append(f"buf {i} {kind} {len(words)} " + " ".join(f"{w:08x}" for w in words))
    return "\n".join(lines) + "\n"


def compile_and_run(kern: str, launch: kt.Launch, opts: list, workdir: str | None = None,
                    env: dict | None = None) -> tuple[str | None, str | None]:
    """Compile with kernc for this GPU and run. Returns (result_text, error)."""
    workdir = workdir or tempfile.mkdtemp(prefix="kernc_gpu_")
    tag = "".join(o.strip("-") for o in opts)  # distinct file per option set
    ptx = os.path.join(workdir, f"{Path(kern).stem}_{tag}_gpu.ptx")
    code, so, se = kt.compile_ptx(kern, ptx, opts + ["--arch", device_arch()], env)
    if code != 0:
        return None, "compile failed: " + (se or so)
    try:
        return run_gpu(ptx, launch), None
    except Exception as e:  # driver JIT errors, launch failures, ...
        return None, f"GPU error: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kern")
    ap.add_argument("launch", nargs="?", help="launch file (default: the test's // LAUNCH: directives)")
    ap.add_argument("-O", dest="opt", default="1", choices=["0", "1"])
    ap.add_argument("--fma", action="store_true")
    a = ap.parse_args()
    if not kt.gpu_available():
        sys.exit("No CUDA GPU visible to CuPy. Install cupy-cuda12x on a machine with an NVIDIA GPU, or use Google Colab.")
    if a.launch:
        launch = kt.parse_launch(Path(a.launch).read_text())
    else:
        import run_tests
        launch = run_tests.build_launch(Path(a.kern).read_text(), Path(a.kern))
    opts = [f"-O{a.opt}"] + (["--fma"] if a.fma else [])
    out, err = compile_and_run(a.kern, launch, opts)
    if err:
        sys.exit(err)
    print(out, end="")


if __name__ == "__main__":
    main()
