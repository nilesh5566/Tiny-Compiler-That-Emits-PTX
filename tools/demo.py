"""demo.py - compile the Mandelbrot kernel with kernc, run it, draw it in the terminal.

Runs on the CPU reference interpreters everywhere, and also on the GPU when
CuPy can see one - then checks that every pixel matches bit-for-bit.

    python tools/demo.py
    python tools/demo.py --width 100 --height 40 --iters 200
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerntools as kt  # noqa: E402

SHADES = " .:-=+*#%@"


def draw(result: str, w: int, h: int, iters: int):
    _, bufs = kt.parse_result(result)
    vals = [kt.b2i(x) for x in bufs[0][1]]
    for y in range(h):
        row = ""
        for x in range(w):
            k = vals[y * w + x]
            row += "@" if k >= iters else SHADES[min(len(SHADES) - 2, k * (len(SHADES) - 1) // max(1, iters // 4))]
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=78)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    w, h, n = a.width, a.height, a.width * a.height
    kern = str(kt.ROOT / "examples" / "mandelbrot.kern")
    block = 128
    launch = kt.Launch("mandelbrot", (n + block - 1) // block, block, [
        kt.Arg("i32buf", [0] * n), kt.Arg("i32", [w]), kt.Arg("i32", [h]), kt.Arg("i32", [a.iters])])
    tmp = tempfile.mkdtemp()
    lp = os.path.join(tmp, "m.launch")
    Path(lp).write_text(launch.text())

    ref, err = kt.run_cpu(kern, lp, "ast", ["-O0"])
    if err:
        sys.exit(err)
    opt, err = kt.run_cpu(kern, lp, "ir", ["-O1"])
    draw(opt, w, h, a.iters)
    print(f"\nAST oracle vs optimised IR (-O1): {'identical' if not kt.compare_results(ref, opt) else 'MISMATCH'}")
    if kt.gpu_available():
        import gpu_run
        g, err = gpu_run.compile_and_run(kern, launch, ["-O1"], tmp)
        print(f"AST oracle vs GPU (-O1 PTX):        "
              f"{err or ('identical, all ' + str(n) + ' pixels' if not kt.compare_results(ref, g) else 'MISMATCH')}")
    else:
        print("GPU: not available here (install CuPy on an NVIDIA machine, or use Colab) - CPU engines only")


if __name__ == "__main__":
    main()
