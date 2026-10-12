# KernC — a verified compiler from a tiny GPU kernel language to NVIDIA PTX

KernC compiles **Kern**, a small statically typed kernel language, to **PTX**
(NVIDIA's GPU assembly), and surrounds the compiler with a full
**compiler-verification** toolkit: an independent reference interpreter,
FileCheck-style codegen tests, differential execution on CPU and GPU, a
Csmith-style random program generator with automatic test-case reduction, and
mutation testing to measure how good all of that is.

```kern
kernel saxpy(f32* x, f32* y, f32 a, i32 n) {
    let i = gid();
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```
```
$ kernc compile saxpy.kern -O1        (body, after the parameter loads)
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.lt.s32 %p1, %r5, %r1;
    @!%p1 bra $L_endif1;
$L_then0:
    mul.wide.s32 %rd5, %r5, 4;
    add.s64 %rd6, %rd4, %rd5;
    add.s64 %rd7, %rd2, %rd5;
    ld.global.f32 %f2, [%rd7];
    mul.rn.f32 %f3, %f1, %f2;
    ld.global.f32 %f4, [%rd6];
    add.rn.f32 %f5, %f3, %f4;
    st.global.f32 [%rd6], %f5;
$L_endif1:
$L__exit:
    ret;
```

## Results at a glance

| | |
|---|---|
| Compiler | ~3,200 lines of C++17: lexer, recursive-descent parser, type checker, three-address IR, 9 optimisation passes, PTX emitter, two interpreters |
| Verification tools | ~1,400 lines of Python: test runner, FileCheck, fuzzer, delta-debugging reducer, mutation tester, GPU runner, benchmarks |
| Test suite | 24 C++ unit tests (84 checks) + 36 error, 17 codegen, 16 execution, 16 edge-case tests — all passing |
| Fuzzing | 8,000+ random UB-free programs after the last change, 0 mismatches; every PTX file validated by `ptxas` |
| Mutation score | **16/16 (100%)** injected compiler bugs caught by the full suite; 9/16 by the fuzzer alone on CPU (each survivor explained) |
| Optimiser | −12% static PTX instructions, **−20% executed instructions** (-O1 vs -O0) |
| Real bug found | The fuzzer found a **segfault in NVIDIA's `ptxas` 12.9** on valid PTX; reduced to a 30-line reproducer and worked around ([write-up](docs/findings/ptxas-crash/README.md)) |

## Architecture

```
               ┌────────────────────────── kernc (C++) ──────────────────────────┐
 .kern ──► Lexer ──► Parser ──► AST ──► Type checker ──► Lowering ──► IR ──► Passes ──► Layout ──► PTX emitter ──► .ptx
                                 │                                    │   constfold      RPO       branch threading
                                 │                                    │   copyprop        │        single exit
                                 │                                    │   globalcopy      │
                                 │                                    │   CSE / LICM      │
                                 │                                    │   strength / DCE  │
                                 │                                    │   FMA (opt-in)    │
                                 ▼                                    ▼                   │
                         AST interpreter                       IR interpreter             │
                        (the oracle, -O0)                   (-O0, -O1, --fma)             │
                                 │                                    │                   ▼
                                 └────────────► compare bit-for-bit ◄─┴──────── GPU via CuPy (optional)
                                                       ▲
                       tools/: run_tests.py · fuzz.py (+ reducer) · mutate.py · bench.py
```

* **The oracle shares no code with the compiler back end.** It walks the AST and
  implements [SPEC.md](SPEC.md) literally. A disagreement localises the bug:
  oracle ≠ IR at -O0 → lowering; IR -O0 ≠ IR -O1 → an optimisation pass;
  IR ≠ GPU → code generation (or a spec/hardware mismatch).
* **The IR is not SSA**: each variable owns one register. That keeps lowering
  simple (no phi nodes) but forces every pass to cope with multiply-defined
  registers — exactly the kind of subtlety the mutation tests probe.

## Quick start

**Windows:** follow [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) (step by step, from a fresh machine).

**Linux / macOS / Colab:**
```bash
./build.sh                                 # cmake build + unit tests + verification suite
pip install nvidia-cuda-nvcc-cu12          # optional: ptxas, validates PTX without a GPU
python3 tools/run_tests.py                 # 86/86 passed
python3 tools/fuzz.py -n 500               # differential fuzzing
python3 tools/mutate.py                    # mutation score
python3 tools/demo.py                      # Mandelbrot in the terminal, CPU (+GPU) checked
```
**With an NVIDIA GPU** (`pip install cupy-cuda12x`) add `--gpu` to `run_tests.py` and
`fuzz.py`. No GPU? Open [notebooks/KernC_Colab.ipynb](notebooks/KernC_Colab.ipynb) on a free Colab T4.

## Command line

```
kernc compile <file.kern> [-o out.ptx] [-O0|-O1] [--fma] [--arch sm_75]
              [--dump-tokens] [--dump-ast] [--dump-ir] [--stats] [--no-workarounds]
kernc check   <file.kern>
kernc run     <file.kern> --launch <file.launch> [--engine ast|ir] [-O0|-O1] [--fma]
kernc mutants
```

## The language

Types `i32`, `f32`, `bool`, and pointer parameters `i32*`/`f32*`; `let`,
assignment, array store, `if`/`else if`/`else`, `while`, `return`;
builtins `gid() tid() bid() bdim() gdim()`, conversions `i32(x)` / `f32(x)`.
The [specification](SPEC.md) pins down everything a verifier needs:
wrap-around integer overflow (defined, unlike C), IEEE float semantics with no
implicit FMA contraction, NaN-aware comparisons (`!=` → `setp.neu`), saturating
float→int conversion matching `cvt.rzi`, short-circuit `&&`/`||`, and an explicit
list of undefined behaviour (division by zero, `INT_MIN / -1`, out-of-bounds,
data races).

## Optimisations (`-O1`)

| pass | what it does |
|---|---|
| constant folding | evaluates constant operations *with the language's semantics*, folds constant branches; never folds UB away |
| copy propagation | block-local, invalidates copies when the source is redefined |
| global copy propagation | across blocks, for registers defined exactly once |
| CSE | block-local common-subexpression elimination, including address arithmetic |
| strength reduction | `x * 2^k → x << k` (only for non-negative powers of two) |
| LICM | hoists loop-invariant, non-trapping code to the preheader (dominators + natural loops) |
| DCE / unreachable blocks | removes unused pure instructions and dead blocks |
| FMA contraction (`--fma`) | `a*b+c → fma.rn.f32`; opt-in because it changes rounding |
| RPO layout (always) | reverse-post-order block layout — also the ptxas-crash workaround |

Benchmarks: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Verification strategy

| layer | location | what it catches |
|---|---|---|
| unit tests | `tests/unit/` | each stage in isolation, including passes run directly on hand-built IR |
| error tests | `tests/errors/` | invalid programs must be rejected with the right message **and** line:column |
| codegen tests | `tests/codegen/` | FileCheck-style `CHECK:` / `CHECK-NOT:` / `CHECK-NEXT:` on the PTX, plus `ptxas` acceptance |
| execution tests | `tests/exec/` | oracle vs IR -O0 vs IR -O1 (vs GPU), bit-exact; golden `EXPECT:` values |
| edge cases | `tests/edge/` | NaN, −0.0, subnormals, ±inf, INT_MIN, overflow, saturation, UB detection, zero-trip loops |
| fuzzing | `tools/fuzz.py` | random well-typed, UB-free programs (Csmith-style); mismatches are delta-debugged automatically |
| mutation testing | `tools/mutate.py` | 16 deliberate compiler bugs (`KERNC_MUTATE=<id>`) — measures the suite itself |

Reports: [MUTATION_REPORT.md](MUTATION_REPORT.md), [docs/MUTATION_FUZZ_ONLY.md](docs/MUTATION_FUZZ_ONLY.md).

### Things the verification work turned up

* **ptxas segfault (12.9.86)** on valid PTX produced by an `||` inside an `if`
  inside nested loops. Found by fuzzing, delta-debugged from 34 to 20 source
  lines, minimised to a 30-line PTX file, triggers isolated, workaround added
  with a regression test. → [docs/findings/ptxas-crash](docs/findings/ptxas-crash/README.md)
* **Two mutants survived the first full run** (#3 copy propagation, #4 CSE):
  end-to-end tests could not reach those pass bugs because earlier passes
  reshape the IR. Fixed by testing passes directly on hand-built IR.
* **The fuzzer had a blind spot**: fuzzer-only mutation testing showed it never
  killed the constant-folding mutants, because random programs rarely combine
  two literals. The generator now targets constant folding and strength reduction.
* **FMA is a semantic decision, not an optimisation detail**: `a*b+c` and
  `fma(a,b,c)` round differently; KernC emits `.rn` on every add/mul to stop
  ptxas from contracting, and makes contraction an explicit opt-in.

## Repository layout

```
src/            compiler (C++17): lexer, parser, typecheck, lower, passes, ptx, interp, main
tests/          unit/ errors/ codegen/ exec/ edge/
tools/          run_tests.py fuzz.py gen.py mutate.py gpu_run.py bench.py demo.py make_launch.py
examples/       saxpy, sum_rows, relu (grid-stride), collatz, mandelbrot, polynomial, histogram
docs/           INSTALL_WINDOWS.md, BENCHMARKS.md, MUTATION_FUZZ_ONLY.md, findings/
notebooks/      KernC_Colab.ipynb (GPU run on a free Colab T4)
SPEC.md         the language specification the compiler is verified against
```

## Limitations and next steps

* One-dimensional launches; no shared memory, barriers or atomics yet
  (`shared` + `bar.sync` would enable tiled reductions, plus a checker for
  barriers in divergent code).
* Passes are block-local or use simple whole-function facts; SSA + global value
  numbering would be the natural next step.
* GPU execution goes through CuPy; a C++ runner using the CUDA Driver API
  (`cuModuleLoadDataEx` with the JIT error log) would remove the Python dependency.
* Differential testing against an LLVM NVPTX back end (emit LLVM IR, compare outputs).

## License

MIT — see [LICENSE](LICENSE).
