# KernC

### A verified compiler from a tiny GPU kernel language to NVIDIA PTX

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Target](https://img.shields.io/badge/target-NVIDIA%20PTX-76B900)
![Tests](https://img.shields.io/badge/tests-86%2F86%20passing-brightgreen)
![Mutation score](https://img.shields.io/badge/mutation%20score-16%2F16-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

KernC compiles **Kern**, a small statically typed language for writing GPU
kernels, into **PTX**, the assembly language of NVIDIA GPUs. The compiler is
only half of the project. The other half is a **compiler-verification
toolkit** that checks the compiler is correct:

* an independent reference interpreter that acts as the "source of truth"
* FileCheck-style tests on the generated assembly
* differential execution on the CPU and on a real GPU, compared bit-for-bit
* a random program generator (in the style of Csmith) with automatic test-case reduction
* mutation testing, which plants bugs in the compiler to measure how good the tests are

While being tested this way, KernC's fuzzer found a **crash in NVIDIA's own
PTX assembler (`ptxas` 12.9)** on valid input. That story is told in
[section 11](#11-a-real-bug-found-ptxas-crashes-on-valid-ptx).

```kern
kernel saxpy(f32* x, f32* y, f32 a, i32 n) {
    let i = gid();
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
```

```
$ kernc compile saxpy.kern -O1          # kernel body, comments added
    mad.lo.s32      %r5, %r2, %r3, %r4;     // global thread id
    setp.lt.s32     %p1, %r5, %r1;          // i < n ?
    @!%p1 bra       $L_endif1;
    mul.wide.s32    %rd5, %r5, 4;           // byte offset, computed once
    add.s64         %rd6, %rd4, %rd5;       // &y[i], shared by load and store
    add.s64         %rd7, %rd2, %rd5;       // &x[i]
    ld.global.f32   %f2, [%rd7];
    mul.rn.f32      %f3, %f1, %f2;
    ld.global.f32   %f4, [%rd6];
    add.rn.f32      %f5, %f3, %f4;
    st.global.f32   [%rd6], %f5;
```

---

## Table of contents

1. [Why this project exists](#1-why-this-project-exists)
2. [Results at a glance](#2-results-at-a-glance)
3. [Quick start](#3-quick-start)
4. [A tour of the Kern language](#4-a-tour-of-the-kern-language)
5. [How the compiler works, stage by stage](#5-how-the-compiler-works-stage-by-stage)
6. [The optimiser](#6-the-optimiser)
7. [Running kernels: CPU interpreters and the GPU](#7-running-kernels-cpu-interpreters-and-the-gpu)
8. [The verification strategy](#8-the-verification-strategy)
9. [Differential fuzzing and test-case reduction](#9-differential-fuzzing-and-test-case-reduction)
10. [Mutation testing: testing the tests](#10-mutation-testing-testing-the-tests)
11. [A real bug found: ptxas crashes on valid PTX](#11-a-real-bug-found-ptxas-crashes-on-valid-ptx)
12. [Benchmarks](#12-benchmarks)
13. [Command-line reference](#13-command-line-reference)
14. [Project layout](#14-project-layout)
15. [Extending KernC](#15-extending-kernc)
16. [Troubleshooting](#16-troubleshooting)
17. [Limitations and roadmap](#17-limitations-and-roadmap)
18. [License](#18-license)

---

## 1. Why this project exists

A compiler is a program whose bugs are unusually dangerous. A miscompilation
produces a binary that runs, but computes the wrong answer, and nothing in the
user's code is wrong. GPU compilers make this harder in three ways:

* **Floating point is subtle.** `a*b + c` and a fused multiply-add give
  different results. NaN compares unequal to itself. `-0.0` is a different bit
  pattern from `0.0`. A compiler that "simplifies" any of these changes results.
* **Undefined behaviour must be handled precisely.** An optimiser may assume
  undefined behaviour never happens, but it must never *introduce* it.
* **The hardware is a black box.** The only way to know what the GPU does is
  to run code on it and compare.

KernC was built to practise the work a compiler-verification engineer does:
write a precise language specification, build a compiler that follows it, and
then build infrastructure that finds the places where the compiler, the
optimiser, the code generator, the assembler or the hardware disagree with the
specification.

The language is deliberately small so that the interesting part, the
verification, gets most of the effort.

---

## 2. Results at a glance

| | |
|---|---|
| **Compiler** | ~3,200 lines of C++17: lexer, recursive-descent parser, type checker, three-address IR, optimisation passes, PTX emitter, two interpreters |
| **Verification tools** | ~1,400 lines of Python: test runner, FileCheck implementation, fuzzer, delta-debugging reducer, mutation tester, GPU runner, benchmarks |
| **Test suite** | 24 C++ unit tests (84 checks), 36 error tests, 17 code-generation tests, 16 execution tests, 16 edge-case tests: **86/86 passing** |
| **Fuzzing** | 8,000+ random programs on both Linux and Windows after the last change, 0 mismatches; every generated PTX file accepted by `ptxas` |
| **Mutation testing** | **16/16** deliberately injected compiler bugs caught by the full suite; 9/16 by the fuzzer alone on the CPU, with every survivor explained |
| **Optimiser** | −12% static PTX instructions and **−20% executed instructions** at `-O1` compared with `-O0` |
| **Bug found** | A crash in NVIDIA `ptxas` 12.9.86 on valid PTX (Linux and Windows), reduced to a 30-line reproducer and worked around in the compiler |

---

## 3. Quick start

### Requirements

| need | why | required? |
|---|---|---|
| C++17 compiler + CMake 3.16+ | builds `kernc` | yes |
| Python 3.8+ | runs the test, fuzzing and benchmark tools (standard library only) | yes |
| `ptxas` | validates generated PTX; works **without a GPU** | recommended |
| NVIDIA GPU + driver + CuPy | runs the PTX on real hardware | optional |

### Linux / macOS

```bash
git clone https://github.com/<your-username>/kernc.git
cd kernc
./build.sh                                  # build + unit tests + full verification suite
pip install nvidia-cuda-nvcc-cu12           # optional: provides ptxas
python3 tools/run_tests.py                  # 86/86 passed - all good
python3 tools/demo.py                       # draws a Mandelbrot set computed by a compiled kernel
```

### Windows

1. Install **Python** from python.org and tick *"Add python.exe to PATH"*.
2. Install **Visual Studio 2022** (Community or Build Tools) with the
   **"Desktop development with C++"** workload. This includes CMake.
3. Open **Developer PowerShell for VS 2022** from the Start menu.
4. Build and test:
   ```
   cd C:\Projects\kernc
   cmake -S . -B build
   cmake --build build --config Release
   build\bin\kernc_unit_tests.exe
   python -m pip install nvidia-cuda-nvcc-cu12
   python tools\run_tests.py
   ```
   `.\build.bat` does all of this in one command.

The complete guide, with expected output for every step and a troubleshooting
table, is in [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md).

### Running on a GPU

```bash
pip install cupy-cuda12x                    # match your driver: nvidia-smi shows "CUDA Version"
python3 tools/run_tests.py --gpu            # every execution test also runs on the GPU
python3 tools/fuzz.py -n 300 --gpu          # every fuzzed program also runs on the GPU
```

No NVIDIA GPU? Open [notebooks/KernC_Colab.ipynb](notebooks/KernC_Colab.ipynb)
in Google Colab, choose *Runtime → Change runtime type → T4 GPU*, and run the cells.

### Your first compilation

```bash
build/bin/kernc compile examples/saxpy.kern -O1 -o saxpy.ptx
ptxas -arch=sm_75 saxpy.ptx -o saxpy.cubin   # NVIDIA's assembler accepts it
```

---

## 4. A tour of the Kern language

The full, precise rules are in [SPEC.md](SPEC.md). This section is the friendly version.

### 4.1 Types

| type | meaning |
|---|---|
| `i32` | 32-bit signed integer |
| `f32` | 32-bit IEEE-754 float |
| `bool` | `true` / `false` |
| `i32*`, `f32*` | pointer to GPU global memory. Only allowed as **kernel parameters** |

There are **no implicit conversions**. `let x: f32 = 0;` is an error; write
`0.0`, or convert explicitly with `f32(x)` and `i32(x)`.

### 4.2 Statements

```kern
let x = 5;              // declare; type inferred from the initialiser
let y: f32 = 1.5;       // declare with an explicit type
x = x + 1;              // assign
out[i] = 3;             // store into an array parameter
if (x < 3) { ... } else if (x < 7) { ... } else { ... }
while (x < 10) { ... }
return;                 // end this thread early
```

### 4.3 Expressions and built-ins

* Arithmetic `+ - * / %` (`%` is integer only), comparisons `< <= > >= == !=`,
  logic `&& || !`, unary minus.
* Comparisons cannot be chained: `a < b < c` is an error.
* Thread geometry:

| built-in | meaning | PTX |
|---|---|---|
| `tid()` | thread index inside its block | `%tid.x` |
| `bid()` | block index | `%ctaid.x` |
| `bdim()` | threads per block | `%ntid.x` |
| `gdim()` | blocks in the grid | `%nctaid.x` |
| `gid()` | global thread index `bid() * bdim() + tid()` | `mad.lo.s32` |

### 4.4 The semantics that matter for verification

These rules are where real compilers get things wrong, so Kern defines each one exactly:

| rule | example |
|---|---|
| Integer overflow **wraps** (defined, unlike C) | `2147483647 + 1 == -2147483648` |
| Integer division truncates toward zero | `-7 / 2 == -3`, `-7 % 2 == -1` |
| Every float operation is rounded separately; **no implicit FMA** | `a*b + c` is a multiply then an add |
| `!=` is true when either side is NaN | `x != x` is true for NaN |
| `-x` flips the sign bit | `-(0.0)` is `-0.0` |
| Subnormal floats are preserved | no flush-to-zero |
| `i32(f)` truncates, **saturates**, maps NaN to 0 | `i32(1e20) == 2147483647` |
| `&&` and `||` short-circuit | `i < n && a[i] > 0` never reads `a[i]` out of bounds |

**Undefined behaviour** (the program has no defined result): integer division
or remainder by zero, `INT_MIN / -1`, out-of-bounds array access, and data
races between threads.

### 4.5 Example programs

All in [`examples/`](examples):

| file | what it shows |
|---|---|
| `saxpy.kern` | `y = a*x + y`, the "hello world" of GPU computing |
| `sum_rows.kern` | one thread per matrix row, a `while` loop accumulating floats |
| `relu.kern` | a grid-stride loop: a few threads cover any array size |
| `collatz.kern` | integer-heavy, data-dependent loops and early `return` |
| `mandelbrot.kern` | float iteration with a compound loop condition |
| `polynomial.kern` | Horner's method, a natural fit for FMA |
| `histogram_rows.kern` | float→int conversion and clamping |

```kern
// Escape-time Mandelbrot: one thread per pixel
kernel mandelbrot(i32* iters, i32 width, i32 height, i32 max_iter) {
    let p = gid();
    if (p >= width * height) { return; }
    let cx = -2.0 + 3.0 * f32(p % width) / f32(width);
    let cy = -1.2 + 2.4 * f32(p / width) / f32(height);
    let x = 0.0;
    let y = 0.0;
    let k = 0;
    while (k < max_iter && x * x + y * y <= 4.0) {
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        k = k + 1;
    }
    iters[p] = k;
}
```

---

## 5. How the compiler works, stage by stage

```
 source.kern
     │
     ▼
 ┌─────────┐   ┌─────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌─────────────┐
 │  Lexer  │──►│ Parser  │──►│ Type checker │──►│ Lowering │──►│ Optimiser│──►│ Layout  │──►│ PTX emitter │──► .ptx
 └─────────┘   └─────────┘   └──────────────┘   └──────────┘   └──────────┘   └─────────┘   └─────────────┘
   tokens         AST          typed AST           IR (-O0)       IR (-O1)       RPO order
                   │                                  │
                   ▼                                  ▼
            AST interpreter                    IR interpreter          (used by the verification tools)
```

Every stage can be inspected from the command line. The examples below use
`examples/saxpy.kern`.

### 5.1 Lexer — `src/lexer.cpp`

Turns characters into tokens and records the line and column of each one, so
every later error can point at the exact spot.

```
$ kernc compile examples/saxpy.kern --dump-tokens
2:1   'kernel'
2:8   identifier   saxpy
2:13  '('
2:14  'f32'
2:17  '*'
2:19  identifier   x
```

It handles `//` and `/* */` comments, two-character operators (`<=`, `&&`),
float literals with exponents, and rejects things Kern does not have (a lone
`&` gets the message *"bitwise '&' is not supported"*).

### 5.2 Parser — `src/parser.cpp`

A hand-written **recursive-descent** parser for statements and **precedence
climbing** for expressions. It produces an abstract syntax tree (AST):

```
$ kernc compile examples/saxpy.kern --dump-ast
kernel saxpy(f32* x, f32* y, f32 a, i32 n)
  let i = (call gid)
  if (< i n)
    y[i] = (+ (* a x[i]) y[i])
```

Precedence follows C: `*` binds tighter than `+`, which binds tighter than
`<`, then `==`, `&&`, `||`. Two details worth noting: `else if` chains are
supported, and `-2147483648` is recognised as a single literal, because the
positive half `2147483648` does not fit in `i32`.

### 5.3 Type checker — `src/typecheck.cpp`

Walks the AST with a stack of scopes, resolves every name, and assigns a type
to every expression. Errors carry the exact position:

```
$ kernc check tests/errors/mixed_arith.kern
tests/errors/mixed_arith.kern:4:16: error: cannot apply '+' to f32 and i32
```

It rejects: undeclared or redeclared names, type mismatches, implicit
conversions, non-bool conditions, indexing non-pointers, copying or assigning
pointers, `bool` kernel parameters (PTX has no predicate parameters), unknown
functions, and wrong argument counts.

### 5.4 Lowering to IR — `src/lower.cpp`, `src/ir.h`

The typed AST becomes a **three-address intermediate representation**: basic
blocks of simple instructions over typed virtual registers.

```
$ kernc compile examples/saxpy.kern -O0 --dump-ir
saxpy(x: f32* = v0, y: f32* = v1, a: f32 = v2, n: i32 = v3)
entry:
  v4:i32 = gid
  v5:i32 = mov v4                       // the variable `i` owns register v5
  v6:bool = cmp.lt v5, v3
  condbr v6, then0, endif1
then0:
  v7:u64 = byteoff v5                   // &y[i] for the store
  v8:f32* = elemptr v1, v7
  v9:u64 = byteoff v5                   // &x[i]
  v10:f32* = elemptr v0, v9
  v11:f32 = load [v10]
  v12:f32 = mul v2, v11
  v13:u64 = byteoff v5                  // &y[i] again, for the load
  v14:f32* = elemptr v1, v13
  v15:f32 = load [v14]
  v16:f32 = add v12, v15
  store [v8], v16
  br endif1
endif1:
  ret
```

Design decisions:

* **Not SSA.** Each source variable owns one register for its whole life, and
  assignment is a `mov` into it. Loops need no phi nodes, which keeps lowering
  simple. The cost is that every optimisation pass must handle registers that
  are written more than once. That cost is deliberate: it is exactly the kind of
  subtlety that optimisation bugs come from, and the mutation tests probe it.
* **Addresses are explicit.** `a[i]` becomes `byteoff` (index × 4, widened to
  64 bits) and `elemptr` (base + offset). Because these are ordinary pure
  instructions, common-subexpression elimination can share them.
* **Short-circuit logic becomes branches.** `a && b` writes `a` into a result
  register and only evaluates `b` in a separate block if `a` was true.

### 5.5 Block layout — `layoutBlocks` in `src/passes.cpp`

Before code generation, blocks are reordered into **reverse post-order**: the
"true" branch is placed directly after its condition, and the only backward
jumps left are loop back-edges. This is the shape `nvcc` produces, and it turns
out to matter: see [section 11](#11-a-real-bug-found-ptxas-crashes-on-valid-ptx).

### 5.6 PTX code generation — `src/ptx.cpp`

Each IR type maps to a PTX register class, and each IR instruction to one or a
few PTX instructions.

| IR | PTX |
|---|---|
| `i32` / `f32` / `bool` / pointer, offset | `%r` (.b32) / `%f` (.f32) / `%p` (.pred) / `%rd` (.b64) |
| `add` / `sub` / `mul` (i32) | `add.s32` / `sub.s32` / `mul.lo.s32` |
| `add` / `sub` / `mul` / `div` (f32) | `add.rn.f32` / `sub.rn.f32` / `mul.rn.f32` / `div.rn.f32` |
| `cmp.ne` (f32) | `setp.neu.f32` (unordered: true for NaN) |
| `cvt.f2i` / `cvt.i2f` | `cvt.rzi.s32.f32` / `cvt.rn.f32.s32` |
| `byteoff` / `elemptr` | `mul.wide.s32` / `add.s64` |
| `load` / `store` | `ld.global` / `st.global` |
| pointer parameter | `ld.param.u64` + `cvta.to.global.u64` |
| float constant | exact hex bits, e.g. `0f3FC00000` for 1.5 |

Points worth calling out:

* **The `.rn` suffix on every float add and multiply.** Without an explicit
  rounding mode, `ptxas` is allowed to fuse a multiply and an add into an FMA,
  which changes results. Writing `.rn` forbids that, so the GPU computes exactly
  what the specification says.
* **`setp.neu` for `!=`.** The ordered comparison `setp.ne` is *false* when an
  operand is NaN, which would break `x != x`.
* **Register declarations are written last.** The emitter generates the body
  first, counts how many registers of each class it used, then writes the
  `.reg` declarations at the top.
* **PTX version is chosen from the target.** `--arch sm_75` emits `.version 7.0`;
  `sm_86` needs 7.1, `sm_89`/`sm_90` need 7.8. KernC picks the lowest version
  that supports the target, so the output works on the widest range of drivers.
* **Branch threading and a single exit.** Blocks that only contain a jump are
  skipped, and every `return` jumps to one shared `ret`.

---

## 6. The optimiser

`-O1` runs these passes repeatedly until nothing changes (at most 4 rounds).
Each one is in `src/passes.cpp`.

| pass | what it does | the trap it has to avoid |
|---|---|---|
| **Constant folding** | Evaluates operations whose operands are constants, and turns branches on constant conditions into jumps. | It must fold with the *language's* semantics (wrapping, truncation, saturation), and must **never fold undefined behaviour away**: `1 / 0` stays a run-time division. |
| **Copy propagation** | Inside a block, replaces uses of `a` after `a = mov b` with `b`. | If `b` is reassigned later, the copy is stale and must be forgotten. |
| **Global copy propagation** | The same across blocks, for registers written exactly once. | Only safe because Kern's scoping guarantees a definition comes before every use. |
| **Common subexpression elimination** | Reuses an earlier identical computation inside a block, including address arithmetic. | An entry must be dropped as soon as one of its operands is reassigned. Loads are never shared, because memory may have changed. |
| **Strength reduction** | `x * 8` → `x << 3`, `x * 1` → `x`. | Only for positive powers of two: `x * -8` is not `x << 3`. |
| **Loop-invariant code motion** | Finds loops (dominators and back edges) and moves computations that don't change inside the loop to before it. | A loop may run zero times, so it never hoists a division that could trap. Hoisting `100 / d` with `d == 0` would *create* undefined behaviour. |
| **Dead code elimination** | Removes pure instructions whose results are never used, and unreachable blocks. | Stores and branches are never dead; branch conditions count as uses. |
| **FMA contraction** (`--fma`) | `a*b + c` → `fma.rn.f32`. | Changes rounding, so it is **opt-in** and tested with a tolerance in ULPs (units in the last place). |

The same kernel after `-O1`:

```
$ kernc compile examples/saxpy.kern -O1 --dump-ir
then0:
  v7:u64 = byteoff v4                   // one offset, shared by x[i] and y[i]
  v8:f32* = elemptr v1, v7              // &y[i] computed once, used by load and store
  v10:f32* = elemptr v0, v7
  v11:f32 = load [v10]
  v12:f32 = mul v2, v11
  v15:f32 = load [v8]
  v16:f32 = add v12, v15
  store [v8], v16
```

`--stats` prints how many instructions each pass removed. The resulting PTX
for saxpy matches what `nvcc` produces for the same CUDA code, apart from the
FMA, which KernC only uses with `--fma`.

---

## 7. Running kernels: CPU interpreters and the GPU

### 7.1 Launch files

A *launch* describes one kernel invocation: which kernel, the grid and block
size, and the arguments. Values are written as hex bit patterns so that every
float, including NaN and `-0.0`, round-trips exactly.

```
kernel saxpy
grid 2
block 4
f32buf 6 3f800000 40000000 40400000 40800000 3f800000 40000000   # x = 1,2,3,4,1,2
f32buf 6 41200000 41200000 41200000 41200000 41200000 41200000   # y = 10 ...
f32 40000000                                                     # a = 2.0
i32 00000006                                                     # n = 6
```

Test files describe their launch in comments, and
`tools/make_launch.py` turns those into a launch file:

```bash
python3 tools/make_launch.py tests/exec/saxpy_golden.kern > saxpy.launch
```

### 7.2 The two CPU interpreters

```bash
kernc run tests/exec/saxpy_golden.kern --launch saxpy.launch --engine ast
kernc run tests/exec/saxpy_golden.kern --launch saxpy.launch --engine ir -O1
```

Both print the same format: a status line (`status ok`, or `status ub <reason>`
when the program does something undefined), then every output buffer as hex words.

* **The AST interpreter is the oracle.** It walks the typed AST and implements
  the specification directly. It shares no code with lowering, the optimiser
  or code generation, so a bug there cannot hide by also being in the oracle.
* **The IR interpreter** runs the compiler's own intermediate code at any
  optimisation level. Comparing it with the oracle tells you *which part* of
  the compiler is wrong without needing a GPU.

Threads run one after another. That is a faithful model of Kern, because the
language has no shared memory or barriers and data races are undefined.

### 7.3 The GPU

`tools/gpu_run.py` compiles a kernel with `kernc`, loads the PTX through CuPy,
runs it, and prints results in the same format, so outputs can be compared
directly:

```bash
python3 tools/gpu_run.py tests/exec/saxpy_golden.kern
```

It targets `sm_75` by default (PTX for an older target compiles on every newer
GPU) and drops to the GPU's own target on older cards.

---

## 8. The verification strategy

### 8.1 The core idea: an oracle and bit-exact comparison

Every execution engine must agree with the oracle, bit for bit. When one
doesn't, the place where they diverge points at the bug:

| disagreement | where the bug is |
|---|---|
| oracle ≠ IR interpreter at `-O0` | lowering (AST → IR) |
| IR at `-O0` ≠ IR at `-O1` | an optimisation pass |
| IR ≠ GPU | code generation, or a mismatch between the spec and the hardware |
| `ptxas` rejects or crashes | invalid PTX, or a bug in `ptxas` itself |

All NaN values are treated as equal (the spec leaves NaN payloads open). When
the oracle reports undefined behaviour, any result from the other engines is
accepted.

### 8.2 The test layers

Run everything with `python3 tools/run_tests.py`. Add `-v` to see every test,
`--only exec` to run one layer, and `--gpu` to add real-hardware runs.

#### Unit tests — `tests/unit/unit_tests.cpp`

24 C++ tests covering each stage: token streams, operator precedence, type
errors, scoping, integer wrapping, float conversion, `-0.0`, NaN comparison,
short-circuit evaluation, undefined-behaviour detection, and each optimisation
pass. Two of them build IR **by hand** and run a single pass on it, because some
pass bugs can't be reached from source code (see [section 10](#10-mutation-testing-testing-the-tests)).

#### Error tests — `tests/errors/` (36)

Programs that must be rejected, with the expected message *and* position:

```kern
// ERROR: cannot apply '+' to f32 and i32
// ERROR-AT: 4:16
kernel k(f32* o, i32 n) {
    o[0] = 1.5 + n;
}
```

#### Code-generation tests — `tests/codegen/` (17)

A small re-implementation of LLVM's **FileCheck**. Patterns must appear in
order; `{{...}}` is a regular expression; `CHECK-NOT` forbids a pattern between
two matches; `CHECK-NEXT` requires the very next line. Every file is also
assembled with `ptxas`.

```kern
// RUN: -O0
// `x != y` must be true when either side is NaN, so it needs the
// unordered comparison setp.neu, not setp.ne.
// CHECK: setp.neu.f32
// CHECK-NOT: setp.ne.f32
kernel k(f32* x, i32* o) {
    o[0] = i32(x[0] != x[1]);
}
```

#### Execution tests — `tests/exec/` (16) and edge cases — `tests/edge/` (16)

Each file declares a launch; inputs are generated deterministically. The
harness then:

1. runs the oracle and checks any `EXPECT:` golden values,
2. runs the IR interpreter at `-O0` and `-O1` and compares bit-for-bit,
3. runs `-O1 --fma` and, if the test allows it, compares within an ULP tolerance,
4. compiles PTX at `-O0`, `-O1` and `-O1 --fma` and assembles each with `ptxas`,
5. with `--gpu`, runs on the GPU and compares bit-for-bit.

```kern
// LAUNCH: kernel=k grid=1 block=1
// ARG: f32buf 1 values(0.0)
// ARG: f32buf 3 zeros
// ARG: i32buf 1 zeros
// EXPECT: arg1[0] == 0x80000000          (-0.0)
// EXPECT: arg1[1] == -inf
// EXPECT: arg2[0] == 1
kernel k(f32* x, f32* o, i32* eq) {
    let z = x[0];
    let nz = -z;
    o[0] = nz;
    o[1] = 1.0 / nz;
    o[2] = -0.0;
    eq[0] = i32(nz == z);      // -0.0 == +0.0 is true
}
```

Argument generators: `zeros`, `range`, `random`, `random(lo,hi)`, `special`
(NaN, ±inf, ±0, subnormals, INT_MIN, …), and `values(a,b,...)`.

The edge-case tests cover NaN comparisons, `-0.0`, integer overflow,
`INT_MIN / -1`, division by zero, out-of-bounds access, float→int saturation,
int→float rounding (`2²⁴ + 1`), subnormals, infinities, short-circuit guards,
deeply nested loops, and a loop that runs zero times with a division by zero
inside it.

---

## 9. Differential fuzzing and test-case reduction

Hand-written tests only check what someone thought of. The fuzzer
(`tools/gen.py` + `tools/fuzz.py`) generates thousands of random programs.

### 9.1 The generator

Every generated program is **well-typed by construction** (types are tracked
while generating) and **free of undefined behaviour by construction**:

| hazard | how the generator avoids it |
|---|---|
| division by zero, `INT_MIN / -1` | divisors are rewritten as `((b % 7) * (b % 7) + 1)`, always in [1, 37] |
| out-of-bounds access | indices are wrapped: `((e % n) + n) % n` |
| infinite loops | loops are counters from 0 to a small constant, never assigned inside the body |
| data races | each thread writes only its own output slot |

So any disagreement is a real bug, never a false alarm. The generator also
deliberately produces patterns that trigger optimisations: constant
sub-expressions (for the constant folder) and multiplications by powers of two
(for strength reduction). It mixes in special inputs such as NaN, infinities,
subnormals and INT_MIN.

```bash
python3 tools/gen.py --seed 7           # print one random program and its launch
```

### 9.2 What is checked for every program

1. The program type-checks and the oracle runs it without undefined behaviour.
2. IR at `-O0` equals the oracle (lowering).
3. IR at `-O1` equals the oracle (optimiser).
4. IR at `-O1 --fma` runs cleanly.
5. PTX at `-O0`, `-O1` and `-O1 --fma` is accepted by `ptxas` (code generator).
6. With `--gpu`: the GPU at `-O0` and `-O1` equals the oracle.

```bash
python3 tools/fuzz.py -n 1000                 # 1000 programs on the CPU
python3 tools/fuzz.py -n 300 --gpu            # also on the GPU
python3 tools/fuzz.py -n 500 --stmts 35       # bigger programs
```

### 9.3 Automatic reduction

A failing 40-line random program is hard to debug. The reducer shrinks it with
**delta debugging**:

1. try deleting each `{ ... }` block together with its header line,
2. then try deleting chunks of lines, halving the chunk size each time.

A candidate is kept only if it still compiles, still runs without undefined
behaviour, and **still fails in the same way**. Failures are saved to
`fuzz_failures/seed_<N>_<kind>/` with the original program, the reduced one,
the launch file and the error.

---

## 10. Mutation testing: testing the tests

A test suite that passes says nothing about how many bugs it would catch.
Mutation testing measures that. The compiler contains 16 deliberate bugs,
each switched on by an environment variable:

```bash
KERNC_MUTATE=7 build/bin/kernc compile file.kern      # Linux/macOS
$env:KERNC_MUTATE = 7                                 # Windows PowerShell
```

| # | mutant | the injected bug |
|---|---|---|
| 1 | constfold-sub-swapped | constant folder computes `b-a` for `a-b` |
| 2 | constfold-div-floor | constant folder rounds division toward −∞ |
| 3 | copyprop-stale-source | copy propagation ignores reassignment of the source |
| 4 | cse-stale-operands | CSE reuses a value after an operand changed |
| 5 | strength-neg-pow2 | `x * -8` turned into `x << 3` |
| 6 | dce-ignores-branch-uses | DCE deletes the comparison a branch depends on |
| 7 | lower-le-as-lt | `<=` compiled as `<` |
| 8 | lower-fneg-as-sub | `-x` compiled as `0.0 - x` (wrong for `-0.0`) |
| 9 | lower-and-eager | `&&` evaluates its right side even when the left is false |
| 10 | lower-while-as-if | a `while` body runs at most once |
| 11 | constfold-cvt-rounds | constant folder rounds float→int instead of truncating |
| 12 | ptx-fne-ordered | float `!=` emitted as `setp.ne` (false for NaN) |
| 13 | ptx-index-scale-8 | array index scaled by 8 bytes instead of 4 |
| 14 | ptx-missing-cvta | pointer parameters used without `cvta.to.global` |
| 15 | typecheck-mixed-arith | type checker accepts `i32 + f32` |
| 16 | licm-hoists-division | LICM hoists a division out of a loop that may run zero times |

```bash
python3 tools/mutate.py                      # full suite: 16/16 killed
python3 tools/mutate.py --no-tests           # fuzzer only: 9/16 on the CPU
```

Reports: [MUTATION_REPORT.md](MUTATION_REPORT.md) and
[docs/MUTATION_FUZZ_ONLY.md](docs/MUTATION_FUZZ_ONLY.md).

### What mutation testing taught this project

* **Two mutants survived the first full run** (#3 and #4). End-to-end tests
  couldn't reach those pass bugs, because earlier passes reshape the IR before
  the buggy pass sees it. The fix was to test each pass directly on hand-built IR.
* **The fuzzer had a blind spot.** Run alone, it caught neither
  constant-folding mutant, because random programs rarely combine two
  literals. The generator now produces constant sub-expressions on purpose.
* **Some survivors are expected, and each one is explained.** Because the
  fuzzer only generates programs without undefined behaviour, evaluating `&&`
  eagerly (#9) or hoisting a safe division (#16) changes nothing it can observe:
  these are *equivalent mutants* for that input space. Hand-written edge tests
  kill them.
* **Some bugs are invisible at run time.** Mutant #14 (missing `cvta.to.global`)
  produces correct results on current GPUs, because generic and global addresses
  coincide there. Only a static check on the PTX catches it.

---

## 11. A real bug found: ptxas crashes on valid PTX

While fuzzing, one program made NVIDIA's PTX assembler **crash with a
segmentation fault** instead of compiling it or reporting an error.

| | |
|---|---|
| tool | `ptxas` 12.9.86 (from the `nvidia-cuda-nvcc-cu12` package) |
| symptom | crash at `-O1`, `-O2` and `-O3`: segfault on Linux, `Internal error` / access violation (`0xC0000005`) on Windows; the same file assembles fine at `-O0` |
| platforms / targets | Linux and Windows x86-64; `sm_61`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` |
| frequency | about 2 in 1,000 random programs, before the workaround |

**How it was narrowed down:**

1. The fuzzer's reducer shrank the program from 34 lines to 20 while the
   crash kept happening.
2. The generated PTX was then minimised line by line, keeping a deletion only
   if `ptxas -O0` still accepted the file and `ptxas -O3` still crashed.
3. The result was rewritten into a clean, well-formed 30-line kernel:
   [`repro_minimal.ptx`](docs/findings/ptxas-crash/repro_minimal.ptx).

**The trigger:** a conditional branch whose two targets lead to the same
place, inside a region entered by a backward jump that is not a loop. KernC's
original block order could produce this for an `||` inside an `if` inside
nested loops; `nvcc` never emits this layout for structured code.

**The workaround** (on by default): reverse-post-order block layout and branch
threading. `--no-workarounds` turns them off so the crash can still be
reproduced, and `tests/codegen/ptxas_crash_regression.kern` keeps it fixed.

```bash
ptxas -arch=sm_75 -O0 docs/findings/ptxas-crash/repro_minimal.ptx -o a.cubin   # ok
ptxas -arch=sm_75 -O3 docs/findings/ptxas-crash/repro_minimal.ptx -o a.cubin   # crashes on 12.9.86 (Linux and Windows)
```

Reproduced with `ptxas` 12.9.86 on both Linux and Windows. Other CUDA versions
may behave differently. Full write-up: [docs/findings/ptxas-crash/README.md](docs/findings/ptxas-crash/README.md).

---

## 12. Benchmarks

`python3 tools/bench.py` measures two things.

**Static size** — instructions in the generated code (PTX excludes the
parameter-loading prologue, which is identical at every level):

| kernel | IR -O0 | IR -O1 | PTX -O0 | PTX -O1 | PTX -O1 --fma | PTX reduction |
|---|---:|---:|---:|---:|---:|---:|
| collatz | 43 | 41 | 43 | 41 | 41 | 5% |
| histogram_rows | 68 | 58 | 67 | 57 | 57 | 15% |
| mandelbrot | 65 | 54 | 66 | 55 | 53 | 17% |
| polynomial | 33 | 32 | 34 | 33 | 32 | 3% |
| relu | 37 | 33 | 38 | 34 | 34 | 11% |
| saxpy | 17 | 13 | 19 | 15 | 14 | 21% |
| sum_rows | 27 | 24 | 28 | 25 | 25 | 11% |
| **total** | **290** | **255** | **295** | **260** | **256** | **12%** |

**Dynamic count** — IR instructions actually *executed*, summed over all threads
of every execution test. This is where loop-invariant code motion shows up,
because it shrinks loop bodies rather than code size: **358,410 at `-O0` →
288,424 at `-O1` (−20%)**. Per-test numbers are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

These are instruction counts, not GPU timings.

---

## 13. Command-line reference

### `kernc`

```
kernc compile <file.kern> [options]      compile to PTX
kernc check   <file.kern>                parse and type-check only
kernc run     <file.kern> --launch <f>   execute on a CPU interpreter
kernc mutants                            list the injectable bugs
kernc --version | --help
```

| option | meaning |
|---|---|
| `-o <file>` | output file (default: standard output) |
| `-O0` / `-O1` | optimisation level (default `-O0`) |
| `--fma` | allow fused multiply-add (changes rounding) |
| `--arch sm_XX` | PTX target, default `sm_75` |
| `--dump-tokens`, `--dump-ast`, `--dump-ir` | print an intermediate stage |
| `--stats` | print instruction counts and what each pass changed |
| `--engine ast\|ir` | for `run`: which interpreter |
| `--no-workarounds` | disable the ptxas-crash workaround |

Exit codes: `0` success, `1` compile error, `2` usage, I/O or runtime error.

### `tools/`

| script | purpose | useful flags |
|---|---|---|
| `run_tests.py` | the whole verification suite | `--gpu`, `--only <layer>`, `-v` |
| `fuzz.py` | differential fuzzing with reduction | `-n`, `--seed`, `--stmts`, `--gpu`, `--no-reduce` |
| `gen.py` | print one random program | `--seed`, `--stmts` |
| `mutate.py` | mutation testing | `--fuzz N`, `--only 3 4`, `--no-tests`, `--no-fuzz`, `--gpu` |
| `bench.py` | static and dynamic instruction counts | `--out file.md` |
| `gpu_run.py` | run one kernel on the GPU | `-O 0\|1`, `--fma` |
| `make_launch.py` | turn a test's directives into a launch file | |
| `demo.py` | Mandelbrot in the terminal, CPU and GPU cross-checked | `--width`, `--height`, `--iters` |

Environment variables: `KERNC_BIN` (path to `kernc`), `PTXAS` (path to
`ptxas`), `KERNC_MUTATE` (enable one mutant).

---

## 14. Project layout

```
kernc/
├── CMakeLists.txt            build: kernclib (library), kernc, kernc_unit_tests
├── SPEC.md                   the language specification the compiler is verified against
├── build.sh / build.bat      one-command build + test (Linux/macOS / Windows)
├── src/
│   ├── common.h              types, errors, and the shared integer/float semantics
│   ├── lexer.*               characters → tokens
│   ├── ast.h, parser.*       tokens → AST (recursive descent + precedence climbing)
│   ├── typecheck.*           scopes, name resolution, typing, error messages
│   ├── ir.h, ir.cpp          the intermediate representation and its printer
│   ├── lower.*               AST → IR
│   ├── semantics.*           evaluation of one IR operation (IR interpreter, constant folder)
│   ├── passes.*              optimisation passes, block layout
│   ├── ptx.*                 IR → PTX
│   ├── interp.*              AST interpreter (oracle) and IR interpreter
│   ├── launch.*              launch-file format shared by CPU and GPU runners
│   ├── mutate.h              the 16 injectable bugs
│   ├── driver.*              the pipeline as one library call
│   └── main.cpp              command-line interface
├── tests/
│   ├── unit/                 C++ unit tests
│   ├── errors/               must-reject programs
│   ├── codegen/              FileCheck tests on PTX
│   ├── exec/                 differential execution tests
│   └── edge/                 corner cases and undefined behaviour
├── tools/                    Python verification tools (standard library only)
├── examples/                 example kernels
├── docs/
│   ├── INSTALL_WINDOWS.md    step-by-step Windows setup
│   ├── BENCHMARKS.md
│   ├── MUTATION_FUZZ_ONLY.md
│   └── findings/ptxas-crash/ the ptxas crash write-up and reproducers
├── notebooks/KernC_Colab.ipynb   run everything on a free Colab GPU
├── MUTATION_REPORT.md
└── .github/workflows/ci.yml  CI on Linux (GCC + ptxas) and Windows (MSVC)
```

---

## 15. Extending KernC

### Write your own kernel

Create `mykernel.kern`, then:

```bash
build/bin/kernc check mykernel.kern               # does it type-check?
build/bin/kernc compile mykernel.kern -O1 --dump-ir
build/bin/kernc compile mykernel.kern -O1 -o mykernel.ptx
```

### Add a test

* **Error test:** a file in `tests/errors/` with `// ERROR: <message part>` and `// ERROR-AT: line:col`.
* **Codegen test:** a file in `tests/codegen/` with `// RUN: <options>` and `// CHECK:` lines.
* **Execution test:** a file in `tests/exec/` or `tests/edge/` with `// LAUNCH:`,
  `// ARG:` and optional `// EXPECT:`, `// EXPECT-STATUS: ub`, `// FMA-ULP: N`.

New files are picked up automatically by `run_tests.py`.

### Add an optimisation pass

1. Write `bool myPass(Function& f)` in `src/passes.cpp`, returning whether it changed anything.
2. Add it to the `-O1` loop in `optimize()`.
3. Add a unit test that runs it on hand-built IR, and a `tests/exec/` test that triggers it.
4. Add a mutant to `src/mutate.h` that breaks it in a realistic way, and check
   `python3 tools/mutate.py` still reaches 100%.
5. Run `python3 tools/fuzz.py -n 2000`.

---

## 16. Troubleshooting

| problem | fix |
|---|---|
| `error: kernc executable not found` | build first, and run the tools from the project folder; or set `KERNC_BIN` |
| `ptxas: not found (PTX validation skipped)` | `pip install nvidia-cuda-nvcc-cu12`, install the CUDA Toolkit, or set `PTXAS` |
| `cmake` not recognised (Windows) | use **Developer PowerShell for VS 2022**, not a normal terminal |
| `--gpu given but CuPy cannot see a CUDA GPU` | no NVIDIA GPU or driver; use the Colab notebook |
| CuPy `DLL load failed` / driver too old | `cupy-cuda12x` needs a driver showing `CUDA Version: 12.x` in `nvidia-smi` |
| a fuzz failure appears | look in `fuzz_failures/`; `reduced.kern` is the small version |

More Windows-specific help: [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md#troubleshooting).

---

## 17. Limitations and roadmap

**Current limitations**

* One-dimensional launches only.
* No shared memory, barriers or atomics.
* Optimisations are mostly local to a basic block, plus whole-function facts
  for copy propagation and loop-invariant code motion.
* GPU execution goes through CuPy (Python), not a native C++ runner.
* The Windows/MSVC build is checked by the CI workflow, and GPU execution by
  the Colab notebook; run both after cloning to confirm them on your setup.

**Next steps**

* `shared` memory and `bar.sync`, enabling tiled reductions, plus a static check
  that warns about barriers inside divergent control flow.
* SSA form and global value numbering.
* A C++ GPU runner using the CUDA Driver API (`cuModuleLoadDataEx` with the JIT error log).
* Differential testing against LLVM's NVPTX back end on the same programs.
* GPU timing benchmarks alongside the instruction counts.

---

## 18. License

MIT. See [LICENSE](LICENSE).

**Author:** Nilesh Kumar Nayak — M.Tech (IT), IIIT Allahabad
