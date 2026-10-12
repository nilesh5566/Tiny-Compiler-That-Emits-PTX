# Finding: ptxas segfaults on valid PTX (found by the KernC fuzzer)

## Summary

While fuzzing KernC, `tools/fuzz.py` flagged a program whose generated PTX made
**ptxas crash with a segmentation fault** instead of either compiling it or
reporting an error.

| | |
|---|---|
| Tool | `ptxas` — Cuda compilation tools, release 12.9, V12.9.86 (from the `nvidia-cuda-nvcc-cu12` pip wheel, Linux x86-64) |
| Symptom | `Segmentation fault` (exit 139) at `-O1`, `-O2`, `-O3` |
| At `-O0` | the same file assembles without error (so the PTX is accepted as valid) |
| Targets | reproduced for `sm_61`, `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` |
| Frequency | 2 out of 1,000 random KernC programs (block layout without the workaround) |

A crash on valid input is a bug in the assembler regardless of what the input
looks like. It has **not** been checked against other CUDA versions — please
re-run the commands below with your own CUDA toolkit before claiming anything
about other versions.

## How it was found and reduced

1. `tools/fuzz.py` generated a random, UB-free Kern program (seed 46).
2. The CPU engines agreed, but `ptxas` exited with signal 11. The fuzzer
   classifies that as a `ptxas` failure, saves it to `fuzz_failures/`, and
3. delta-debugs it: `reduce_program()` deleted blocks and lines while the
   crash kept happening, shrinking the 34-line program to 20 lines
   (`repro.kern`).
4. The PTX itself was then minimised line by line (keep a line deletion only
   if `ptxas -O0` still accepts the file and `ptxas -O3` still crashes), and
   finally rewritten by hand into a clean, well-formed 30-line kernel:
   `repro_minimal.ptx`.

## Reproduce

```bash
ptxas -arch=sm_75 -O0 repro_minimal.ptx -o out.cubin   # ok
ptxas -arch=sm_75 -O3 repro_minimal.ptx -o out.cubin   # Segmentation fault on 12.9.86

# the original compiler output (workarounds disabled):
kernc compile repro.kern -O1 --no-workarounds -o crash.ptx
ptxas -arch=sm_75 crash.ptx -o out.cubin               # Segmentation fault on 12.9.86
```

## What triggers it

All crashing shapes share the same control-flow layout:

```
X:  @%p bra A        // A is an empty block that just falls into B
    bra.uni B
...
Y:  ... bra.uni X    // a non-loop BACKWARD branch into X
```

i.e. a conditional branch whose two targets are effectively the same block,
inside a region that is entered by a backward jump that is *not* a loop
back-edge. KernC's original layout emitted blocks in creation order, so an
`||` / `&&` inside an `if` could produce exactly this. `nvcc` never emits such
layouts for structured code, which is presumably why the path is untested.

## Workaround in KernC (enabled by default)

* **Reverse-post-order block layout** (`layoutBlocks` in `src/passes.cpp`):
  every edge points forward except loop back-edges.
* **Branch threading** in the PTX emitter: blocks that only contain a jump are
  skipped, and `@p bra L; bra.uni L` collapses into `bra.uni L`.
* **Single exit block**: every `return` jumps to one `$L__exit: ret;`.

`--no-workarounds` turns off the first two (the ones that matter; the single
exit alone does not prevent the crash) so the crash can still be reproduced.
`tests/codegen/ptxas_crash_regression.kern` keeps the workaround from
regressing. After the fix, 7,000+ further fuzz programs produced no ptxas
failures.

## Files

* `repro_minimal.ptx` – hand-cleaned 30-line reproducer
* `repro.kern` – fuzzer-reduced Kern source
* `repro_kernc_noworkaround.ptx` – what KernC emits for it with `--no-workarounds`
