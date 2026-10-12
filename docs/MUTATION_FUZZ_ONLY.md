# Mutation testing report

Mutation score: **9/16 (56%)**  
Fuzz budget per surviving mutant: 300 programs; GPU: no

| # | mutant | injected bug | result | killed by |
|---|---|---|---|---|
| 1 | `constfold-sub-swapped` | constant folder computes b-a for a-b | KILLED | fuzzer (ir-O1) |
| 2 | `constfold-div-floor` | constant folder rounds i32 division toward -inf | KILLED | fuzzer (ir-O1) |
| 3 | `copyprop-stale-source` | copy propagation ignores redefinition of the copy source | KILLED | fuzzer (ir-O1) |
| 4 | `cse-stale-operands` | CSE reuses a value after one of its operands changed | survived | - |
| 5 | `strength-neg-pow2` | strength reduction turns x*(-2^k) into x<<k | KILLED | fuzzer (ir-O1) |
| 6 | `dce-ignores-branch-uses` | DCE forgets that branch conditions are uses | KILLED | fuzzer (ir-O1) |
| 7 | `lower-le-as-lt` | `<=` is lowered as `<` | KILLED | fuzzer (ir-O0) |
| 8 | `lower-fneg-as-sub` | f32 negation lowered as 0.0 - x (wrong for -0.0) | KILLED | fuzzer (ir-O0) |
| 9 | `lower-and-eager` | `&&` evaluates its right side even when the left is false | survived | - |
| 10 | `lower-while-as-if` | while loop body runs at most once | KILLED | fuzzer (ir-O0) |
| 11 | `constfold-cvt-rounds` | constant folder rounds f32->i32 instead of truncating | KILLED | fuzzer (ir-O1) |
| 12 | `ptx-fne-ordered` | f32 `!=` emitted as ordered setp.ne (false for NaN) | survived | - |
| 13 | `ptx-index-scale-8` | array index scaled by 8 bytes instead of 4 | survived | - |
| 14 | `ptx-missing-cvta` | pointer params used without cvta.to.global | survived | - |
| 15 | `typecheck-mixed-arith` | type checker accepts i32 + f32 | survived | - |
| 16 | `licm-hoists-division` | LICM hoists i32 division out of loops (may run zero times) | survived | - |

## First failure seen for each killed mutant

- **#1 constfold-sub-swapped**: seed 37: arg3[0]: expected 3 got -3
- **#2 constfold-div-floor**: seed 45: arg3[0]: expected 85976919 got 85053336; arg3[1]: expected 916162716 got 915239133; arg3[2]: expected 1545735919 got 1544812336
- **#3 copyprop-stale-source**: seed 80: arg3[0]: expected 1803725889 got 1803726850; arg3[1]: expected 1803725920 got 1803726881; arg3[2]: expected 1803725920 got 1803726881
- **#5 strength-neg-pow2**: seed 1: arg3[1]: expected 15 got 47; arg3[2]: expected -1 got 63; arg3[3]: expected -17 got 79
- **#6 dce-ignores-branch-uses**: seed 1: arg3[0]: expected 31 got 0; arg3[1]: expected 15 got 0; arg3[2]: expected -1 got 0
- **#7 lower-le-as-lt**: seed 12: arg3[0]: expected 1201859487 got 1201829696; arg3[1]: expected 1202783008 got 1202753217; arg3[2]: expected 1203706529 got 1203676738
- **#8 lower-fneg-as-sub**: seed 79: arg2[0]: expected -inf (0xff800000) got inf (0x7f800000); arg2[1]: expected -inf (0xff800000) got inf (0x7f800000); arg2[2]: expected -inf (0xff800000) got inf (0x7f800000); arg3[0]: expected -957 got -988
- **#10 lower-while-as-if**: seed 2: arg3[0]: expected 901923420 got -1652081544
- **#11 constfold-cvt-rounds**: seed 206: arg3[0]: expected 1021033810 got 1021063601; arg3[1]: expected -2045619933 got -2045590142; arg3[2]: expected 1275000227 got 1275030018

## Why some mutants survive the fuzzer alone (on the CPU)

This run uses **only** `tools/fuzz.py` (no hand-written tests) and no GPU.
Every survivor is explained by a design property, and each one is killed by
another layer of the full suite (see `MUTATION_REPORT.md`, 16/16):

| mutant | why the fuzzer cannot see it | caught by |
|---|---|---|
| #9 `lower-and-eager` | Generated programs are UB-free, so evaluating the right side of `&&` early has no visible effect: it is an *equivalent mutant* for this input space. | `codegen/short_circuit.kern`, `edge/short_circuit_guard.kern` |
| #16 `licm-hoists-division` | Same reason: the generator forces divisors into [1, 37], so a hoisted division can never trap. | `edge/licm_zero_trip.kern`, unit `pass_licm` |
| #12, #13, #14 (`ptx-*`) | On the CPU the fuzzer only checks that ptxas *accepts* the PTX; these bugs produce valid PTX that computes the wrong thing. With `--gpu`, #13 and usually #12 are killed; #14 survives even on a GPU, because generic and global addresses coincide on current hardware. | FileCheck tests in `tests/codegen/` |
| #15 `typecheck-mixed-arith` | The generator only produces well-typed programs. | `tests/errors/`, unit `typecheck_errors` |
| #4 `cse-stale-operands` | Copy propagation runs before CSE and usually renames the stale operand away, so the bug is only reachable through rare IR shapes. | unit `pass_cse_isolated` |

Two lessons that came out of this experiment and were fed back into the tools:

* The first fuzzer-only run killed **neither** constant-folding mutant (#1, #2):
  random programs rarely contained an operation whose operands were *both*
  literals. The generator now emits constant sub-expressions and
  multiplications by powers of two on purpose, and both mutants (plus #5) die.
* Mutants #3 and #4 originally survived the *whole* suite. That showed the
  end-to-end tests could not reach those pass bugs, which led to
  `pass_copyprop_isolated`, `pass_cse_isolated` (passes tested directly on
  hand-built IR) and `exec/copy_prop_param.kern`.
