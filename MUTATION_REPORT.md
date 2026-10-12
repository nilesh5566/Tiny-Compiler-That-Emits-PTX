# Mutation testing report

Mutation score: **16/16 (100%)**  
Fuzz budget per surviving mutant: 200 programs; GPU: no

| # | mutant | injected bug | result | killed by |
|---|---|---|---|---|
| 1 | `constfold-sub-swapped` | constant folder computes b-a for a-b | KILLED | tests (unit) |
| 2 | `constfold-div-floor` | constant folder rounds i32 division toward -inf | KILLED | tests (unit) |
| 3 | `copyprop-stale-source` | copy propagation ignores redefinition of the copy source | KILLED | tests (unit) |
| 4 | `cse-stale-operands` | CSE reuses a value after one of its operands changed | KILLED | tests (unit) |
| 5 | `strength-neg-pow2` | strength reduction turns x*(-2^k) into x<<k | KILLED | tests (unit) |
| 6 | `dce-ignores-branch-uses` | DCE forgets that branch conditions are uses | KILLED | tests (codegen) |
| 7 | `lower-le-as-lt` | `<=` is lowered as `<` | KILLED | tests (exec) |
| 8 | `lower-fneg-as-sub` | f32 negation lowered as 0.0 - x (wrong for -0.0) | KILLED | tests (unit) |
| 9 | `lower-and-eager` | `&&` evaluates its right side even when the left is false | KILLED | tests (codegen) |
| 10 | `lower-while-as-if` | while loop body runs at most once | KILLED | tests (unit) |
| 11 | `constfold-cvt-rounds` | constant folder rounds f32->i32 instead of truncating | KILLED | tests (exec) |
| 12 | `ptx-fne-ordered` | f32 `!=` emitted as ordered setp.ne (false for NaN) | KILLED | tests (codegen) |
| 13 | `ptx-index-scale-8` | array index scaled by 8 bytes instead of 4 | KILLED | tests (codegen) |
| 14 | `ptx-missing-cvta` | pointer params used without cvta.to.global | KILLED | tests (unit) |
| 15 | `typecheck-mixed-arith` | type checker accepts i32 + f32 | KILLED | tests (unit) |
| 16 | `licm-hoists-division` | LICM hoists i32 division out of loops (may run zero times) | KILLED | tests (unit) |

## First failure seen for each killed mutant

- **#1 constfold-sub-swapped**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_constfold_and_dce;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:185: ir.find("const 13") != std::string::npos
- **#2 constfold-div-floor**: unit/kernc_unit_tests: unit tests failed: [FAIL] semantics_int_wrap_and_div;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:112: (int32_t)l.args[0].buf[2] == -3
- **#3 copyprop-stale-source**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_copyprop_isolated;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:252: store.srcs[2].id != a.id
- **#4 cse-stale-operands**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_cse_isolated;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:267: f.blocks[0].instrs[3].op == Op::Mul
- **#5 strength-neg-pow2**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_strength_reduction;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:205: ir.find("mul") != std::string::npos
- **#6 dce-ignores-branch-uses**: codegen/saxpy_o1.kern: CHECK: 'setp.lt.s32' not found after line 33
- **#7 lower-le-as-lt**: exec/le_vs_lt.kern: [ir -O0] differs from AST oracle: arg1[3]: expected 2 got 1
- **#8 lower-fneg-as-sub**: unit/kernc_unit_tests: unit tests failed: [FAIL] semantics_negative_zero;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:132: l.args[0].buf[0] == 0x80000000u;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:133: std::isinf(bits2f(l.args[0].buf[1])) && bits2f(l.args[0].buf[1]
- **#9 lower-and-eager**: codegen/short_circuit.kern: CHECK-NOT: 'ld.global' matched line 32: ld.global.u32 %r3, [%rd6];
- **#10 lower-while-as-if**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_licm;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:277: ir.find("byteoff") < body
- **#11 constfold-cvt-rounds**: exec/const_fold_semantics.kern: [ir -O1] differs from AST oracle: arg0[3]: expected 3 got 4; arg0[4]: expected -3 got -4
- **#12 ptx-fne-ordered**: codegen/float_ne_unordered.kern: CHECK: 'setp.neu.f32' not found after line 1
- **#13 ptx-index-scale-8**: codegen/saxpy_o1.kern: CHECK: 'mul.wide.s32 {{%rd[0-9]+}}, {{%r[0-9]+}}, 4' not found after line 34
- **#14 ptx-missing-cvta**: unit/kernc_unit_tests: unit tests failed: [FAIL] ptx_basics;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:287: p.find("cvta.to.global.u64") != std::string::npos
- **#15 typecheck-mixed-arith**: unit/kernc_unit_tests: unit tests failed: [FAIL] typecheck_errors;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:86: compileError("kernel k(i32* o) { o[0] = 1 + 2.0; }").find("cannot apply '+'") != std::string::npos
- **#16 licm-hoists-division**: unit/kernc_unit_tests: unit tests failed: [FAIL] pass_licm;   FAILED D:\Downloads\kernc\kernc\tests\unit\unit_tests.cpp:279: ir.find("div") > body
