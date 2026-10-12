// mutate.h - deliberate compiler bugs for mutation testing.
//
// Setting the environment variable KERNC_MUTATE=<id> switches ON exactly one
// injected bug. The verification suite is then run against the broken
// compiler; a good suite "kills" (detects) the mutant. See fuzz/mutate.py.
// With KERNC_MUTATE unset, every mutant() call returns false.
#pragma once
#include <cstdlib>

namespace kernc {

struct MutantInfo { int id; const char* name; const char* description; };

inline const MutantInfo kMutants[] = {
    {1,  "constfold-sub-swapped",   "constant folder computes b-a for a-b"},
    {2,  "constfold-div-floor",     "constant folder rounds i32 division toward -inf"},
    {3,  "copyprop-stale-source",   "copy propagation ignores redefinition of the copy source"},
    {4,  "cse-stale-operands",      "CSE reuses a value after one of its operands changed"},
    {5,  "strength-neg-pow2",       "strength reduction turns x*(-2^k) into x<<k"},
    {6,  "dce-ignores-branch-uses", "DCE forgets that branch conditions are uses"},
    {7,  "lower-le-as-lt",          "`<=` is lowered as `<`"},
    {8,  "lower-fneg-as-sub",       "f32 negation lowered as 0.0 - x (wrong for -0.0)"},
    {9,  "lower-and-eager",         "`&&` evaluates its right side even when the left is false"},
    {10, "lower-while-as-if",       "while loop body runs at most once"},
    {11, "constfold-cvt-rounds",    "constant folder rounds f32->i32 instead of truncating"},
    {12, "ptx-fne-ordered",         "f32 `!=` emitted as ordered setp.ne (false for NaN)"},
    {13, "ptx-index-scale-8",       "array index scaled by 8 bytes instead of 4"},
    {14, "ptx-missing-cvta",        "pointer params used without cvta.to.global"},
    {15, "typecheck-mixed-arith",   "type checker accepts i32 + f32"},
    {16, "licm-hoists-division",    "LICM hoists i32 division out of loops (may run zero times)"},
};
inline constexpr int kNumMutants = sizeof(kMutants) / sizeof(kMutants[0]);

inline int activeMutant() {
    static int id = [] {
        const char* s = std::getenv("KERNC_MUTATE");
        return s ? std::atoi(s) : 0;
    }();
    return id;
}
inline bool mutant(int id) { return activeMutant() == id; }

}  // namespace kernc
