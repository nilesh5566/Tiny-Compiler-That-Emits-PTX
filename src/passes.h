// passes.h - IR optimisation passes.
#pragma once
#include <string>
#include <vector>
#include "ir.h"

namespace kernc {

struct PassStats { std::string name; size_t before, after; };

bool constFold(Function& f);        // fold ops with constant operands, fold constant branches
bool copyPropagate(Function& f);    // replace uses of `a = mov b` with b (block-local)
bool globalCopyProp(Function& f);   // same, across blocks, when a and b each have one definition
bool cse(Function& f);              // common subexpression elimination (block-local)
bool strengthReduce(Function& f);   // i32 x * 2^k -> x << k, x * 1 -> x
bool deadCodeElim(Function& f);     // drop pure instructions whose result is never used
bool licm(Function& f);             // hoist loop-invariant, non-trapping computations to the preheader
bool fmaContract(Function& f);      // f32 a*b + c -> fma(a,b,c)  (changes rounding!)

// Reorders blocks into reverse post-order (every non-loop edge points forward).
// Always run before code generation, at every -O level.
void layoutBlocks(Function& f);

// Runs the -O1 pipeline. If `fma` is set, contraction runs at the end.
std::vector<PassStats> optimize(Function& f, int level, bool fma);

}  // namespace kernc
