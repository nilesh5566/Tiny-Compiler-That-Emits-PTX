// ptx.h - PTX code generation from IR.
#pragma once
#include <string>
#include "ir.h"

namespace kernc {

struct PtxOptions {
    std::string arch = "sm_75";       // T4 / RTX 20xx. Use sm_86 for RTX 30xx, sm_89 for RTX 40xx.
    std::string version;              // PTX ISA version; empty = lowest version that supports `arch`
    bool threadBranches = true;       // skip jump-only blocks (also part of the ptxas-crash workaround)
};

std::string emitPtx(const Module& m, const PtxOptions& opt);

}  // namespace kernc
