// driver.h - the compilation pipeline as a library call (used by main and unit tests).
#pragma once
#include <string>
#include "ast.h"
#include "ir.h"
#include "passes.h"
#include "ptx.h"

namespace kernc {

struct CompileOptions {
    int optLevel = 0;
    bool fma = false;
    bool workarounds = true;  // RPO layout + branch threading (avoid the ptxas crash)
    PtxOptions ptx;
};

struct Compilation {
    Program ast;
    Module ir;
    std::vector<PassStats> stats;
};

// lex + parse + typecheck + lower + optimise. Throws CompileError.
Compilation compileSource(const std::string& src, const CompileOptions& opt);

}  // namespace kernc
