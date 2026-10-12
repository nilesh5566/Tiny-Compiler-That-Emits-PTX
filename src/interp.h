// interp.h - two CPU reference executors.
//
//  * runAst - walks the type-checked AST directly. It shares no code with the
//             lowering, optimiser or code generator, so it is the independent
//             oracle for differential testing.
//  * runIr  - executes the (possibly optimised) IR. Comparing runIr at -O0
//             and -O1 against runAst isolates bugs in lowering vs. in passes,
//             and works on any machine - no GPU needed.
//
// Threads are executed one after another. That is a faithful model because
// Kern has no shared memory or barriers and data races are undefined (SPEC.md).
// Both throw UndefinedBehavior when the program does something undefined.
#pragma once
#include "ast.h"
#include "ir.h"
#include "launch.h"

namespace kernc {

void runAst(const Program& prog, Launch& launch);
uint64_t runIr(const Module& mod, Launch& launch);  // returns #instructions executed

}  // namespace kernc
