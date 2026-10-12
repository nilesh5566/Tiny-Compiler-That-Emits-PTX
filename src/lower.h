// lower.h - AST -> IR lowering.
#pragma once
#include "ast.h"
#include "ir.h"

namespace kernc {

Module lowerProgram(const Program& prog);  // program must be type-checked

}  // namespace kernc
