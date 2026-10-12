// typecheck.h - semantic analysis: scopes, name resolution and typing.
#pragma once
#include "ast.h"

namespace kernc {

// Annotates every Expr::type and throws CompileError on the first error.
void typecheck(Program& prog);

// Builtins shared by the checker, lowering and interpreters.
bool isThreadBuiltin(const std::string& fn);  // gid, tid, bid, bdim, gdim

}  // namespace kernc
