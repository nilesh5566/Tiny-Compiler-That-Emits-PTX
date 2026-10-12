// semantics.h - evaluation of one pure IR operation on raw 32-bit values.
//
// Used by the IR interpreter and the constant folder. (The AST interpreter
// deliberately does NOT use this: it is the independent oracle.)
#pragma once
#include "ir.h"

namespace kernc {

// Values are raw bits: i32 as two's complement, f32 as IEEE bits, bool as 0/1.
// Throws UndefinedBehavior for i32 division by zero and INT_MIN / -1.
// Only valid for arithmetic / compare / convert ops (not Load, Store, geometry).
uint32_t evalPure(const Instr& in, const uint32_t* src);

}  // namespace kernc
