// common.h - shared types, errors and value semantics for KernC.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace kernc {

// ---------------------------------------------------------------- types
enum class Type { I32, F32, Bool, PtrI32, PtrF32, U64, Void, Error };  // U64: IR-only byte offsets

inline bool isPtr(Type t) { return t == Type::PtrI32 || t == Type::PtrF32; }
inline Type elemOf(Type t) { return t == Type::PtrI32 ? Type::I32 : Type::F32; }
inline const char* typeName(Type t) {
    switch (t) {
        case Type::I32: return "i32";
        case Type::F32: return "f32";
        case Type::Bool: return "bool";
        case Type::PtrI32: return "i32*";
        case Type::PtrF32: return "f32*";
        case Type::U64: return "u64";
        case Type::Void: return "void";
        default: return "<error>";
    }
}

// ---------------------------------------------------------------- errors
struct CompileError : std::runtime_error {
    int line, col;
    CompileError(int l, int c, const std::string& m) : std::runtime_error(m), line(l), col(c) {}
};

// Raised by the interpreters when a program performs an operation the
// language spec leaves undefined (division by zero, out-of-bounds, ...).
struct UndefinedBehavior : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- bits
inline uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------- scalar semantics
// These functions ARE the language semantics from SPEC.md. Both interpreters
// and the constant folder use them, so they are written to be obviously
// correct rather than fast.
inline int32_t wrapAdd(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
inline int32_t wrapSub(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
inline int32_t wrapMul(int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }
inline int32_t wrapNeg(int32_t a) { return (int32_t)(0u - (uint32_t)a); }
inline bool divIsUB(int32_t a, int32_t b) { return b == 0 || (a == INT32_MIN && b == -1); }

// f32 -> i32: truncate toward zero, saturate out-of-range, NaN -> 0
// (matches PTX cvt.rzi.s32.f32).
inline int32_t cvtF2I(float f) {
    if (f != f) return 0;
    if (f >= 2147483648.0f) return INT32_MAX;
    if (f < -2147483648.0f) return INT32_MIN;
    return (int32_t)f;  // C++ conversion truncates toward zero
}

}  // namespace kernc
