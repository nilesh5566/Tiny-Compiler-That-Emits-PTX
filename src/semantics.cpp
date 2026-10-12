// semantics.cpp
#include "semantics.h"
#include <cmath>

namespace kernc {

static bool compare(Cmp c, Type t, uint32_t a, uint32_t b) {
    if (t == Type::F32) {
        float x = bits2f(a), y = bits2f(b);
        switch (c) {
            case Cmp::Lt: return x < y; case Cmp::Le: return x <= y;
            case Cmp::Gt: return x > y; case Cmp::Ge: return x >= y;
            case Cmp::Eq: return x == y; case Cmp::Ne: return x != y;  // NaN != NaN is true
        }
    }
    if (t == Type::Bool) return c == Cmp::Eq ? a == b : a != b;
    int32_t x = (int32_t)a, y = (int32_t)b;
    switch (c) {
        case Cmp::Lt: return x < y; case Cmp::Le: return x <= y;
        case Cmp::Gt: return x > y; case Cmp::Ge: return x >= y;
        case Cmp::Eq: return x == y; case Cmp::Ne: return x != y;
    }
    return false;
}

uint32_t evalPure(const Instr& in, const uint32_t* s) {
    Type t = in.dst.type;
    bool f = (t == Type::F32);
    auto F = [](uint32_t u) { return bits2f(u); };
    auto I = [](uint32_t u) { return (int32_t)u; };
    switch (in.op) {
        case Op::Const: return in.imm;
        case Op::Mov: return s[0];
        case Op::Add: return f ? f2bits(F(s[0]) + F(s[1])) : (uint32_t)wrapAdd(I(s[0]), I(s[1]));
        case Op::Sub: return f ? f2bits(F(s[0]) - F(s[1])) : (uint32_t)wrapSub(I(s[0]), I(s[1]));
        case Op::Mul: return f ? f2bits(F(s[0]) * F(s[1])) : (uint32_t)wrapMul(I(s[0]), I(s[1]));
        case Op::Div:
            if (f) return f2bits(F(s[0]) / F(s[1]));
            if (divIsUB(I(s[0]), I(s[1]))) throw UndefinedBehavior("i32 division by zero or INT_MIN / -1");
            return (uint32_t)(I(s[0]) / I(s[1]));
        case Op::Rem:
            if (divIsUB(I(s[0]), I(s[1]))) throw UndefinedBehavior("i32 remainder by zero or INT_MIN % -1");
            return (uint32_t)(I(s[0]) % I(s[1]));
        case Op::Shl: return s[0] << (in.imm & 31);
        case Op::Neg: return f ? (s[0] ^ 0x80000000u) : (uint32_t)wrapNeg(I(s[0]));  // f32 neg flips sign bit
        case Op::Fma: return f2bits(std::fmaf(F(s[0]), F(s[1]), F(s[2])));
        case Op::Cmp: return compare(in.cmp, in.srcs[0].type, s[0], s[1]) ? 1u : 0u;
        case Op::Not: return s[0] ? 0u : 1u;
        case Op::BoolToInt: return s[0] ? 1u : 0u;
        case Op::CvtI2F: return f2bits((float)I(s[0]));  // round to nearest even
        case Op::CvtF2I: return (uint32_t)cvtF2I(F(s[0]));
        default: break;
    }
    throw std::logic_error(std::string("evalPure: not a pure op: ") + opName(in.op));
}

}  // namespace kernc
