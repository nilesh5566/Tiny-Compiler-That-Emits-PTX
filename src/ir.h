// ir.h - three-address intermediate representation.
//
// Virtual registers are typed and NOT in SSA form: a source variable owns one
// register for its whole lifetime and assignment is a Mov into it. This keeps
// loops simple (no phi nodes); every optimisation pass therefore has to cope
// with registers that are defined more than once.
#pragma once
#include <string>
#include <vector>
#include "common.h"

namespace kernc {

enum class Op {
    Const, Mov,
    Add, Sub, Mul, Div, Rem, Shl, Neg, Fma,  // arithmetic (Shl/Fma only from passes)
    Cmp, Not, BoolToInt, CvtI2F, CvtF2I,
    ByteOff, ElemPtr, Load, Store,           // memory: off = idx*4 ; addr = base + off ; ld/st [addr]
    Gid, Tid, Bid, Bdim, Gdim,               // thread geometry
    Br, CondBr, Ret                          // terminators
};
enum class Cmp { Lt, Le, Gt, Ge, Eq, Ne };

struct VReg {
    int id = -1;
    Type type = Type::Void;
    bool valid() const { return id >= 0; }
};

struct Instr {
    Instr(Op o = Op::Ret) : op(o) {}  // NOLINT: implicit so `emit({Op::Ret})` reads naturally
    Op op;
    VReg dst;                  // invalid for Store / terminators
    std::vector<VReg> srcs;    // ByteOff: {idx}; ElemPtr: {base, off}; Load: {addr}; Store: {addr, value}; CondBr: {cond}
    uint32_t imm = 0;          // Const: raw bits (i32, f32 bits, or 0/1); Shl: shift amount
    Cmp cmp = Cmp::Eq;         // Cmp only
    int target = -1, target2 = -1;  // block indices for Br / CondBr (true, false)
};

struct BasicBlock {
    std::string label;
    std::vector<Instr> instrs;
};

struct Function {
    std::string name;
    std::vector<VReg> params;           // one register per kernel parameter
    std::vector<std::string> paramNames;
    std::vector<BasicBlock> blocks;     // blocks[0] is the entry
    std::vector<Type> regTypes;         // indexed by VReg::id

    VReg newReg(Type t) {
        VReg r{(int)regTypes.size(), t};
        regTypes.push_back(t);
        return r;
    }
};

struct Module { std::vector<Function> functions; };

inline bool isTerminator(Op op) { return op == Op::Br || op == Op::CondBr || op == Op::Ret; }
// Instructions without side effects: removable if their result is unused.
inline bool isPure(Op op) { return !(op == Op::Store || isTerminator(op)); }

const char* opName(Op op);
const char* cmpName(Cmp c);
std::string dumpIr(const Module& m);
size_t countInstrs(const Function& f);

}  // namespace kernc
