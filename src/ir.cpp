// ir.cpp - IR printing helpers.
#include "ir.h"
#include <cstdio>
#include <sstream>

namespace kernc {

const char* opName(Op op) {
    switch (op) {
        case Op::Const: return "const"; case Op::Mov: return "mov"; case Op::Add: return "add";
        case Op::Sub: return "sub"; case Op::Mul: return "mul"; case Op::Div: return "div";
        case Op::Rem: return "rem"; case Op::Shl: return "shl"; case Op::Neg: return "neg";
        case Op::Fma: return "fma"; case Op::Cmp: return "cmp"; case Op::Not: return "not";
        case Op::BoolToInt: return "bool2int"; case Op::CvtI2F: return "cvt.i2f";
        case Op::CvtF2I: return "cvt.f2i"; case Op::ByteOff: return "byteoff";
        case Op::ElemPtr: return "elemptr"; case Op::Load: return "load"; case Op::Store: return "store";
        case Op::Gid: return "gid"; case Op::Tid: return "tid"; case Op::Bid: return "bid";
        case Op::Bdim: return "bdim"; case Op::Gdim: return "gdim"; case Op::Br: return "br";
        case Op::CondBr: return "condbr"; case Op::Ret: return "ret";
    }
    return "?";
}

const char* cmpName(Cmp c) {
    switch (c) {
        case Cmp::Lt: return "lt"; case Cmp::Le: return "le"; case Cmp::Gt: return "gt";
        case Cmp::Ge: return "ge"; case Cmp::Eq: return "eq"; case Cmp::Ne: return "ne";
    }
    return "?";
}

static std::string reg(const VReg& r) { return "v" + std::to_string(r.id); }

static std::string constText(const Instr& in) {
    char buf[64];
    switch (in.dst.type) {
        case Type::I32: return std::to_string((int32_t)in.imm);
        case Type::Bool: return in.imm ? "true" : "false";
        default:
            std::snprintf(buf, sizeof buf, "%.9g (0x%08X)", bits2f(in.imm), in.imm);
            return buf;
    }
}

std::string dumpIr(const Module& m) {
    std::ostringstream o;
    for (auto& f : m.functions) {
        o << f.name << "(";
        for (size_t i = 0; i < f.params.size(); i++)
            o << (i ? ", " : "") << f.paramNames[i] << ": " << typeName(f.params[i].type) << " = " << reg(f.params[i]);
        o << ")\n";
        for (auto& b : f.blocks) {
            o << b.label << ":\n";
            for (auto& in : b.instrs) {
                o << "  ";
                if (in.dst.valid()) o << reg(in.dst) << ":" << typeName(in.dst.type) << " = ";
                o << opName(in.op);
                if (in.op == Op::Cmp) o << "." << cmpName(in.cmp);
                switch (in.op) {
                    case Op::Const: o << " " << constText(in); break;
                    case Op::Load: o << " [" << reg(in.srcs[0]) << "]"; break;
                    case Op::Store: o << " [" << reg(in.srcs[0]) << "], " << reg(in.srcs[1]); break;
                    case Op::Br: o << " " << f.blocks[in.target].label; break;
                    case Op::CondBr:
                        o << " " << reg(in.srcs[0]) << ", " << f.blocks[in.target].label << ", " << f.blocks[in.target2].label;
                        break;
                    case Op::Shl: o << " " << reg(in.srcs[0]) << ", " << in.imm; break;
                    default:
                        for (size_t i = 0; i < in.srcs.size(); i++) o << (i ? ", " : " ") << reg(in.srcs[i]);
                }
                o << "\n";
            }
        }
        o << "\n";
    }
    return o.str();
}

size_t countInstrs(const Function& f) {
    size_t n = 0;
    for (auto& b : f.blocks) n += b.instrs.size();
    return n;
}

}  // namespace kernc
