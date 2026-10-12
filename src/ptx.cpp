// ptx.cpp - IR -> PTX.
//
// Register classes:  i32 -> %r (.b32)   f32 -> %f (.f32)   bool -> %p (.pred)
//                    pointers -> %rd (.b64)
// Float add/sub/mul carry an explicit .rn modifier: that stops ptxas from
// fusing them into FMA, so the generated code keeps the exact IEEE semantics
// promised in SPEC.md. FMA appears only when the --fma pass asked for it.
#include "ptx.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "mutate.h"

namespace kernc {
namespace {

class Emitter {
public:
    Emitter(const Function& f, bool thread) : f_(f), thread_(thread) {}

    std::string run() {
        // which blocks need labels?
        for (auto& b : f_.blocks)
            for (auto& in : b.instrs) {
                if (in.target >= 0) targeted_.insert(resolve(in.target));
                if (in.target2 >= 0) targeted_.insert(resolve(in.target2));
            }
        prologue();
        for (size_t bi = 0; bi < f_.blocks.size(); bi++) {
            curBlock_ = (int)bi;
            if (targeted_.count((int)bi)) body_ << label((int)bi) << ":\n";
            for (auto& in : f_.blocks[bi].instrs) instr(in);
        }
        body_ << "$L__exit:\n";
        line("ret");
        return header() + body_.str() + "}\n";
    }

private:
    const Function& f_;
    bool thread_;
    std::ostringstream body_;
    std::unordered_map<int, std::string> names_;
    std::unordered_set<int> targeted_;
    int curBlock_ = 0;
    int nR_ = 1, nF_ = 1, nP_ = 1, nRD_ = 1;

    std::string label(int b) const { return "$L_" + f_.blocks[b].label; }

    // Branch threading: a block that only contains `br T` is skipped, so branches
    // go straight to T and a conditional branch whose two targets coincide becomes
    // an unconditional one. Besides producing tidier code this avoids a ptxas
    // crash found by fuzzing (docs/findings/ptxas-crash): ptxas 12.9 segfaults at
    // -O1+ on `@p bra EMPTY; bra.uni NEXT;` shapes where EMPTY just falls into NEXT.
    int resolve(int b) const {
        for (int hops = 0; thread_ && hops < 64; hops++) {
            const auto& in = f_.blocks[b].instrs;
            if (in.size() != 1 || in[0].op != Op::Br || in[0].target == b) return b;
            b = in[0].target;
        }
        return b;
    }

    std::string fresh(Type t) {
        switch (t) {
            case Type::I32: return "%r" + std::to_string(nR_++);
            case Type::F32: return "%f" + std::to_string(nF_++);
            case Type::Bool: return "%p" + std::to_string(nP_++);
            default: return "%rd" + std::to_string(nRD_++);
        }
    }
    const std::string& reg(const VReg& v) {
        auto it = names_.find(v.id);
        if (it != names_.end()) return it->second;
        return names_[v.id] = fresh(v.type);
    }

    void line(const std::string& s) { body_ << "    " << s << ";\n"; }

    std::string paramName(size_t i) const { return f_.name + "_param_" + std::to_string(i); }

    std::string header() {
        std::ostringstream h;
        h << "\n.visible .entry " << f_.name << "(";
        for (size_t i = 0; i < f_.params.size(); i++) {
            Type t = f_.params[i].type;
            const char* pt = isPtr(t) ? ".u64" : t == Type::F32 ? ".f32" : ".u32";
            h << (i ? ",\n    " : "\n    ") << ".param " << pt << " " << paramName(i);
        }
        h << (f_.params.empty() ? ")\n{\n" : "\n)\n{\n");
        h << "    // kernel parameters: ";
        for (size_t i = 0; i < f_.params.size(); i++)
            h << (i ? ", " : "") << typeName(f_.params[i].type) << " " << f_.paramNames[i];
        h << "\n";
        h << "    .reg .pred  %p<" << nP_ << ">;\n";
        h << "    .reg .b32   %r<" << nR_ << ">;\n";
        h << "    .reg .f32   %f<" << nF_ << ">;\n";
        h << "    .reg .b64   %rd<" << nRD_ << ">;\n\n";
        return h.str();
    }

    void prologue() {
        for (size_t i = 0; i < f_.params.size(); i++) {
            const VReg& p = f_.params[i];
            if (isPtr(p.type)) {
                std::string raw = fresh(Type::PtrF32);
                line("ld.param.u64 " + raw + ", [" + paramName(i) + "]");
                if (mutant(14)) { names_[p.id] = raw; continue; }  // MUTANT 14: generic address used as global
                line("cvta.to.global.u64 " + reg(p) + ", " + raw);
            } else if (p.type == Type::F32) {
                line("ld.param.f32 " + reg(p) + ", [" + paramName(i) + "]");
            } else {
                line("ld.param.u32 " + reg(p) + ", [" + paramName(i) + "]");
            }
        }
        body_ << "\n";
    }

    static const char* movSuffix(Type t) {
        switch (t) {
            case Type::I32: return "u32"; case Type::F32: return "f32"; case Type::Bool: return "pred";
            default: return "u64";
        }
    }
    static const char* cc(Cmp c) {
        switch (c) {
            case Cmp::Lt: return "lt"; case Cmp::Le: return "le"; case Cmp::Gt: return "gt";
            case Cmp::Ge: return "ge"; case Cmp::Eq: return "eq"; case Cmp::Ne: return "ne";
        }
        return "?";
    }

    void arith(const Instr& in, const char* intOp, const char* fltOp) {
        bool f = in.dst.type == Type::F32;
        std::string d = reg(in.dst), a = reg(in.srcs[0]), b = reg(in.srcs[1]);
        line(std::string(f ? fltOp : intOp) + " " + d + ", " + a + ", " + b);
    }

    void instr(const Instr& in) {
        switch (in.op) {
            case Op::Const: {
                char buf[32];
                std::string d = reg(in.dst);
                if (in.dst.type == Type::F32) {
                    std::snprintf(buf, sizeof buf, "0f%08X", in.imm);
                    line("mov.f32 " + d + ", " + buf);
                } else if (in.dst.type == Type::I32) {
                    std::snprintf(buf, sizeof buf, "0x%08X", in.imm);
                    line("mov.u32 " + d + ", " + buf);
                } else {
                    std::string t = fresh(Type::I32);
                    line("mov.u32 " + t + ", " + std::to_string(in.imm ? 1 : 0));
                    line("setp.ne.u32 " + d + ", " + t + ", 0");
                }
                break;
            }
            case Op::Mov: {
                std::string d = reg(in.dst), s = reg(in.srcs[0]);
                line(std::string("mov.") + movSuffix(in.dst.type) + " " + d + ", " + s);
                break;
            }
            case Op::Add: arith(in, "add.s32", "add.rn.f32"); break;
            case Op::Sub: arith(in, "sub.s32", "sub.rn.f32"); break;
            case Op::Mul: arith(in, "mul.lo.s32", "mul.rn.f32"); break;
            case Op::Div: arith(in, "div.s32", "div.rn.f32"); break;
            case Op::Rem: arith(in, "rem.s32", "rem.s32"); break;
            case Op::Shl: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line("shl.b32 " + d + ", " + a + ", " + std::to_string(in.imm));
                break;
            }
            case Op::Neg: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line(std::string(in.dst.type == Type::F32 ? "neg.f32 " : "neg.s32 ") + d + ", " + a);
                break;
            }
            case Op::Fma: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]), b = reg(in.srcs[1]), c = reg(in.srcs[2]);
                line("fma.rn.f32 " + d + ", " + a + ", " + b + ", " + c);
                break;
            }
            case Op::Cmp: {
                Type t = in.srcs[0].type;
                std::string d = reg(in.dst), a = reg(in.srcs[0]), b = reg(in.srcs[1]);
                if (t == Type::Bool) {
                    line("xor.pred " + d + ", " + a + ", " + b);
                    if (in.cmp == Cmp::Eq) line("not.pred " + d + ", " + d);
                } else if (t == Type::F32) {
                    // `!=` must be true when either side is NaN -> unordered "neu".
                    const char* c = (in.cmp == Cmp::Ne && !mutant(12)) ? "neu" : cc(in.cmp);  // MUTANT 12
                    line(std::string("setp.") + c + ".f32 " + d + ", " + a + ", " + b);
                } else {
                    line(std::string("setp.") + cc(in.cmp) + ".s32 " + d + ", " + a + ", " + b);
                }
                break;
            }
            case Op::Not: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line("not.pred " + d + ", " + a);
                break;
            }
            case Op::BoolToInt: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line("selp.u32 " + d + ", 1, 0, " + a);
                break;
            }
            case Op::CvtI2F: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line("cvt.rn.f32.s32 " + d + ", " + a);
                break;
            }
            case Op::CvtF2I: {  // truncates, saturates, NaN -> 0
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line("cvt.rzi.s32.f32 " + d + ", " + a);
                break;
            }
            case Op::ByteOff: {  // 64-bit byte offset = sign-extended index * 4
                std::string d = reg(in.dst), i = reg(in.srcs[0]);
                line("mul.wide.s32 " + d + ", " + i + (mutant(13) ? ", 8" : ", 4"));  // MUTANT 13
                break;
            }
            case Op::ElemPtr: {
                std::string d = reg(in.dst), b = reg(in.srcs[0]), o = reg(in.srcs[1]);
                line("add.s64 " + d + ", " + b + ", " + o);
                break;
            }
            case Op::Load: {
                std::string d = reg(in.dst), a = reg(in.srcs[0]);
                line(std::string(in.dst.type == Type::F32 ? "ld.global.f32 " : "ld.global.u32 ") + d + ", [" + a + "]");
                break;
            }
            case Op::Store: {
                std::string a = reg(in.srcs[0]), v = reg(in.srcs[1]);
                line(std::string(in.srcs[1].type == Type::F32 ? "st.global.f32 [" : "st.global.u32 [") + a + "], " + v);
                break;
            }
            case Op::Gid: {
                std::string a = fresh(Type::I32), b = fresh(Type::I32), c = fresh(Type::I32);
                line("mov.u32 " + a + ", %ctaid.x");
                line("mov.u32 " + b + ", %ntid.x");
                line("mov.u32 " + c + ", %tid.x");
                std::string d = reg(in.dst);
                line("mad.lo.s32 " + d + ", " + a + ", " + b + ", " + c);
                break;
            }
            case Op::Tid: { std::string d = reg(in.dst); line("mov.u32 " + d + ", %tid.x"); break; }
            case Op::Bid: { std::string d = reg(in.dst); line("mov.u32 " + d + ", %ctaid.x"); break; }
            case Op::Bdim: { std::string d = reg(in.dst); line("mov.u32 " + d + ", %ntid.x"); break; }
            case Op::Gdim: { std::string d = reg(in.dst); line("mov.u32 " + d + ", %nctaid.x"); break; }
            case Op::Br: {
                int t = resolve(in.target);
                if (t != curBlock_ + 1) line("bra.uni " + label(t));
                break;
            }
            case Op::CondBr: {
                std::string p = reg(in.srcs[0]);
                int t = resolve(in.target), e = resolve(in.target2);
                if (t == e) {  // both ways lead to the same place
                    if (t != curBlock_ + 1) line("bra.uni " + label(t));
                } else if (t == curBlock_ + 1) {  // fall into the true block, branch on !p
                    line("@!" + p + " bra " + label(e));
                } else {
                    line("@" + p + " bra " + label(t));
                    if (e != curBlock_ + 1) line("bra.uni " + label(e));
                }
                break;
            }
            case Op::Ret:
                // Single-exit normalisation: every `return` jumps to one shared exit block.
                // Workaround for a ptxas crash found by fuzzing (docs/findings/ptxas-crash):
                // ptxas 12.9 segfaults at -O1+ on some CFGs with a `ret` mid-kernel.
                if (curBlock_ + 1 != (int)f_.blocks.size()) line("bra.uni $L__exit");
                break;
        }
    }
};

}  // namespace

// Lowest PTX ISA version that can target a given sm_XX (older = accepted by more drivers).
static std::string ptxVersionFor(const std::string& arch) {
    int sm = 0;
    if (arch.rfind("sm_", 0) == 0) sm = std::atoi(arch.c_str() + 3);
    if (sm >= 100) return "8.6";
    if (sm >= 89) return "7.8";   // sm_89 (Ada), sm_90 (Hopper)
    if (sm >= 87) return "7.4";
    if (sm >= 86) return "7.1";   // Ampere consumer cards
    return "7.0";                 // sm_50 .. sm_80
}

std::string emitPtx(const Module& m, const PtxOptions& opt) {
    std::ostringstream o;
    o << "//\n// Generated by kernc (KernC compiler for the Kern GPU kernel language)\n//\n\n";
    o << ".version " << (opt.version.empty() ? ptxVersionFor(opt.arch) : opt.version) << "\n";
    o << ".target " << opt.arch << "\n";
    o << ".address_size 64\n";
    for (auto& f : m.functions) o << Emitter(f, opt.threadBranches).run();
    return o.str();
}

}  // namespace kernc
