// lower.cpp - AST -> three-address IR.
#include "lower.h"
#include <unordered_map>
#include "mutate.h"

namespace kernc {
namespace {

class Lowerer {
public:
    Function run(const Kernel& k) {
        f_.name = k.name;
        cur_ = newBlock("entry");
        scopes_.emplace_back();
        for (auto& p : k.params) {
            VReg r = f_.newReg(p.type);
            f_.params.push_back(r);
            f_.paramNames.push_back(p.name);
            scopes_.back()[p.name] = r;
        }
        block(k.body);
        if (!terminated()) emit({Op::Ret});
        return std::move(f_);
    }

private:
    Function f_;
    int cur_ = 0;
    int labelCounter_ = 0;
    std::vector<std::unordered_map<std::string, VReg>> scopes_;

    int newBlock(const std::string& base) {
        f_.blocks.push_back({base + (base == "entry" ? "" : std::to_string(labelCounter_++)), {}});
        return (int)f_.blocks.size() - 1;
    }
    bool terminated() const {
        auto& in = f_.blocks[cur_].instrs;
        return !in.empty() && isTerminator(in.back().op);
    }
    void emit(Instr in) { f_.blocks[cur_].instrs.push_back(std::move(in)); }
    void br(int target) { Instr i{Op::Br}; i.target = target; emit(i); }
    void condBr(VReg c, int t, int e) { Instr i{Op::CondBr}; i.srcs = {c}; i.target = t; i.target2 = e; emit(i); }
    void mov(VReg dst, VReg src) { Instr i{Op::Mov}; i.dst = dst; i.srcs = {src}; emit(i); }
    VReg constant(Type t, uint32_t bits) {
        VReg d = f_.newReg(t);
        Instr i{Op::Const}; i.dst = d; i.imm = bits; emit(i);
        return d;
    }
    VReg op(Op o, Type t, std::vector<VReg> srcs) {
        VReg d = f_.newReg(t);
        Instr i{o}; i.dst = d; i.srcs = std::move(srcs); emit(i);
        return d;
    }
    // &base[idx] as two pure ops, so CSE can share the offset between arrays
    // (x[i] and y[i]) and the whole address between a load and a store (y[i]).
    VReg address(VReg base, VReg idx) {
        VReg off = op(Op::ByteOff, Type::U64, {idx});
        return op(Op::ElemPtr, base.type, {base, off});
    }
    VReg lookup(const std::string& n) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) return f->second;
        }
        throw CompileError(0, 0, "internal: unresolved name '" + n + "' during lowering");
    }

    void block(const Block& b) {
        scopes_.emplace_back();
        for (auto& s : b) stmt(*s);
        scopes_.pop_back();
    }

    void stmt(const Stmt& s) {
        switch (s.kind) {
            case Stmt::Kind::Let: {
                auto& x = static_cast<const LetStmt&>(s);
                VReg v = expr(*x.init);
                VReg var = f_.newReg(v.type);   // the variable's own register
                mov(var, v);
                scopes_.back()[x.name] = var;
                break;
            }
            case Stmt::Kind::Assign: {
                auto& x = static_cast<const AssignStmt&>(s);
                VReg v = expr(*x.value);
                mov(lookup(x.name), v);
                break;
            }
            case Stmt::Kind::Store: {
                auto& x = static_cast<const StoreStmt&>(s);
                VReg addr = address(lookup(x.array), expr(*x.idx));
                VReg val = expr(*x.value);
                Instr i{Op::Store}; i.srcs = {addr, val}; emit(i);
                break;
            }
            case Stmt::Kind::If: {
                auto& x = static_cast<const IfStmt&>(s);
                VReg c = expr(*x.cond);
                int thenB = newBlock("then"), endB;
                if (x.elseB.empty()) {
                    endB = newBlock("endif");
                    condBr(c, thenB, endB);
                    cur_ = thenB; block(x.thenB); if (!terminated()) br(endB);
                } else {
                    int elseB = newBlock("else");
                    endB = newBlock("endif");
                    condBr(c, thenB, elseB);
                    cur_ = thenB; block(x.thenB); if (!terminated()) br(endB);
                    cur_ = elseB; block(x.elseB); if (!terminated()) br(endB);
                }
                cur_ = endB;
                break;
            }
            case Stmt::Kind::While: {
                auto& x = static_cast<const WhileStmt&>(s);
                int condB = newBlock("wcond"), bodyB = newBlock("wbody"), exitB = newBlock("wexit");
                br(condB);
                cur_ = condB;
                VReg c = expr(*x.cond);
                condBr(c, bodyB, exitB);
                cur_ = bodyB;
                block(x.body);
                if (!terminated()) br(mutant(10) ? exitB : condB);  // MUTANT 10: no back edge
                cur_ = exitB;
                break;
            }
            case Stmt::Kind::Return: {
                emit({Op::Ret});
                cur_ = newBlock("dead");  // anything after `return` is unreachable
                break;
            }
        }
    }

    VReg expr(const Expr& e) {
        switch (e.kind) {
            case Expr::Kind::IntLit: return constant(Type::I32, (uint32_t)static_cast<const IntLit&>(e).value);
            case Expr::Kind::FloatLit: return constant(Type::F32, f2bits(static_cast<const FloatLit&>(e).value));
            case Expr::Kind::BoolLit: return constant(Type::Bool, static_cast<const BoolLit&>(e).value ? 1 : 0);
            case Expr::Kind::Var: return lookup(static_cast<const VarRef&>(e).name);
            case Expr::Kind::Index: {
                auto& x = static_cast<const Index&>(e);
                VReg addr = address(lookup(x.array), expr(*x.idx));
                return op(Op::Load, e.type, {addr});
            }
            case Expr::Kind::Unary: {
                auto& u = static_cast<const Unary&>(e);
                VReg v = expr(*u.operand);
                if (u.op == UnOp::Not) return op(Op::Not, Type::Bool, {v});
                if (mutant(8) && v.type == Type::F32) {  // MUTANT 8: -x as 0.0 - x
                    VReg z = constant(Type::F32, f2bits(0.0f));
                    return op(Op::Sub, Type::F32, {z, v});
                }
                return op(Op::Neg, v.type, {v});
            }
            case Expr::Kind::Binary: return binary(static_cast<const Binary&>(e));
            case Expr::Kind::Call: {
                auto& c = static_cast<const Call&>(e);
                if (c.fn == "gid") return op(Op::Gid, Type::I32, {});
                if (c.fn == "tid") return op(Op::Tid, Type::I32, {});
                if (c.fn == "bid") return op(Op::Bid, Type::I32, {});
                if (c.fn == "bdim") return op(Op::Bdim, Type::I32, {});
                if (c.fn == "gdim") return op(Op::Gdim, Type::I32, {});
                VReg a = expr(*c.args[0]);
                if (c.fn == "f32") return a.type == Type::F32 ? a : op(Op::CvtI2F, Type::F32, {a});
                // i32(...)
                if (a.type == Type::I32) return a;
                if (a.type == Type::Bool) return op(Op::BoolToInt, Type::I32, {a});
                return op(Op::CvtF2I, Type::I32, {a});
            }
        }
        throw CompileError(e.line, e.col, "internal: unknown expression in lowering");
    }

    VReg binary(const Binary& b) {
        if (b.op == BinOp::And || b.op == BinOp::Or) return shortCircuit(b);
        VReg l = expr(*b.lhs), r = expr(*b.rhs);
        auto cmp = [&](Cmp c) {
            VReg d = f_.newReg(Type::Bool);
            Instr i{Op::Cmp}; i.dst = d; i.srcs = {l, r}; i.cmp = c; emit(i);
            return d;
        };
        switch (b.op) {
            case BinOp::Add: return op(Op::Add, l.type, {l, r});
            case BinOp::Sub: return op(Op::Sub, l.type, {l, r});
            case BinOp::Mul: return op(Op::Mul, l.type, {l, r});
            case BinOp::Div: return op(Op::Div, l.type, {l, r});
            case BinOp::Rem: return op(Op::Rem, l.type, {l, r});
            case BinOp::Lt: return cmp(Cmp::Lt);
            case BinOp::Le: return cmp(mutant(7) ? Cmp::Lt : Cmp::Le);  // MUTANT 7
            case BinOp::Gt: return cmp(Cmp::Gt);
            case BinOp::Ge: return cmp(Cmp::Ge);
            case BinOp::Eq: return cmp(Cmp::Eq);
            case BinOp::Ne: return cmp(Cmp::Ne);
            default: break;
        }
        throw CompileError(b.line, b.col, "internal: bad binary operator");
    }

    // a && b :  r = a; if (r) r = b;      a || b :  r = a; if (!r) r = b;
    VReg shortCircuit(const Binary& b) {
        bool isAnd = b.op == BinOp::And;
        VReg result = f_.newReg(Type::Bool);
        VReg l = expr(*b.lhs);
        mov(result, l);
        VReg eagerR{};
        if (mutant(9) && isAnd) eagerR = expr(*b.rhs);  // MUTANT 9: evaluate rhs unconditionally
        int rhsB = newBlock(isAnd ? "and_rhs" : "or_rhs"), endB = newBlock(isAnd ? "and_end" : "or_end");
        if (isAnd) condBr(result, rhsB, endB);
        else condBr(result, endB, rhsB);
        cur_ = rhsB;
        VReg r = eagerR.valid() ? eagerR : expr(*b.rhs);
        mov(result, r);
        br(endB);
        cur_ = endB;
        return result;
    }
};

}  // namespace

Module lowerProgram(const Program& prog) {
    Module m;
    for (auto& k : prog.kernels) m.functions.push_back(Lowerer().run(k));
    return m;
}

}  // namespace kernc
