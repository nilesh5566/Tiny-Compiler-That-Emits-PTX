// interp.cpp - AST interpreter (oracle) and IR interpreter.
#include "interp.h"
#include <algorithm>
#include <unordered_map>
#include "semantics.h"

namespace kernc {

static const uint64_t kStepLimit = 200'000'000;

static void checkArgs(const std::vector<Type>& params, const Launch& l, const std::string& name) {
    if (params.size() != l.args.size())
        throw std::runtime_error("kernel '" + name + "' takes " + std::to_string(params.size()) +
                                 " arguments but the launch file provides " + std::to_string(l.args.size()));
    for (size_t i = 0; i < params.size(); i++)
        if (params[i] != l.args[i].type)
            throw std::runtime_error("argument " + std::to_string(i) + " of '" + name + "' should be " +
                                     typeName(params[i]) + " but launch file gives " + typeName(l.args[i].type));
}

static uint32_t& element(Launch& l, int arg, int32_t idx) {
    if (arg < 0 || (size_t)arg >= l.args.size()) throw std::logic_error("interpreter: register is not a pointer");
    auto& buf = l.args[arg].buf;
    if (idx < 0 || (size_t)idx >= buf.size())
        throw UndefinedBehavior("out-of-bounds access: index " + std::to_string(idx) + " into buffer of " +
                                std::to_string(buf.size()) + " elements (argument " + std::to_string(arg) + ")");
    return buf[idx];
}

// ============================================================ AST interpreter
namespace {

struct Val {
    Type type = Type::Error;
    uint32_t bits = 0;  // i32 / f32 bits / bool 0-1
    int arg = -1;       // pointer: index into launch args
    int32_t i() const { return (int32_t)bits; }
    float f() const { return bits2f(bits); }
};
Val mkI(int32_t v) { return {Type::I32, (uint32_t)v, -1}; }
Val mkF(float v) { return {Type::F32, f2bits(v), -1}; }
Val mkB(bool v) { return {Type::Bool, v ? 1u : 0u, -1}; }

struct ThreadCtx { int tid, bid, bdim, gdim; };

class AstInterp {
public:
    AstInterp(const Kernel& k, Launch& l) : k_(k), l_(l) {}

    void run() {
        for (int b = 0; b < l_.grid; b++)
            for (int t = 0; t < l_.block; t++) {
                ctx_ = {t, b, l_.block, l_.grid};
                env_.clear();
                env_.emplace_back();
                for (size_t i = 0; i < k_.params.size(); i++) {
                    Val v{k_.params[i].type, l_.args[i].scalar, isPtr(k_.params[i].type) ? (int)i : -1};
                    env_.back()[k_.params[i].name] = v;
                }
                block(k_.body);
            }
    }

private:
    const Kernel& k_;
    Launch& l_;
    ThreadCtx ctx_{};
    std::vector<std::unordered_map<std::string, Val>> env_;
    uint64_t steps_ = 0;

    void tick() {
        if (++steps_ > kStepLimit) throw UndefinedBehavior("step limit exceeded (non-terminating loop?)");
    }
    Val& var(const std::string& n) {
        for (auto it = env_.rbegin(); it != env_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) return f->second;
        }
        throw std::logic_error("AST interpreter: unknown variable " + n);
    }

    enum class Flow { Next, Return };

    Flow block(const Block& b) {
        env_.emplace_back();
        Flow r = Flow::Next;
        for (auto& s : b) {
            if (stmt(*s) == Flow::Return) { r = Flow::Return; break; }
        }
        env_.pop_back();
        return r;
    }

    Flow stmt(const Stmt& s) {
        tick();
        switch (s.kind) {
            case Stmt::Kind::Let: {
                auto& x = static_cast<const LetStmt&>(s);
                Val v = eval(*x.init);
                env_.back()[x.name] = v;
                return Flow::Next;
            }
            case Stmt::Kind::Assign: {
                auto& x = static_cast<const AssignStmt&>(s);
                Val v = eval(*x.value);
                var(x.name) = v;
                return Flow::Next;
            }
            case Stmt::Kind::Store: {
                auto& x = static_cast<const StoreStmt&>(s);
                int arg = var(x.array).arg;
                int32_t idx = eval(*x.idx).i();
                Val v = eval(*x.value);
                element(l_, arg, idx) = v.bits;
                return Flow::Next;
            }
            case Stmt::Kind::If: {
                auto& x = static_cast<const IfStmt&>(s);
                return eval(*x.cond).bits ? block(x.thenB) : block(x.elseB);
            }
            case Stmt::Kind::While: {
                auto& x = static_cast<const WhileStmt&>(s);
                while (eval(*x.cond).bits) {
                    tick();
                    if (block(x.body) == Flow::Return) return Flow::Return;
                }
                return Flow::Next;
            }
            case Stmt::Kind::Return: return Flow::Return;
        }
        return Flow::Next;
    }

    Val eval(const Expr& e) {
        switch (e.kind) {
            case Expr::Kind::IntLit: return mkI(static_cast<const IntLit&>(e).value);
            case Expr::Kind::FloatLit: return mkF(static_cast<const FloatLit&>(e).value);
            case Expr::Kind::BoolLit: return mkB(static_cast<const BoolLit&>(e).value);
            case Expr::Kind::Var: return var(static_cast<const VarRef&>(e).name);
            case Expr::Kind::Index: {
                auto& x = static_cast<const Index&>(e);
                int arg = var(x.array).arg;
                int32_t idx = eval(*x.idx).i();
                return {e.type, element(l_, arg, idx), -1};
            }
            case Expr::Kind::Unary: {
                auto& u = static_cast<const Unary&>(e);
                Val v = eval(*u.operand);
                if (u.op == UnOp::Not) return mkB(!v.bits);
                return v.type == Type::F32 ? mkF(-v.f()) : mkI(wrapNeg(v.i()));
            }
            case Expr::Kind::Binary: return binary(static_cast<const Binary&>(e));
            case Expr::Kind::Call: {
                auto& c = static_cast<const Call&>(e);
                if (c.fn == "gid") return mkI(ctx_.bid * ctx_.bdim + ctx_.tid);
                if (c.fn == "tid") return mkI(ctx_.tid);
                if (c.fn == "bid") return mkI(ctx_.bid);
                if (c.fn == "bdim") return mkI(ctx_.bdim);
                if (c.fn == "gdim") return mkI(ctx_.gdim);
                Val a = eval(*c.args[0]);
                if (c.fn == "f32") return a.type == Type::F32 ? a : mkF((float)a.i());
                if (a.type == Type::I32) return a;
                if (a.type == Type::Bool) return mkI(a.bits ? 1 : 0);
                return mkI(cvtF2I(a.f()));
            }
        }
        throw std::logic_error("AST interpreter: bad expression");
    }

    Val binary(const Binary& b) {
        if (b.op == BinOp::And) { Val l = eval(*b.lhs); return l.bits ? mkB(eval(*b.rhs).bits) : mkB(false); }
        if (b.op == BinOp::Or) { Val l = eval(*b.lhs); return l.bits ? mkB(true) : mkB(eval(*b.rhs).bits); }
        Val l = eval(*b.lhs), r = eval(*b.rhs);
        if (l.type == Type::F32) {
            float x = l.f(), y = r.f();
            switch (b.op) {
                case BinOp::Add: return mkF(x + y);
                case BinOp::Sub: return mkF(x - y);
                case BinOp::Mul: return mkF(x * y);
                case BinOp::Div: return mkF(x / y);
                case BinOp::Lt: return mkB(x < y);
                case BinOp::Le: return mkB(x <= y);
                case BinOp::Gt: return mkB(x > y);
                case BinOp::Ge: return mkB(x >= y);
                case BinOp::Eq: return mkB(x == y);
                case BinOp::Ne: return mkB(x != y);
                default: break;
            }
        } else if (l.type == Type::Bool) {
            if (b.op == BinOp::Eq) return mkB(l.bits == r.bits);
            if (b.op == BinOp::Ne) return mkB(l.bits != r.bits);
        } else {
            int32_t x = l.i(), y = r.i();
            switch (b.op) {
                case BinOp::Add: return mkI(wrapAdd(x, y));
                case BinOp::Sub: return mkI(wrapSub(x, y));
                case BinOp::Mul: return mkI(wrapMul(x, y));
                case BinOp::Div:
                    if (divIsUB(x, y)) throw UndefinedBehavior("i32 division by zero or INT_MIN / -1");
                    return mkI(x / y);
                case BinOp::Rem:
                    if (divIsUB(x, y)) throw UndefinedBehavior("i32 remainder by zero or INT_MIN % -1");
                    return mkI(x % y);
                case BinOp::Lt: return mkB(x < y);
                case BinOp::Le: return mkB(x <= y);
                case BinOp::Gt: return mkB(x > y);
                case BinOp::Ge: return mkB(x >= y);
                case BinOp::Eq: return mkB(x == y);
                case BinOp::Ne: return mkB(x != y);
                default: break;
            }
        }
        throw std::logic_error("AST interpreter: bad binary op");
    }
};

}  // namespace

void runAst(const Program& prog, Launch& l) {
    for (auto& k : prog.kernels) {
        if (k.name != l.kernel) continue;
        std::vector<Type> ts;
        for (auto& p : k.params) ts.push_back(p.type);
        checkArgs(ts, l, k.name);
        AstInterp(k, l).run();
        return;
    }
    throw std::runtime_error("no kernel named '" + l.kernel + "'");
}

// ============================================================ IR interpreter
// Registers are 64-bit. Pointers are modelled as  (arg+1) << 40  |  2^39 + byte_offset
// so ElemPtr arithmetic is ordinary integer addition, and Load/Store can recover
// which buffer and which element an address refers to (and bounds-check it).
static const uint64_t kBias = 1ull << 39, kMask = (1ull << 40) - 1;
static uint64_t encodePtr(int arg) { return ((uint64_t)(arg + 1) << 40) | kBias; }
static uint32_t& deref(Launch& l, uint64_t addr) {
    int arg = (int)(addr >> 40) - 1;
    int64_t byteOff = (int64_t)(addr & kMask) - (int64_t)kBias;
    if (byteOff % 4 != 0) throw UndefinedBehavior("misaligned address");
    int64_t idx = byteOff / 4;
    if (idx < INT32_MIN || idx > INT32_MAX) throw UndefinedBehavior("address out of range");
    return element(l, arg, (int32_t)idx);
}

uint64_t runIr(const Module& mod, Launch& l) {
    const Function* fn = nullptr;
    for (auto& f : mod.functions) if (f.name == l.kernel) fn = &f;
    if (!fn) throw std::runtime_error("no kernel named '" + l.kernel + "'");
    std::vector<Type> ts;
    for (auto& p : fn->params) ts.push_back(p.type);
    checkArgs(ts, l, fn->name);

    std::vector<uint64_t> regs(fn->regTypes.size());
    uint64_t steps = 0;

    for (int b = 0; b < l.grid; b++)
        for (int t = 0; t < l.block; t++) {
            std::fill(regs.begin(), regs.end(), 0u);  // reading an undefined register gives 0
            for (size_t i = 0; i < fn->params.size(); i++)
                regs[fn->params[i].id] = isPtr(fn->params[i].type) ? encodePtr((int)i) : l.args[i].scalar;
            int bi = 0;
            for (;;) {
                const BasicBlock& blk = fn->blocks[bi];
                int next = -1;
                bool done = false;
                for (const Instr& in : blk.instrs) {
                    if (++steps > kStepLimit) throw UndefinedBehavior("step limit exceeded (non-terminating loop?)");
                    uint32_t s[3] = {0, 0, 0};
                    for (size_t i = 0; i < in.srcs.size() && i < 3; i++) s[i] = (uint32_t)regs[in.srcs[i].id];
                    switch (in.op) {
                        case Op::ByteOff:  // sign-extend, scale by sizeof(element) = 4
                            regs[in.dst.id] = (uint64_t)((int64_t)(int32_t)s[0] * 4);
                            break;
                        case Op::ElemPtr: regs[in.dst.id] = regs[in.srcs[0].id] + regs[in.srcs[1].id]; break;
                        case Op::Load: regs[in.dst.id] = deref(l, regs[in.srcs[0].id]); break;
                        case Op::Store: deref(l, regs[in.srcs[0].id]) = s[1]; break;
                        case Op::Gid: regs[in.dst.id] = (uint32_t)(b * l.block + t); break;
                        case Op::Tid: regs[in.dst.id] = (uint32_t)t; break;
                        case Op::Bid: regs[in.dst.id] = (uint32_t)b; break;
                        case Op::Bdim: regs[in.dst.id] = (uint32_t)l.block; break;
                        case Op::Gdim: regs[in.dst.id] = (uint32_t)l.grid; break;
                        case Op::Br: next = in.target; break;
                        case Op::CondBr: next = s[0] ? in.target : in.target2; break;
                        case Op::Ret: done = true; break;
                        case Op::Mov: regs[in.dst.id] = regs[in.srcs[0].id]; break;  // also moves 64-bit values
                        default: regs[in.dst.id] = evalPure(in, s); break;
                    }
                    if (next >= 0 || done) break;
                }
                if (done) break;
                if (next < 0) throw std::logic_error("IR interpreter: block without terminator: " + blk.label);
                bi = next;
            }
        }
    return steps;
}

}  // namespace kernc
