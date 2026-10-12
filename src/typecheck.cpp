// typecheck.cpp
#include "typecheck.h"
#include <unordered_map>
#include <unordered_set>
#include "mutate.h"

namespace kernc {

bool isThreadBuiltin(const std::string& fn) {
    return fn == "gid" || fn == "tid" || fn == "bid" || fn == "bdim" || fn == "gdim";
}

namespace {

struct Sym { Type type; bool isParam; };

class Checker {
public:
    void kernel(Kernel& k) {
        scopes_.clear();
        push();
        for (auto& p : k.params) {
            if (p.type == Type::Bool)
                throw CompileError(p.line, p.col, "kernel parameter '" + p.name + "' cannot be bool (PTX has no predicate parameters)");
            if (!declare(p.name, {p.type, true}))
                throw CompileError(p.line, p.col, "duplicate parameter '" + p.name + "'");
        }
        block(k.body);
        pop();
    }

private:
    std::vector<std::unordered_map<std::string, Sym>> scopes_;

    void push() { scopes_.emplace_back(); }
    void pop() { scopes_.pop_back(); }
    bool declare(const std::string& n, Sym s) { return scopes_.back().emplace(n, s).second; }
    const Sym* lookup(const std::string& n) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) return &f->second;
        }
        return nullptr;
    }
    [[noreturn]] static void err(int l, int c, const std::string& m) { throw CompileError(l, c, m); }

    void block(Block& b) {
        push();
        for (auto& s : b) stmt(*s);
        pop();
    }

    void stmt(Stmt& s) {
        switch (s.kind) {
            case Stmt::Kind::Let: {
                auto& x = static_cast<LetStmt&>(s);
                Type t = expr(*x.init);
                if (isPtr(t)) err(s.line, s.col, "cannot copy pointer '" + std::string(typeName(t)) + "' into a local variable");
                if (x.declared && *x.declared != t)
                    err(x.init->line, x.init->col, "variable '" + x.name + "' declared " + typeName(*x.declared) +
                        " but initialised with " + typeName(t) + " (no implicit conversions)");
                if (x.declared && isPtr(*x.declared)) err(s.line, s.col, "local variables cannot have pointer type");
                if (!declare(x.name, {t, false}))
                    err(s.line, s.col, "redeclaration of '" + x.name + "' in the same scope");
                break;
            }
            case Stmt::Kind::Assign: {
                auto& x = static_cast<AssignStmt&>(s);
                const Sym* sym = lookup(x.name);
                if (!sym) err(s.line, s.col, "assignment to undeclared variable '" + x.name + "'");
                if (isPtr(sym->type)) err(s.line, s.col, "cannot assign to pointer parameter '" + x.name + "'");
                Type t = expr(*x.value);
                if (t != sym->type)
                    err(x.value->line, x.value->col, "cannot assign " + std::string(typeName(t)) + " to '" + x.name +
                        "' of type " + typeName(sym->type));
                break;
            }
            case Stmt::Kind::Store: {
                auto& x = static_cast<StoreStmt&>(s);
                const Sym* sym = lookup(x.array);
                if (!sym) err(s.line, s.col, "use of undeclared array '" + x.array + "'");
                if (!isPtr(sym->type)) err(s.line, s.col, "'" + x.array + "' is not a pointer and cannot be indexed");
                if (expr(*x.idx) != Type::I32) err(x.idx->line, x.idx->col, "array index must be i32");
                Type v = expr(*x.value);
                if (v != elemOf(sym->type))
                    err(x.value->line, x.value->col, "cannot store " + std::string(typeName(v)) + " into " + typeName(sym->type));
                break;
            }
            case Stmt::Kind::If: {
                auto& x = static_cast<IfStmt&>(s);
                if (expr(*x.cond) != Type::Bool) err(x.cond->line, x.cond->col, "if condition must be bool");
                block(x.thenB);
                block(x.elseB);
                break;
            }
            case Stmt::Kind::While: {
                auto& x = static_cast<WhileStmt&>(s);
                if (expr(*x.cond) != Type::Bool) err(x.cond->line, x.cond->col, "while condition must be bool");
                block(x.body);
                break;
            }
            case Stmt::Kind::Return: break;
        }
    }

    Type expr(Expr& e) { return e.type = exprImpl(e); }

    Type exprImpl(Expr& e) {
        switch (e.kind) {
            case Expr::Kind::IntLit: return Type::I32;
            case Expr::Kind::FloatLit: return Type::F32;
            case Expr::Kind::BoolLit: return Type::Bool;
            case Expr::Kind::Var: {
                auto& v = static_cast<VarRef&>(e);
                const Sym* s = lookup(v.name);
                if (!s) err(e.line, e.col, "use of undeclared variable '" + v.name + "'");
                return s->type;
            }
            case Expr::Kind::Index: {
                auto& x = static_cast<Index&>(e);
                const Sym* s = lookup(x.array);
                if (!s) err(e.line, e.col, "use of undeclared array '" + x.array + "'");
                if (!isPtr(s->type)) err(e.line, e.col, "'" + x.array + "' is not a pointer and cannot be indexed");
                if (expr(*x.idx) != Type::I32) err(x.idx->line, x.idx->col, "array index must be i32");
                return elemOf(s->type);
            }
            case Expr::Kind::Unary: {
                auto& u = static_cast<Unary&>(e);
                Type t = expr(*u.operand);
                if (u.op == UnOp::Neg) {
                    if (t != Type::I32 && t != Type::F32) err(e.line, e.col, std::string("cannot negate ") + typeName(t));
                    return t;
                }
                if (t != Type::Bool) err(e.line, e.col, std::string("operator '!' needs bool, got ") + typeName(t));
                return Type::Bool;
            }
            case Expr::Kind::Binary: {
                auto& b = static_cast<Binary&>(e);
                Type l = expr(*b.lhs), r = expr(*b.rhs);
                std::string op = binOpText(b.op);
                auto mismatch = [&] {
                    err(e.line, e.col, "cannot apply '" + op + "' to " + typeName(l) + " and " + typeName(r));
                };
                switch (b.op) {
                    case BinOp::Add: case BinOp::Sub: case BinOp::Mul: case BinOp::Div:
                        if (mutant(15) && ((l == Type::I32 && r == Type::F32) || (l == Type::F32 && r == Type::I32)))
                            return Type::F32;  // MUTANT: accept mixed arithmetic
                        if (l != r || (l != Type::I32 && l != Type::F32)) mismatch();
                        return l;
                    case BinOp::Rem:
                        if (l != Type::I32 || r != Type::I32) mismatch();
                        return Type::I32;
                    case BinOp::Lt: case BinOp::Le: case BinOp::Gt: case BinOp::Ge:
                        if (l != r || (l != Type::I32 && l != Type::F32)) mismatch();
                        return Type::Bool;
                    case BinOp::Eq: case BinOp::Ne:
                        if (l != r || isPtr(l)) mismatch();
                        return Type::Bool;
                    case BinOp::And: case BinOp::Or:
                        if (l != Type::Bool || r != Type::Bool) mismatch();
                        return Type::Bool;
                }
                return Type::Error;
            }
            case Expr::Kind::Call: {
                auto& c = static_cast<Call&>(e);
                if (isThreadBuiltin(c.fn)) {
                    if (!c.args.empty()) err(e.line, e.col, c.fn + "() takes no arguments");
                    return Type::I32;
                }
                if (c.fn == "f32" || c.fn == "i32") {
                    if (c.args.size() != 1) err(e.line, e.col, c.fn + "() takes exactly one argument");
                    Type a = expr(*c.args[0]);
                    if (c.fn == "f32" && a != Type::I32 && a != Type::F32)
                        err(e.line, e.col, std::string("cannot convert ") + typeName(a) + " to f32");
                    if (c.fn == "i32" && a != Type::I32 && a != Type::F32 && a != Type::Bool)
                        err(e.line, e.col, std::string("cannot convert ") + typeName(a) + " to i32");
                    return c.fn == "f32" ? Type::F32 : Type::I32;
                }
                err(e.line, e.col, "unknown function '" + c.fn + "'");
            }
        }
        return Type::Error;
    }
};

}  // namespace

void typecheck(Program& prog) {
    std::unordered_set<std::string> names;
    for (auto& k : prog.kernels) {
        if (!names.insert(k.name).second) throw CompileError(k.line, k.col, "duplicate kernel '" + k.name + "'");
        Checker().kernel(k);
    }
}

}  // namespace kernc
