// ast.h - abstract syntax tree for Kern.
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "common.h"

namespace kernc {

enum class BinOp { Add, Sub, Mul, Div, Rem, Lt, Le, Gt, Ge, Eq, Ne, And, Or };
enum class UnOp { Neg, Not };

const char* binOpText(BinOp op);

// ------------------------------------------------------------------ expressions
struct Expr {
    enum class Kind { IntLit, FloatLit, BoolLit, Var, Index, Binary, Unary, Call };
    Kind kind;
    int line, col;
    Type type = Type::Error;  // set by the type checker
    Expr(Kind k, int l, int c) : kind(k), line(l), col(c) {}
    virtual ~Expr() = default;
};
using ExprPtr = std::unique_ptr<Expr>;

struct IntLit : Expr { int32_t value; IntLit(int32_t v, int l, int c) : Expr(Kind::IntLit, l, c), value(v) {} };
struct FloatLit : Expr { float value; FloatLit(float v, int l, int c) : Expr(Kind::FloatLit, l, c), value(v) {} };
struct BoolLit : Expr { bool value; BoolLit(bool v, int l, int c) : Expr(Kind::BoolLit, l, c), value(v) {} };
struct VarRef : Expr { std::string name; VarRef(std::string n, int l, int c) : Expr(Kind::Var, l, c), name(std::move(n)) {} };
struct Index : Expr {
    std::string array; ExprPtr idx;
    Index(std::string a, ExprPtr i, int l, int c) : Expr(Kind::Index, l, c), array(std::move(a)), idx(std::move(i)) {}
};
struct Binary : Expr {
    BinOp op; ExprPtr lhs, rhs;
    Binary(BinOp o, ExprPtr a, ExprPtr b, int l, int c) : Expr(Kind::Binary, l, c), op(o), lhs(std::move(a)), rhs(std::move(b)) {}
};
struct Unary : Expr {
    UnOp op; ExprPtr operand;
    Unary(UnOp o, ExprPtr e, int l, int c) : Expr(Kind::Unary, l, c), op(o), operand(std::move(e)) {}
};
struct Call : Expr {
    std::string fn; std::vector<ExprPtr> args;
    Call(std::string f, std::vector<ExprPtr> a, int l, int c) : Expr(Kind::Call, l, c), fn(std::move(f)), args(std::move(a)) {}
};

// ------------------------------------------------------------------ statements
struct Stmt {
    enum class Kind { Let, Assign, Store, If, While, Return };
    Kind kind;
    int line, col;
    Stmt(Kind k, int l, int c) : kind(k), line(l), col(c) {}
    virtual ~Stmt() = default;
};
using StmtPtr = std::unique_ptr<Stmt>;
using Block = std::vector<StmtPtr>;

struct LetStmt : Stmt {
    std::string name; std::optional<Type> declared; ExprPtr init;
    LetStmt(std::string n, std::optional<Type> t, ExprPtr e, int l, int c)
        : Stmt(Kind::Let, l, c), name(std::move(n)), declared(t), init(std::move(e)) {}
};
struct AssignStmt : Stmt {
    std::string name; ExprPtr value;
    AssignStmt(std::string n, ExprPtr v, int l, int c) : Stmt(Kind::Assign, l, c), name(std::move(n)), value(std::move(v)) {}
};
struct StoreStmt : Stmt {
    std::string array; ExprPtr idx, value;
    StoreStmt(std::string a, ExprPtr i, ExprPtr v, int l, int c)
        : Stmt(Kind::Store, l, c), array(std::move(a)), idx(std::move(i)), value(std::move(v)) {}
};
struct IfStmt : Stmt {
    ExprPtr cond; Block thenB, elseB;
    IfStmt(ExprPtr c0, Block t, Block e, int l, int c)
        : Stmt(Kind::If, l, c), cond(std::move(c0)), thenB(std::move(t)), elseB(std::move(e)) {}
};
struct WhileStmt : Stmt {
    ExprPtr cond; Block body;
    WhileStmt(ExprPtr c0, Block b, int l, int c) : Stmt(Kind::While, l, c), cond(std::move(c0)), body(std::move(b)) {}
};
struct ReturnStmt : Stmt { ReturnStmt(int l, int c) : Stmt(Kind::Return, l, c) {} };

// ------------------------------------------------------------------ top level
struct Param { Type type; std::string name; int line, col; };

struct Kernel {
    std::string name;
    std::vector<Param> params;
    Block body;
    int line, col;
};

struct Program { std::vector<Kernel> kernels; };

std::string dumpAst(const Program& p);

}  // namespace kernc
