// parser.cpp
#include "parser.h"
#include <cerrno>
#include <cstdlib>
#include <sstream>

namespace kernc {

const char* binOpText(BinOp op) {
    switch (op) {
        case BinOp::Add: return "+"; case BinOp::Sub: return "-"; case BinOp::Mul: return "*";
        case BinOp::Div: return "/"; case BinOp::Rem: return "%"; case BinOp::Lt: return "<";
        case BinOp::Le: return "<="; case BinOp::Gt: return ">"; case BinOp::Ge: return ">=";
        case BinOp::Eq: return "=="; case BinOp::Ne: return "!="; case BinOp::And: return "&&";
        case BinOp::Or: return "||";
    }
    return "?";
}

const Token& Parser::peek(size_t k) const {
    size_t i = pos_ + k;
    return i < toks_.size() ? toks_[i] : toks_.back();
}
const Token& Parser::advance() {
    const Token& t = toks_[pos_];
    if (pos_ + 1 < toks_.size()) pos_++;
    return t;
}
bool Parser::match(Tok k) {
    if (check(k)) { advance(); return true; }
    return false;
}
void Parser::error(const Token& t, const std::string& msg) { throw CompileError(t.line, t.col, msg); }

const Token& Parser::expect(Tok k, const char* context) {
    if (!check(k))
        error(peek(), std::string("expected ") + tokName(k) + " " + context + ", found " + tokName(peek().kind));
    return advance();
}

Program parseSource(const std::string& src) {
    Lexer lx(src);
    Parser p(lx.tokenize());
    return p.parseProgram();
}

Program Parser::parseProgram() {
    Program prog;
    while (!check(Tok::Eof)) prog.kernels.push_back(parseKernel());
    if (prog.kernels.empty()) error(peek(), "program contains no kernels");
    return prog;
}

Type Parser::parseType() {
    Type base;
    if (match(Tok::KwI32)) base = Type::I32;
    else if (match(Tok::KwF32)) base = Type::F32;
    else if (match(Tok::KwBool)) base = Type::Bool;
    else error(peek(), std::string("expected a type, found ") + tokName(peek().kind));
    if (check(Tok::Star)) {
        const Token& star = advance();
        if (base == Type::Bool) error(star, "bool pointers are not supported");
        return base == Type::I32 ? Type::PtrI32 : Type::PtrF32;
    }
    return base;
}

Kernel Parser::parseKernel() {
    const Token& kw = expect(Tok::Kernel, "at start of kernel");
    Kernel k;
    k.line = kw.line; k.col = kw.col;
    k.name = expect(Tok::Ident, "as kernel name").text;
    expect(Tok::LParen, "after kernel name");
    if (!check(Tok::RParen)) {
        do {
            const Token& start = peek();
            Type t = parseType();
            std::string n = expect(Tok::Ident, "as parameter name").text;
            k.params.push_back({t, n, start.line, start.col});
        } while (match(Tok::Comma));
    }
    expect(Tok::RParen, "after parameter list");
    k.body = parseBlock();
    return k;
}

Block Parser::parseBlock() {
    expect(Tok::LBrace, "to open block");
    Block b;
    while (!check(Tok::RBrace)) {
        if (check(Tok::Eof)) error(peek(), "unexpected end of file inside block (missing '}')");
        b.push_back(parseStmt());
    }
    advance();
    return b;
}

StmtPtr Parser::parseStmt() {
    const Token& t = peek();
    switch (t.kind) {
        case Tok::Let: {
            advance();
            std::string name = expect(Tok::Ident, "after 'let'").text;
            std::optional<Type> declared;
            if (match(Tok::Colon)) declared = parseType();
            expect(Tok::Assign, "in let statement");
            ExprPtr init = parseExpr();
            expect(Tok::Semi, "after let statement");
            return std::make_unique<LetStmt>(name, declared, std::move(init), t.line, t.col);
        }
        case Tok::If: {
            advance();
            expect(Tok::LParen, "after 'if'");
            ExprPtr cond = parseExpr();
            expect(Tok::RParen, "after if condition");
            Block thenB = parseBlock(), elseB;
            if (match(Tok::Else)) {
                if (check(Tok::If)) elseB.push_back(parseStmt());  // else if
                else elseB = parseBlock();
            }
            return std::make_unique<IfStmt>(std::move(cond), std::move(thenB), std::move(elseB), t.line, t.col);
        }
        case Tok::While: {
            advance();
            expect(Tok::LParen, "after 'while'");
            ExprPtr cond = parseExpr();
            expect(Tok::RParen, "after while condition");
            Block body = parseBlock();
            return std::make_unique<WhileStmt>(std::move(cond), std::move(body), t.line, t.col);
        }
        case Tok::Return:
            advance();
            expect(Tok::Semi, "after 'return'");
            return std::make_unique<ReturnStmt>(t.line, t.col);
        case Tok::Ident: {
            std::string name = advance().text;
            if (match(Tok::LBracket)) {
                ExprPtr idx = parseExpr();
                expect(Tok::RBracket, "after array index");
                expect(Tok::Assign, "in array store");
                ExprPtr val = parseExpr();
                expect(Tok::Semi, "after store");
                return std::make_unique<StoreStmt>(name, std::move(idx), std::move(val), t.line, t.col);
            }
            expect(Tok::Assign, "in assignment (expressions are not statements)");
            ExprPtr val = parseExpr();
            expect(Tok::Semi, "after assignment");
            return std::make_unique<AssignStmt>(name, std::move(val), t.line, t.col);
        }
        default:
            error(t, std::string("expected a statement, found ") + tokName(t.kind));
    }
}

// ------------------------------------------------------------- expressions
static int precedence(Tok t) {
    switch (t) {
        case Tok::OrOr: return 1;
        case Tok::AndAnd: return 2;
        case Tok::Eq: case Tok::Ne: return 3;
        case Tok::Lt: case Tok::Le: case Tok::Gt: case Tok::Ge: return 4;
        case Tok::Plus: case Tok::Minus: return 5;
        case Tok::Star: case Tok::Slash: case Tok::Percent: return 6;
        default: return -1;
    }
}
static BinOp toBinOp(Tok t) {
    switch (t) {
        case Tok::OrOr: return BinOp::Or; case Tok::AndAnd: return BinOp::And;
        case Tok::Eq: return BinOp::Eq; case Tok::Ne: return BinOp::Ne;
        case Tok::Lt: return BinOp::Lt; case Tok::Le: return BinOp::Le;
        case Tok::Gt: return BinOp::Gt; case Tok::Ge: return BinOp::Ge;
        case Tok::Plus: return BinOp::Add; case Tok::Minus: return BinOp::Sub;
        case Tok::Star: return BinOp::Mul; case Tok::Slash: return BinOp::Div;
        default: return BinOp::Rem;
    }
}

ExprPtr Parser::parseExpr(int minPrec) {
    ExprPtr lhs = parseUnary();
    for (;;) {
        int p = precedence(peek().kind);
        if (p < minPrec) return lhs;
        const Token& op = advance();
        // Comparisons are non-associative: `a < b < c` is rejected.
        bool nonAssoc = (p == 3 || p == 4);
        ExprPtr rhs = parseExpr(p + 1);
        lhs = std::make_unique<Binary>(toBinOp(op.kind), std::move(lhs), std::move(rhs), op.line, op.col);
        if (nonAssoc && precedence(peek().kind) == p)
            error(peek(), "comparison operators cannot be chained; use parentheses");
    }
}

ExprPtr Parser::parseUnary() {
    const Token& t = peek();
    if (t.kind == Tok::Minus) {
        advance();
        // `-2147483648` is the only way to write INT32_MIN as a literal.
        if (check(Tok::IntLit) && peek().text == "2147483648") {
            advance();
            return std::make_unique<IntLit>(INT32_MIN, t.line, t.col);
        }
        return std::make_unique<Unary>(UnOp::Neg, parseUnary(), t.line, t.col);
    }
    if (t.kind == Tok::Bang) {
        advance();
        return std::make_unique<Unary>(UnOp::Not, parseUnary(), t.line, t.col);
    }
    return parsePrimary();
}

ExprPtr Parser::parsePrimary() {
    const Token& t = advance();
    switch (t.kind) {
        case Tok::IntLit: {
            errno = 0;
            long long v = std::strtoll(t.text.c_str(), nullptr, 10);
            if (errno || v > INT32_MAX) error(t, "integer literal " + t.text + " does not fit in i32");
            return std::make_unique<IntLit>((int32_t)v, t.line, t.col);
        }
        case Tok::FloatLit: {
            float v = std::strtof(t.text.c_str(), nullptr);  // correctly rounded
            return std::make_unique<FloatLit>(v, t.line, t.col);
        }
        case Tok::True: return std::make_unique<BoolLit>(true, t.line, t.col);
        case Tok::False: return std::make_unique<BoolLit>(false, t.line, t.col);
        case Tok::LParen: {
            ExprPtr e = parseExpr();
            expect(Tok::RParen, "to close parenthesis");
            return e;
        }
        case Tok::KwI32: case Tok::KwF32: case Tok::Ident: {
            // i32(x) / f32(x) conversions parse like calls.
            std::string name = t.kind == Tok::KwI32 ? "i32" : t.kind == Tok::KwF32 ? "f32" : t.text;
            if (match(Tok::LParen)) {
                std::vector<ExprPtr> args;
                if (!check(Tok::RParen)) {
                    do args.push_back(parseExpr()); while (match(Tok::Comma));
                }
                expect(Tok::RParen, "after call arguments");
                return std::make_unique<Call>(name, std::move(args), t.line, t.col);
            }
            if (t.kind != Tok::Ident) error(t, "type name used as a value");
            if (match(Tok::LBracket)) {
                ExprPtr idx = parseExpr();
                expect(Tok::RBracket, "after array index");
                return std::make_unique<Index>(name, std::move(idx), t.line, t.col);
            }
            return std::make_unique<VarRef>(name, t.line, t.col);
        }
        default:
            error(t, std::string("expected an expression, found ") + tokName(t.kind));
    }
}

// ------------------------------------------------------------- AST dump
static void dumpExpr(std::ostringstream& o, const Expr& e) {
    switch (e.kind) {
        case Expr::Kind::IntLit: o << static_cast<const IntLit&>(e).value; break;
        case Expr::Kind::FloatLit: o << static_cast<const FloatLit&>(e).value << "f"; break;
        case Expr::Kind::BoolLit: o << (static_cast<const BoolLit&>(e).value ? "true" : "false"); break;
        case Expr::Kind::Var: o << static_cast<const VarRef&>(e).name; break;
        case Expr::Kind::Index: {
            auto& x = static_cast<const Index&>(e);
            o << x.array << "["; dumpExpr(o, *x.idx); o << "]"; break;
        }
        case Expr::Kind::Binary: {
            auto& b = static_cast<const Binary&>(e);
            o << "(" << binOpText(b.op) << " "; dumpExpr(o, *b.lhs); o << " "; dumpExpr(o, *b.rhs); o << ")"; break;
        }
        case Expr::Kind::Unary: {
            auto& u = static_cast<const Unary&>(e);
            o << "(" << (u.op == UnOp::Neg ? "neg" : "not") << " "; dumpExpr(o, *u.operand); o << ")"; break;
        }
        case Expr::Kind::Call: {
            auto& c = static_cast<const Call&>(e);
            o << "(call " << c.fn;
            for (auto& a : c.args) { o << " "; dumpExpr(o, *a); }
            o << ")"; break;
        }
    }
}

static void dumpBlock(std::ostringstream& o, const Block& b, int ind);

static void dumpStmt(std::ostringstream& o, const Stmt& s, int ind) {
    std::string pad(ind * 2, ' ');
    switch (s.kind) {
        case Stmt::Kind::Let: {
            auto& x = static_cast<const LetStmt&>(s);
            o << pad << "let " << x.name;
            if (x.declared) o << ": " << typeName(*x.declared);
            o << " = "; dumpExpr(o, *x.init); o << "\n"; break;
        }
        case Stmt::Kind::Assign: {
            auto& x = static_cast<const AssignStmt&>(s);
            o << pad << x.name << " = "; dumpExpr(o, *x.value); o << "\n"; break;
        }
        case Stmt::Kind::Store: {
            auto& x = static_cast<const StoreStmt&>(s);
            o << pad << x.array << "["; dumpExpr(o, *x.idx); o << "] = "; dumpExpr(o, *x.value); o << "\n"; break;
        }
        case Stmt::Kind::If: {
            auto& x = static_cast<const IfStmt&>(s);
            o << pad << "if "; dumpExpr(o, *x.cond); o << "\n";
            dumpBlock(o, x.thenB, ind + 1);
            if (!x.elseB.empty()) { o << pad << "else\n"; dumpBlock(o, x.elseB, ind + 1); }
            break;
        }
        case Stmt::Kind::While: {
            auto& x = static_cast<const WhileStmt&>(s);
            o << pad << "while "; dumpExpr(o, *x.cond); o << "\n";
            dumpBlock(o, x.body, ind + 1); break;
        }
        case Stmt::Kind::Return: o << pad << "return\n"; break;
    }
}
static void dumpBlock(std::ostringstream& o, const Block& b, int ind) {
    for (auto& s : b) dumpStmt(o, *s, ind);
}

std::string dumpAst(const Program& p) {
    std::ostringstream o;
    for (auto& k : p.kernels) {
        o << "kernel " << k.name << "(";
        for (size_t i = 0; i < k.params.size(); i++)
            o << (i ? ", " : "") << typeName(k.params[i].type) << " " << k.params[i].name;
        o << ")\n";
        dumpBlock(o, k.body, 1);
    }
    return o.str();
}

}  // namespace kernc
