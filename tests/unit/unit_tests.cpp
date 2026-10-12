// unit_tests.cpp - fast C++ unit tests for each compiler stage.
// No external framework: CHECK records failures, main() reports a summary.
#include <cmath>
#include <functional>
#include <iostream>
#include "driver.h"
#include "interp.h"
#include "lexer.h"
#include "parser.h"
#include "passes.h"
#include "semantics.h"

using namespace kernc;

static int g_checks = 0, g_failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        g_checks++;                                                                    \
        if (!(cond)) { g_failures++; std::cerr << "  FAILED " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; } \
    } while (0)

static std::vector<std::pair<const char*, std::function<void()>>>& registry() {
    static std::vector<std::pair<const char*, std::function<void()>>> r;
    return r;
}
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()

static std::string compileError(const std::string& src) {
    try { compileSource(src, {}); } catch (const CompileError& e) { return e.what(); }
    return "";
}

// Run a one-kernel program on a launch, with the given engine and opt level.
static Launch runWith(const std::string& src, Launch l, bool ir, int opt, bool fma = false) {
    CompileOptions o; o.optLevel = opt; o.fma = fma;
    Compilation c = compileSource(src, o);
    if (ir) runIr(c.ir, l); else runAst(c.ast, l);
    return l;
}
static LaunchArg i32buf(size_t n) { return {Type::PtrI32, 0, std::vector<uint32_t>(n, 0)}; }
static LaunchArg f32buf(size_t n) { return {Type::PtrF32, 0, std::vector<uint32_t>(n, 0)}; }

// ---------------------------------------------------------------- lexer
TEST(lexer_tokens) {
    auto t = Lexer("let x: f32 = 1.5e2; // c\n a<=b && !c").tokenize();
    std::vector<Tok> k;
    for (auto& x : t) k.push_back(x.kind);
    std::vector<Tok> want = {Tok::Let, Tok::Ident, Tok::Colon, Tok::KwF32, Tok::Assign, Tok::FloatLit, Tok::Semi,
                             Tok::Ident, Tok::Le, Tok::Ident, Tok::AndAnd, Tok::Bang, Tok::Ident, Tok::Eof};
    CHECK(k == want);
    CHECK(t[5].text == "1.5e2");
    CHECK(t[7].line == 2 && t[7].col == 2);
}

TEST(lexer_errors) {
    bool threw = false;
    try { Lexer("a & b").tokenize(); } catch (const CompileError&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { Lexer("1.").tokenize(); } catch (const CompileError&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------- parser
TEST(parser_precedence) {
    Program p = parseSource("kernel k(i32* o) { o[0] = 1 + 2 * 3 - 4; }");
    std::string d = dumpAst(p);
    CHECK(d.find("(- (+ 1 (* 2 3)) 4)") != std::string::npos);
    Program q = parseSource("kernel k(i32* o) { let b = 1 < 2 && 3 == 3 || false; }");
    CHECK(dumpAst(q).find("(|| (&& (< 1 2) (== 3 3)) false)") != std::string::npos);
}

TEST(parser_int_min_literal) {
    Program p = parseSource("kernel k(i32* o) { o[0] = -2147483648; }");
    CHECK(dumpAst(p).find("-2147483648") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { o[0] = 2147483648; }").find("does not fit") != std::string::npos);
}

TEST(parser_rejects_chained_compare) {
    CHECK(compileError("kernel k(i32* o) { let b = 1 < 2 < 3; }").find("chained") != std::string::npos);
}

// ---------------------------------------------------------------- type checker
TEST(typecheck_errors) {
    CHECK(compileError("kernel k(i32* o) { o[0] = 1 + 2.0; }").find("cannot apply '+'") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { o[0] = y; }").find("undeclared") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { let a = 1; let a = 2; }").find("redeclaration") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { if (1) { } }").find("must be bool") != std::string::npos);
    CHECK(compileError("kernel k(bool b) { }").find("cannot be bool") != std::string::npos);
    CHECK(compileError("kernel k(f32* o) { o[0] = 1; }").find("cannot store") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { let x: f32 = 0; }").find("no implicit") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { let x = 1; } kernel k(i32* o) { }").find("duplicate kernel") != std::string::npos);
    CHECK(compileError("kernel k(i32* o) { o[0] = foo(); }").find("unknown function") != std::string::npos);
}

TEST(typecheck_scoping) {
    // inner variable shadows, outer is visible again afterwards
    CHECK(compileError("kernel k(i32* o) { let a = 1; if (true) { let a = 2.0; o[0] = i32(a); } o[1] = a; }").empty());
    CHECK(compileError("kernel k(i32* o) { if (true) { let a = 1; } o[0] = a; }").find("undeclared") != std::string::npos);
}

// ---------------------------------------------------------------- semantics
TEST(semantics_int_wrap_and_div) {
    const char* src = "kernel k(i32* o, i32 a) { o[0] = a + 1; o[1] = a * 2; o[2] = -7 / 2; o[3] = -7 % 2; o[4] = -a; }";
    for (bool ir : {false, true})
        for (int opt : {0, 1}) {
            Launch l{"k", 1, 1, {i32buf(5), {Type::I32, (uint32_t)INT32_MAX, {}}}};
            l = runWith(src, l, ir, opt);
            CHECK((int32_t)l.args[0].buf[0] == INT32_MIN);   // wraps
            CHECK((int32_t)l.args[0].buf[1] == -2);
            CHECK((int32_t)l.args[0].buf[2] == -3);          // truncates toward zero
            CHECK((int32_t)l.args[0].buf[3] == -1);
            CHECK((int32_t)l.args[0].buf[4] == -INT32_MAX);
        }
}

TEST(semantics_cvt_saturates) {
    CHECK(cvtF2I(3.9f) == 3);
    CHECK(cvtF2I(-3.9f) == -3);
    CHECK(cvtF2I(1e20f) == INT32_MAX);
    CHECK(cvtF2I(-1e20f) == INT32_MIN);
    CHECK(cvtF2I(std::nanf("")) == 0);
}

TEST(semantics_negative_zero) {
    const char* src = "kernel k(f32* o, f32 z) { o[0] = -z; o[1] = 1.0 / -z; }";
    for (bool ir : {false, true})
        for (int opt : {0, 1}) {
            Launch l{"k", 1, 1, {f32buf(2), {Type::F32, f2bits(0.0f), {}}}};
            l = runWith(src, l, ir, opt);
            CHECK(l.args[0].buf[0] == 0x80000000u);  // -0.0
            CHECK(std::isinf(bits2f(l.args[0].buf[1])) && bits2f(l.args[0].buf[1]) < 0);
        }
}

TEST(semantics_nan_not_equal) {
    const char* src = "kernel k(i32* o, f32 x) { o[0] = i32(x != x); o[1] = i32(x == x); o[2] = i32(x < 1.0); }";
    for (bool ir : {false, true}) {
        Launch l{"k", 1, 1, {i32buf(3), {Type::F32, 0x7fc00000u, {}}}};
        l = runWith(src, l, ir, 1);
        CHECK(l.args[0].buf[0] == 1 && l.args[0].buf[1] == 0 && l.args[0].buf[2] == 0);
    }
}

TEST(semantics_short_circuit) {
    // a[5] would be out of bounds, but must never be evaluated
    const char* src = "kernel k(i32* a, i32* o) { let i = 5; if (i < 4 && a[i] > 0) { o[0] = 1; } else { o[0] = 2; } }";
    for (bool ir : {false, true}) {
        Launch l{"k", 1, 1, {i32buf(4), i32buf(1)}};
        l = runWith(src, l, ir, 1);
        CHECK(l.args[1].buf[0] == 2);
    }
}

TEST(semantics_ub_detected) {
    const char* src = "kernel k(i32* o, i32 d) { o[0] = 10 / d; }";
    bool ub = false;
    try { runWith(src, {"k", 1, 1, {i32buf(1), {Type::I32, 0, {}}}}, false, 0); } catch (const UndefinedBehavior&) { ub = true; }
    CHECK(ub);
    ub = false;
    try { runWith("kernel k(i32* o) { o[1] = 3; }", {"k", 1, 1, {i32buf(1)}}, true, 0); } catch (const UndefinedBehavior&) { ub = true; }
    CHECK(ub);
}

TEST(semantics_threads) {
    const char* src = "kernel k(i32* o) { o[gid()] = bid() * 1000 + tid() * 10 + bdim() + gdim(); }";
    for (bool ir : {false, true}) {
        Launch l{"k", 2, 3, {i32buf(6)}};
        l = runWith(src, l, ir, 1);
        CHECK(l.args[0].buf[0] == 5 && l.args[0].buf[4] == 1000 + 10 + 5);
    }
}

// ---------------------------------------------------------------- passes
static size_t irSize(const std::string& src, int opt, bool fma = false) {
    CompileOptions o; o.optLevel = opt; o.fma = fma;
    return countInstrs(compileSource(src, o).ir.functions[0]);
}

TEST(pass_constfold_and_dce) {
    CompileOptions o; o.optLevel = 1;
    Compilation c = compileSource("kernel k(i32* o) { let a = 2 + 3 * 4; o[0] = a - 1; }", o);
    std::string ir = dumpIr(c.ir);
    CHECK(ir.find("const 13") != std::string::npos);
    CHECK(ir.find("add") == std::string::npos && ir.find("mul") == std::string::npos);
}

TEST(pass_constfold_keeps_ub) {
    CompileOptions o; o.optLevel = 1;
    Compilation c = compileSource("kernel k(i32* o) { o[0] = 1 / 0; }", o);
    CHECK(dumpIr(c.ir).find("div") != std::string::npos);  // must not fold UB away
}

TEST(pass_branch_folding) {
    CHECK(irSize("kernel k(i32* o) { if (1 < 2) { o[0] = 1; } else { o[0] = 2; } }", 1) <
          irSize("kernel k(i32* o) { if (1 < 2) { o[0] = 1; } else { o[0] = 2; } }", 0));
}

TEST(pass_strength_reduction) {
    CompileOptions o; o.optLevel = 1;
    Compilation c = compileSource("kernel k(i32* o, i32 x) { o[0] = x * 8; o[1] = x * -8; }", o);
    std::string ir = dumpIr(c.ir);
    CHECK(ir.find("shl") != std::string::npos);
    CHECK(ir.find("mul") != std::string::npos);  // x * -8 must stay a multiply
}

TEST(pass_cse) {
    CompileOptions o; o.optLevel = 1;
    Compilation c = compileSource("kernel k(i32* o, i32 x, i32 y) { o[0] = x * y + x * y; }", o);
    std::string ir = dumpIr(c.ir);
    CHECK(ir.find("mul") == ir.rfind("mul"));  // only one multiply left
}

TEST(pass_fma_contraction) {
    CompileOptions o; o.optLevel = 1; o.fma = true;
    Compilation c = compileSource("kernel k(f32* o, f32 a, f32 b) { o[0] = a * b + o[0]; }", o);
    CHECK(dumpIr(c.ir).find("fma") != std::string::npos);
    o.fma = false;
    CHECK(dumpIr(compileSource("kernel k(f32* o, f32 a, f32 b) { o[0] = a * b + o[0]; }", o).ir).find("fma") == std::string::npos);
}

// Passes tested in isolation on hand-built IR. These two tests were added after
// mutation testing showed that mutants #3 and #4 survived the end-to-end suite:
// in the normal pipeline copy propagation hides the stale-operand situation from
// CSE, so only a direct test of each pass can catch those bugs.
static Function handBuilt(VReg& a, VReg& out) {
    Function f;
    f.name = "t";
    out = f.newReg(Type::PtrI32);
    a = f.newReg(Type::I32);
    f.params = {out, a};
    f.paramNames = {"out", "a"};
    f.blocks.push_back({"entry", {}});
    return f;
}

TEST(pass_copyprop_isolated) {
    VReg a, out;
    Function f = handBuilt(a, out);
    auto& v = f.blocks[0].instrs;
    VReg t = f.newReg(Type::I32), one = f.newReg(Type::I32), zero = f.newReg(Type::I32);
    Instr mv(Op::Mov); mv.dst = t; mv.srcs = {a}; v.push_back(mv);                 // t = a
    Instr c1(Op::Const); c1.dst = one; c1.imm = 1; v.push_back(c1);
    Instr add(Op::Add); add.dst = a; add.srcs = {a, one}; v.push_back(add);        // a = a + 1   (source changes!)
    Instr c0(Op::Const); c0.dst = zero; c0.imm = 0; v.push_back(c0);
    Instr st(Op::Store); st.srcs = {out, zero, t}; v.push_back(st);                // out[0] = t  (old a)
    v.push_back(Instr(Op::Ret));
    copyPropagate(f);
    const Instr& store = f.blocks[0].instrs[4];
    CHECK(store.op == Op::Store);
    CHECK(store.srcs[2].id != a.id);  // must NOT read the new value of a
}

TEST(pass_cse_isolated) {
    VReg a, out;
    Function f = handBuilt(a, out);
    auto& v = f.blocks[0].instrs;
    VReg m1 = f.newReg(Type::I32), m2 = f.newReg(Type::I32), one = f.newReg(Type::I32);
    Instr mul1(Op::Mul); mul1.dst = m1; mul1.srcs = {a, a}; v.push_back(mul1);   // m1 = a * a
    Instr c1(Op::Const); c1.dst = one; c1.imm = 1; v.push_back(c1);
    Instr add(Op::Add); add.dst = a; add.srcs = {a, one}; v.push_back(add);      // a = a + 1
    Instr mul2(Op::Mul); mul2.dst = m2; mul2.srcs = {a, a}; v.push_back(mul2);   // m2 = a * a  (NOT m1!)
    Instr st(Op::Store); st.srcs = {out, m1, m2}; v.push_back(st);
    v.push_back(Instr(Op::Ret));
    cse(f);
    CHECK(f.blocks[0].instrs[3].op == Op::Mul);
}

TEST(pass_licm) {
    CompileOptions o; o.optLevel = 1;
    std::string ir = dumpIr(compileSource(
        "kernel k(f32* x, f32* o, i32 n, i32 d) { let i = gid(); let s = 0.0; let c = 0;"
        "  while (c < n) { s = s + x[i] * 2.0 + f32(100 / d); c = c + 1; } o[i] = s; }", o).ir);
    size_t body = ir.find("wbody");
    CHECK(body != std::string::npos);
    CHECK(ir.find("byteoff") < body);        // address of x[i] is loop-invariant: hoisted
    CHECK(ir.find("load") > body);           // loads stay in the loop
    CHECK(ir.find("div") > body);            // i32 division may trap: must stay in the loop
}

TEST(ptx_basics) {
    CompileOptions o; o.optLevel = 1;
    Compilation c = compileSource("kernel saxpy(f32* x, f32* y, f32 a, i32 n) { let i = gid(); if (i < n) { y[i] = a * x[i] + y[i]; } }", o);
    std::string p = emitPtx(c.ir, o.ptx);
    CHECK(p.find(".visible .entry saxpy(") != std::string::npos);
    CHECK(p.find("cvta.to.global.u64") != std::string::npos);
    CHECK(p.find("mad.lo.s32") != std::string::npos);
    CHECK(p.find("mul.rn.f32") != std::string::npos);
    CHECK(p.find("ld.global.f32") != std::string::npos && p.find("st.global.f32") != std::string::npos);
    CHECK(p.find("fma") == std::string::npos);
}

int main() {
    for (auto& [name, fn] : registry()) {
        int before = g_failures;
        try { fn(); } catch (const std::exception& e) { g_failures++; std::cerr << "  EXCEPTION: " << e.what() << "\n"; }
        std::cout << (g_failures == before ? "[ OK ] " : "[FAIL] ") << name << "\n";
    }
    std::cout << "\n" << registry().size() << " tests, " << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures ? 1 : 0;
}
