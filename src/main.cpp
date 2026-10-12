// main.cpp - kernc command line.
#include <fstream>
#include <iostream>
#include <sstream>
#include "driver.h"
#include "interp.h"
#include "launch.h"
#include "lexer.h"
#include "mutate.h"

using namespace kernc;

static const char* kVersion = "kernc 1.0.0";

static const char* kUsage = R"(kernc - compiler for the Kern GPU kernel language (emits NVIDIA PTX)

usage:
  kernc compile <file.kern> [-o out.ptx] [-O0|-O1] [--fma] [--arch sm_75]
                [--dump-tokens] [--dump-ast] [--dump-ir] [--stats]
                [--no-workarounds]   (disable the ptxas-crash workaround, see docs/findings)
  kernc check   <file.kern>                       parse + type-check only
  kernc run     <file.kern> --launch <file.launch> [--engine ast|ir] [-O0|-O1] [--fma]
                                                  execute on the CPU reference interpreters
  kernc mutants                                   list injectable bugs (KERNC_MUTATE=<id>)
  kernc --version | --help

exit codes: 0 ok, 1 compile error, 2 usage / I/O / runtime error
)";

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "--help" || args[0] == "-h") { std::cout << kUsage; return args.empty() ? 2 : 0; }
    if (args[0] == "--version") { std::cout << kVersion << "\n"; return 0; }
    if (args[0] == "mutants") {
        for (auto& m : kMutants) std::cout << m.id << "\t" << m.name << "\t" << m.description << "\n";
        return 0;
    }

    std::string cmd = args[0], file, out, launchFile, engine = "ast";
    CompileOptions opt;
    bool dumpTokens = false, dumpAstF = false, dumpIrF = false, stats = false;
    try {
        for (size_t i = 1; i < args.size(); i++) {
            const std::string& a = args[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= args.size()) throw std::runtime_error("missing value after " + a);
                return args[++i];
            };
            if (a == "-o") out = value();
            else if (a == "-O0") opt.optLevel = 0;
            else if (a == "-O1") opt.optLevel = 1;
            else if (a == "--fma") opt.fma = true;
            else if (a == "--no-workarounds") { opt.workarounds = false; opt.ptx.threadBranches = false; }
            else if (a == "--arch") opt.ptx.arch = value();
            else if (a == "--dump-tokens") dumpTokens = true;
            else if (a == "--dump-ast") dumpAstF = true;
            else if (a == "--dump-ir") dumpIrF = true;
            else if (a == "--stats") stats = true;
            else if (a == "--launch") launchFile = value();
            else if (a == "--engine") engine = value();
            else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a);
            else if (file.empty()) file = a;
            else throw std::runtime_error("unexpected argument " + a);
        }
        if (cmd != "compile" && cmd != "check" && cmd != "run") throw std::runtime_error("unknown command '" + cmd + "'");
        if (file.empty()) throw std::runtime_error("no input file");
    } catch (const std::exception& e) {
        std::cerr << "kernc: " << e.what() << "\n" << kUsage;
        return 2;
    }

    std::string src;
    try {
        src = readFile(file);
    } catch (const std::exception& e) {
        std::cerr << "kernc: " << e.what() << "\n";
        return 2;
    }

    try {
        if (dumpTokens) {
            for (auto& t : Lexer(src).tokenize())
                std::cout << t.line << ":" << t.col << "\t" << tokName(t.kind)
                          << (t.kind == Tok::Ident || t.kind == Tok::IntLit || t.kind == Tok::FloatLit ? "\t" + t.text : "")
                          << "\n";
        }
        Compilation c = compileSource(src, opt);
        if (cmd == "check") { std::cout << "ok\n"; return 0; }
        if (dumpAstF) std::cout << dumpAst(c.ast);
        if (dumpIrF) std::cout << dumpIr(c.ir);
        if (stats) {
            for (auto& f : c.ir.functions) std::cerr << "ir-instructions " << f.name << " " << countInstrs(f) << "\n";
            for (auto& s : c.stats) std::cerr << "pass " << s.name << " " << s.before << " -> " << s.after << "\n";
        }

        if (cmd == "run") {
            if (launchFile.empty()) { std::cerr << "kernc: run needs --launch <file>\n"; return 2; }
            Launch l = parseLaunch(readFile(launchFile));
            std::string status = "ok";
            try {
                if (engine == "ast") runAst(c.ast, l);
                else if (engine == "ir") {
                    uint64_t executed = runIr(c.ir, l);
                    if (stats) std::cerr << "ir-executed " << executed << "\n";
                }
                else { std::cerr << "kernc: --engine must be ast or ir\n"; return 2; }
            } catch (const UndefinedBehavior& ub) {
                status = std::string("ub ") + ub.what();
            }
            std::cout << formatResult(l, status);
            return 0;
        }

        std::string ptx = emitPtx(c.ir, opt.ptx);
        if (dumpTokens || dumpAstF || dumpIrF) {
            if (!out.empty()) { std::ofstream(out, std::ios::binary) << ptx; }
            return 0;
        }
        if (out.empty() || out == "-") std::cout << ptx;
        else {
            std::ofstream o(out, std::ios::binary);
            if (!o) { std::cerr << "kernc: cannot write '" << out << "'\n"; return 2; }
            o << ptx;
        }
        return 0;
    } catch (const CompileError& e) {
        std::cerr << file << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "kernc: " << e.what() << "\n";
        return 2;
    }
}
