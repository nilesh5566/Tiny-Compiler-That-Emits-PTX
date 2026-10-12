// driver.cpp
#include "driver.h"
#include "lower.h"
#include "parser.h"
#include "typecheck.h"

namespace kernc {

Compilation compileSource(const std::string& src, const CompileOptions& opt) {
    Compilation c;
    c.ast = parseSource(src);
    typecheck(c.ast);
    c.ir = lowerProgram(c.ast);
    for (auto& f : c.ir.functions) {
        auto s = optimize(f, opt.optLevel, opt.fma);
        for (auto& x : s) c.stats.push_back({f.name + ":" + x.name, x.before, x.after});
        if (opt.workarounds) layoutBlocks(f);
    }
    return c;
}

}  // namespace kernc
