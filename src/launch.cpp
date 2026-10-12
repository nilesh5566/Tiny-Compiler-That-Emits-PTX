// launch.cpp
#include "launch.h"
#include <cstdio>
#include <sstream>

namespace kernc {

static uint32_t hex(const std::string& s) { return (uint32_t)std::stoul(s, nullptr, 16); }

Launch parseLaunch(const std::string& text) {
    Launch l;
    std::istringstream in(text);
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        lineNo++;
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        auto fail = [&](const std::string& m) {
            throw std::runtime_error("launch file line " + std::to_string(lineNo) + ": " + m);
        };
        if (key == "kernel") ls >> l.kernel;
        else if (key == "grid") ls >> l.grid;
        else if (key == "block") ls >> l.block;
        else if (key == "i32" || key == "f32") {
            std::string v;
            if (!(ls >> v)) fail("missing scalar value");
            l.args.push_back({key == "i32" ? Type::I32 : Type::F32, hex(v), {}});
        } else if (key == "i32buf" || key == "f32buf") {
            size_t n;
            if (!(ls >> n)) fail("missing buffer length");
            LaunchArg a{key == "i32buf" ? Type::PtrI32 : Type::PtrF32, 0, {}};
            a.buf.reserve(n);
            std::string v;
            for (size_t i = 0; i < n; i++) {
                if (!(ls >> v)) fail("buffer has fewer values than its length");
                a.buf.push_back(hex(v));
            }
            l.args.push_back(std::move(a));
        } else {
            fail("unknown key '" + key + "'");
        }
    }
    if (l.grid < 1 || l.block < 1) throw std::runtime_error("launch file: grid and block must be >= 1");
    return l;
}

std::string formatResult(const Launch& l, const std::string& status) {
    std::ostringstream o;
    o << "status " << status << "\n";
    char buf[16];
    for (size_t i = 0; i < l.args.size(); i++) {
        const auto& a = l.args[i];
        if (!isPtr(a.type)) continue;
        o << "buf " << i << " " << (a.type == Type::PtrF32 ? "f32" : "i32") << " " << a.buf.size();
        for (uint32_t w : a.buf) { std::snprintf(buf, sizeof buf, " %08x", w); o << buf; }
        o << "\n";
    }
    return o.str();
}

}  // namespace kernc
