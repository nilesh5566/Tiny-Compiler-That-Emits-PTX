// passes.cpp - IR optimisation passes.
//
// Every pass is block-local or uses conservative whole-function facts, and
// every pass must cope with non-SSA registers (a register may be defined
// many times). The `mutant(n)` hooks are deliberate bugs used only for
// mutation testing; they are inactive unless KERNC_MUTATE=n is set.
#include "passes.h"
#include <cmath>
#include <iterator>
#include <unordered_map>
#include "mutate.h"
#include "semantics.h"

namespace kernc {

static bool isFoldable(Op op) {
    switch (op) {
        case Op::Const: case Op::Mov: case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        case Op::Rem: case Op::Shl: case Op::Neg: case Op::Fma: case Op::Cmp: case Op::Not:
        case Op::BoolToInt: case Op::CvtI2F: case Op::CvtF2I:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------- constant folding
bool constFold(Function& f) {
    bool changed = false;
    for (auto& b : f.blocks) {
        std::unordered_map<int, uint32_t> known;  // register -> constant bits, valid at this point
        for (auto& in : b.instrs) {
            if (in.op == Op::CondBr) {
                auto it = known.find(in.srcs[0].id);
                if (it != known.end()) {
                    int t = it->second ? in.target : in.target2;
                    in = Instr{Op::Br};
                    in.target = t;
                    changed = true;
                }
                continue;
            }
            if (!in.dst.valid()) continue;
            bool allKnown = isFoldable(in.op);
            uint32_t vals[3] = {0, 0, 0};
            for (size_t i = 0; allKnown && i < in.srcs.size(); i++) {
                auto it = known.find(in.srcs[i].id);
                if (it == known.end()) allKnown = false; else vals[i] = it->second;
            }
            if (!allKnown) { known.erase(in.dst.id); continue; }
            uint32_t v;
            try {
                v = evalPure(in, vals);
            } catch (const UndefinedBehavior&) {
                known.erase(in.dst.id);  // leave UB for run time; never fold it away
                continue;
            }
            // ---- mutation hooks (deliberate bugs) ----
            if (mutant(1) && in.op == Op::Sub && in.dst.type == Type::I32)
                v = (uint32_t)wrapSub((int32_t)vals[1], (int32_t)vals[0]);
            if (mutant(2) && in.op == Op::Div && in.dst.type == Type::I32) {
                int32_t a = (int32_t)vals[0], d = (int32_t)vals[1], q = a / d;
                if ((a % d != 0) && ((a < 0) != (d < 0))) q--;
                v = (uint32_t)q;
            }
            if (mutant(11) && in.op == Op::CvtF2I) {
                float x = bits2f(vals[0]);
                if (x == x && x > -2e9f && x < 2e9f) v = (uint32_t)(int32_t)std::lround(x);
            }
            if (in.op != Op::Const) {
                VReg d = in.dst;
                in = Instr{Op::Const};
                in.dst = d;
                in.imm = v;
                changed = true;
            }
            known[in.dst.id] = v;
        }
    }
    return changed;
}

// ---------------------------------------------------------------- copy propagation
bool copyPropagate(Function& f) {
    bool changed = false;
    for (auto& b : f.blocks) {
        std::unordered_map<int, VReg> copyOf;  // dst -> src of a still-valid `dst = mov src`
        for (auto& in : b.instrs) {
            for (auto& s : in.srcs) {
                auto it = copyOf.find(s.id);
                if (it != copyOf.end()) { s = it->second; changed = true; }
            }
            if (!in.dst.valid()) continue;
            int d = in.dst.id;
            copyOf.erase(d);
            if (!mutant(3)) {  // MUTANT 3 forgets that the copy's source was overwritten
                for (auto it = copyOf.begin(); it != copyOf.end();)
                    it = (it->second.id == d) ? copyOf.erase(it) : std::next(it);
            }
            if (in.op == Op::Mov && in.srcs[0].id != d) copyOf[d] = in.srcs[0];
        }
    }
    return changed;
}

// ---------------------------------------------------------------- global copy propagation
// `d = mov s` where BOTH d and s are defined exactly once in the whole function:
// every use of d can read s instead, in any block. (Kern's scoping guarantees a
// definition dominates its uses, and a single definition can never be clobbered.)
bool globalCopyProp(Function& f) {
    std::vector<int> defs(f.regTypes.size(), 0);
    for (auto& p : f.params) defs[p.id]++;
    for (auto& b : f.blocks)
        for (auto& in : b.instrs)
            if (in.dst.valid()) defs[in.dst.id]++;
    std::vector<int> repl(f.regTypes.size(), -1);
    std::vector<VReg> target(f.regTypes.size());
    for (auto& b : f.blocks)
        for (auto& in : b.instrs)
            if (in.op == Op::Mov && defs[in.dst.id] == 1 && defs[in.srcs[0].id] == 1 && in.dst.id != in.srcs[0].id) {
                repl[in.dst.id] = in.srcs[0].id;
                target[in.dst.id] = in.srcs[0];
            }
    bool changed = false;
    for (auto& b : f.blocks)
        for (auto& in : b.instrs)
            for (auto& s : in.srcs) {
                int hops = 0;
                while (repl[s.id] >= 0 && hops++ < 64) { s = target[s.id]; changed = true; }
            }
    return changed;
}

// ---------------------------------------------------------------- CSE
namespace {
struct Avail { Op op; Cmp cmp; uint32_t imm; Type type; std::vector<int> srcs; VReg result; };
}

static bool cseEligible(const Instr& in) {
    return in.dst.valid() && isPure(in.op) && in.op != Op::Load && in.op != Op::Mov;
}

bool cse(Function& f) {
    bool changed = false;
    for (auto& b : f.blocks) {
        std::vector<Avail> avail;
        for (auto& in : b.instrs) {
            bool eligible = cseEligible(in);
            std::vector<int> ids;
            for (auto& s : in.srcs) ids.push_back(s.id);
            bool replaced = false;
            if (eligible) {
                for (auto& a : avail) {
                    if (a.op == in.op && a.cmp == in.cmp && a.imm == in.imm && a.type == in.dst.type && a.srcs == ids) {
                        VReg d = in.dst;
                        in = Instr{Op::Mov};
                        in.dst = d;
                        in.srcs = {a.result};
                        replaced = changed = true;
                        break;
                    }
                }
            }
            if (!in.dst.valid()) continue;
            int d = in.dst.id;
            for (auto it = avail.begin(); it != avail.end();) {
                bool stale = it->result.id == d;
                if (!mutant(4))  // MUTANT 4: keep entries whose operands were redefined
                    for (int s : it->srcs) stale |= (s == d);
                it = stale ? avail.erase(it) : std::next(it);
            }
            bool selfRef = false;
            for (int s : ids) selfRef |= (s == d);
            if (eligible && !replaced && !selfRef)
                avail.push_back({in.op, in.cmp, in.imm, in.dst.type, ids, in.dst});
        }
    }
    return changed;
}

// ---------------------------------------------------------------- strength reduction
bool strengthReduce(Function& f) {
    bool changed = false;
    for (auto& b : f.blocks) {
        std::unordered_map<int, uint32_t> konst;
        for (auto& in : b.instrs) {
            if (in.op == Op::Mul && in.dst.type == Type::I32) {
                for (int side = 0; side < 2; side++) {
                    auto it = konst.find(in.srcs[side].id);
                    if (it == konst.end()) continue;
                    uint32_t c = it->second;
                    VReg other = in.srcs[1 - side], d = in.dst;
                    if (mutant(5) && (int32_t)c < -1 && ((0u - c) & (0u - c - 1)) == 0) c = 0u - c;  // MUTANT 5
                    if (c == 1) {
                        in = Instr{Op::Mov}; in.dst = d; in.srcs = {other};
                        changed = true;
                    } else if (c > 1 && (c & (c - 1)) == 0) {
                        uint32_t k = 0;
                        while ((1u << k) != c) k++;
                        in = Instr{Op::Shl}; in.dst = d; in.srcs = {other}; in.imm = k;
                        changed = true;
                    }
                    break;
                }
            }
            if (!in.dst.valid()) continue;
            if (in.op == Op::Const) konst[in.dst.id] = in.imm;
            else konst.erase(in.dst.id);
        }
    }
    return changed;
}

// ---------------------------------------------------------------- dead code elimination
bool deadCodeElim(Function& f) {
    bool changedAny = false;
    for (;;) {
        std::vector<int> uses(f.regTypes.size(), 0);
        for (auto& b : f.blocks)
            for (auto& in : b.instrs) {
                if (mutant(6) && isTerminator(in.op)) continue;  // MUTANT 6
                for (auto& s : in.srcs) uses[s.id]++;
            }
        bool changed = false;
        for (auto& b : f.blocks) {
            auto& v = b.instrs;
            size_t w = 0;
            for (size_t r = 0; r < v.size(); r++) {
                const Instr& in = v[r];
                if (isPure(in.op) && in.dst.valid() && uses[in.dst.id] == 0) { changed = true; continue; }
                if (w != r) v[w] = std::move(v[r]);
                w++;
            }
            v.resize(w);
        }
        if (!changed) return changedAny;
        changedAny = true;
    }
}

// ---------------------------------------------------------------- unreachable blocks
static bool removeUnreachable(Function& f) {
    std::vector<int> newIndex(f.blocks.size(), -1), work = {0};
    std::vector<bool> seen(f.blocks.size(), false);
    seen[0] = true;
    while (!work.empty()) {
        int bi = work.back(); work.pop_back();
        for (auto& in : f.blocks[bi].instrs)
            for (int t : {in.target, in.target2})
                if (t >= 0 && !seen[t]) { seen[t] = true; work.push_back(t); }
    }
    std::vector<BasicBlock> kept;
    for (size_t i = 0; i < f.blocks.size(); i++)
        if (seen[i]) { newIndex[i] = (int)kept.size(); kept.push_back(std::move(f.blocks[i])); }
    if (kept.size() == f.blocks.size()) { f.blocks = std::move(kept); return false; }
    for (auto& b : kept)
        for (auto& in : b.instrs) {
            if (in.target >= 0) in.target = newIndex[in.target];
            if (in.target2 >= 0) in.target2 = newIndex[in.target2];
        }
    f.blocks = std::move(kept);
    return true;
}

// ---------------------------------------------------------------- block layout
// Reverse post-order layout, the same shape nvcc produces for structured code:
// the "true" successor is placed right after its branch, and the only backward
// branches are loop back-edges. Our creation-order layout also had backward jumps
// into join blocks (e.g. `||` inside an `if`), and fuzzing showed that ptxas 12.9
// can segfault on such layouts (docs/findings/ptxas-crash). Unreachable blocks are dropped.
void layoutBlocks(Function& f) {
    size_t n = f.blocks.size();
    std::vector<int> post;
    std::vector<char> state(n, 0);  // 0 new, 1 on stack, 2 done
    std::vector<std::pair<int, int>> stack = {{0, 0}};
    state[0] = 1;
    auto succ = [&](int b) {
        std::vector<int> s;
        const auto& in = f.blocks[b].instrs;
        if (!in.empty()) {
            const Instr& t = in.back();
            if (t.op == Op::Br) s = {t.target};
            if (t.op == Op::CondBr) s = {t.target2, t.target};  // visit false first -> true lands next in RPO
        }
        return s;
    };
    while (!stack.empty()) {
        auto& [b, i] = stack.back();
        auto s = succ(b);
        if (i < (int)s.size()) {
            int next = s[i++];
            if (state[next] == 0) { state[next] = 1; stack.push_back({next, 0}); }
        } else {
            state[b] = 2;
            post.push_back(b);
            stack.pop_back();
        }
    }
    std::vector<int> order(post.rbegin(), post.rend()), newIndex(n, -1);
    for (size_t i = 0; i < order.size(); i++) newIndex[order[i]] = (int)i;
    std::vector<BasicBlock> blocks;
    for (int b : order) blocks.push_back(std::move(f.blocks[b]));
    for (auto& b : blocks)
        for (auto& in : b.instrs) {
            if (in.target >= 0) in.target = newIndex[in.target];
            if (in.target2 >= 0) in.target2 = newIndex[in.target2];
        }
    f.blocks = std::move(blocks);
}

// ---------------------------------------------------------------- loop-invariant code motion
static std::vector<int> successors(const BasicBlock& b) {
    if (b.instrs.empty()) return {};
    const Instr& t = b.instrs.back();
    if (t.op == Op::Br) return {t.target};
    if (t.op == Op::CondBr) return {t.target, t.target2};
    return {};
}

// Hoists pure computations whose operands do not change inside a loop into the
// loop's preheader. Only instructions that can never trap are hoisted: the loop
// may run zero times, so hoisting `a / b` (i32) could create undefined behaviour
// that the original program never executed. Loads are not hoisted (a store in
// the loop might change memory).
bool licm(Function& f) {
    size_t n = f.blocks.size();
    std::vector<std::vector<int>> preds(n);
    for (size_t b = 0; b < n; b++)
        for (int s : successors(f.blocks[b])) preds[s].push_back((int)b);

    // dominator sets (iterative; functions are small)
    std::vector<std::vector<bool>> dom(n, std::vector<bool>(n, true));
    dom[0].assign(n, false);
    dom[0][0] = true;
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t b = 1; b < n; b++) {
            std::vector<bool> d(n, !preds[b].empty());
            for (int p : preds[b])
                for (size_t i = 0; i < n; i++) d[i] = d[i] && dom[p][i];
            d[b] = true;
            if (d != dom[b]) { dom[b] = d; changed = true; }
        }
    }

    std::vector<int> defs(f.regTypes.size(), 0);
    for (auto& p : f.params) defs[p.id]++;
    for (auto& b : f.blocks)
        for (auto& in : b.instrs)
            if (in.dst.valid()) defs[in.dst.id]++;

    bool changed = false;
    for (size_t t = 0; t < n; t++)
        for (int h : successors(f.blocks[t])) {
            if (!dom[t][h]) continue;  // not a back edge
            // natural loop of back edge t -> h
            std::vector<bool> inLoop(n, false);
            inLoop[h] = true;
            std::vector<int> work;
            if (!inLoop[t]) { inLoop[t] = true; work.push_back((int)t); }
            while (!work.empty()) {
                int b = work.back(); work.pop_back();
                for (int p : preds[b]) if (!inLoop[p]) { inLoop[p] = true; work.push_back(p); }
            }
            // preheader: the single outside predecessor, ending in an unconditional branch to h
            int pre = -1, outside = 0;
            for (int p : preds[h]) if (!inLoop[p]) { pre = p; outside++; }
            if (outside != 1 || f.blocks[pre].instrs.back().op != Op::Br) continue;

            std::vector<bool> definedInLoop(f.regTypes.size(), false);
            for (size_t b = 0; b < n; b++)
                if (inLoop[b])
                    for (auto& in : f.blocks[b].instrs)
                        if (in.dst.valid()) definedInLoop[in.dst.id] = true;

            for (bool moved = true; moved;) {
                moved = false;
                for (size_t b = 0; b < n; b++) {
                    if (!inLoop[b]) continue;
                    auto& v = f.blocks[b].instrs;
                    for (size_t i = 0; i < v.size(); i++) {
                        Instr& in = v[i];
                        if (!in.dst.valid() || !isPure(in.op) || in.op == Op::Load) continue;
                        bool traps = (in.op == Op::Div || in.op == Op::Rem) && in.dst.type == Type::I32;
                        if (traps && !mutant(16)) continue;  // MUTANT 16: hoist possibly-trapping division
                        if (defs[in.dst.id] != 1) continue;
                        bool invariant = true;
                        for (auto& s : in.srcs) invariant = invariant && !definedInLoop[s.id];
                        if (!invariant) continue;
                        auto& pv = f.blocks[pre].instrs;
                        pv.insert(pv.end() - 1, in);   // before the preheader's branch
                        definedInLoop[in.dst.id] = false;
                        v.erase(v.begin() + i);
                        moved = changed = true;
                        break;
                    }
                }
            }
        }
    return changed;
}

// ---------------------------------------------------------------- FMA contraction
bool fmaContract(Function& f) {
    std::vector<int> defs(f.regTypes.size(), 0), uses(f.regTypes.size(), 0);
    for (auto& b : f.blocks)
        for (auto& in : b.instrs) {
            if (in.dst.valid()) defs[in.dst.id]++;
            for (auto& s : in.srcs) uses[s.id]++;
        }
    for (auto& p : f.params) defs[p.id]++;
    bool changed = false;
    for (auto& b : f.blocks) {
        auto& v = b.instrs;
        for (size_t j = 0; j < v.size(); j++) {
            if (v[j].op != Op::Add || v[j].dst.type != Type::F32) continue;
            for (int side = 0; side < 2; side++) {
                int t = v[j].srcs[side].id;
                if (defs[t] != 1 || uses[t] != 1) continue;
                size_t i = j;
                while (i > 0 && !(v[i - 1].dst.valid() && v[i - 1].dst.id == t)) i--;
                if (i == 0) continue;  // defined in another block
                const Instr& mul = v[i - 1];
                if (mul.op != Op::Mul || mul.dst.type != Type::F32) continue;
                bool clobbered = false;  // mul operands redefined between mul and add?
                for (size_t k = i; k < j; k++)
                    if (v[k].dst.valid() && (v[k].dst.id == mul.srcs[0].id || v[k].dst.id == mul.srcs[1].id))
                        clobbered = true;
                if (clobbered) continue;
                VReg d = v[j].dst, c = v[j].srcs[1 - side];
                Instr fma{Op::Fma};
                fma.dst = d;
                fma.srcs = {mul.srcs[0], mul.srcs[1], c};
                v[j] = fma;
                changed = true;
                break;
            }
        }
    }
    return changed;
}

// ---------------------------------------------------------------- pipeline
std::vector<PassStats> optimize(Function& f, int level, bool fma) {
    std::vector<PassStats> stats;
    auto run = [&](const char* name, bool (*pass)(Function&)) {
        size_t before = countInstrs(f);
        bool c = pass(f);
        if (c) stats.push_back({name, before, countInstrs(f)});
        return c;
    };
    if (level >= 1) {
        for (int iter = 0; iter < 4; iter++) {
            bool c = false;
            c |= run("constfold", constFold);
            c |= run("copyprop", copyPropagate);
            c |= run("globalcopy", globalCopyProp);
            c |= run("cse", cse);
            c |= run("strength", strengthReduce);
            c |= run("constfold", constFold);
            c |= run("licm", licm);
            c |= run("dce", deadCodeElim);
            c |= run("unreachable", removeUnreachable);
            if (!c) break;
        }
    }
    if (fma) {
        run("fma", fmaContract);
        run("dce", deadCodeElim);
    }
    return stats;
}

}  // namespace kernc
