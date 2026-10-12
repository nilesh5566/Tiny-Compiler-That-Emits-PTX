"""gen.py - random program generator for differential fuzzing (Csmith-style, for Kern).

Every generated program is
  * well-typed by construction (types are tracked while generating), and
  * free of undefined behaviour:
      - integer divisors are forced into [1, 37]:   a / ((b % 7) * (b % 7) + 1)
      - array indices are wrapped into [0, n):     ia[((e % n) + n) % n]
      - loops are bounded counters the body never assigns
      - each thread writes only its own output slot  (no data races)
So any disagreement between the engines is a compiler (or oracle) bug,
never a "the program was undefined anyway" false alarm.

    python tools/gen.py --seed 42            # print one program + its launch
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerntools as kt  # noqa: E402

INT_LITS = ["0", "1", "2", "3", "7", "31", "-1", "-5", "-8", "16", "100", "65536", "2147483647", "-2147483648", "46341"]
FLT_LITS = ["0.0", "1.0", "0.5", "2.0", "3.25", "0.1", "100.0", "1.0e30", "1.0e-30", "1.0e-40", "-0.0",
            "16777216.0", "3.0e38"]
SIGNATURE = "kernel fuzz(f32* fa, i32* ia, f32* fo, i32* io, f32 fs, i32 is, i32 n)"


class Gen:
    def __init__(self, seed: int, max_stmts: int = 14):
        self.r = random.Random(seed)
        self.scopes = [[]]          # list of (name, type, assignable)
        self.counter = 0
        self.budget = max_stmts
        self.loop_depth = 0
        self.lines = []

    # ---------------------------------------------------------------- helpers
    def chance(self, p):
        return self.r.random() < p

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def vars_of(self, t, assignable=False):
        return [n for s in self.scopes for (n, ty, a) in s if ty == t and (a or not assignable)]

    def emit(self, depth, text):
        self.lines.append("    " * depth + text)

    # ---------------------------------------------------------------- expressions
    def expr(self, t, d=0):
        leaf = d >= 3 or self.chance(0.3 + 0.15 * d)
        return self.leaf(t) if leaf else self.compound(t, d)

    def idx(self, d):
        return f"((({self.expr('i32', d + 1)}) % n + n) % n)"

    def leaf(self, t):
        r = self.r
        if t == "i32":
            opts = [lambda: r.choice(INT_LITS), lambda: f"ia[{self.idx(2)}]", lambda: "is", lambda: "n",
                    lambda: r.choice(["gid()", "tid()", "bid()", "bdim()", "gdim()"])]
            vs = self.vars_of("i32")
            if vs:
                opts += [lambda: r.choice(vs)] * 3
        elif t == "f32":
            opts = [lambda: r.choice(FLT_LITS), lambda: f"fa[{self.idx(2)}]", lambda: "fs"]
            vs = self.vars_of("f32")
            if vs:
                opts += [lambda: r.choice(vs)] * 3
        else:
            opts = [lambda: r.choice(["true", "false"])]
            vs = self.vars_of("bool")
            if vs:
                opts += [lambda: r.choice(vs)] * 2
            opts += [lambda: self.compare(3)] * 2
        return r.choice(opts)()

    def compare(self, d):
        t = self.r.choice(["i32", "f32"])
        op = self.r.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"({self.expr(t, d + 1)} {op} {self.expr(t, d + 1)})"

    def compound(self, t, d):
        r = self.r
        if t == "i32":
            k = r.randrange(11)
            if k == 10:  # multiply by (+-) a power of two: exercises strength reduction
                return f"({self.expr('i32', d + 1)} * {r.choice(['4', '8', '-4', '-8', '16', '-16', '1024', '-2147483648'])})"
            if k == 9:  # constant sub-expression: exercises the constant folder
                lit = lambda: r.choice(INT_LITS + ["-7", "-9", "13"])  # noqa: E731
                op = r.choice(["+", "-", "*", "/", "%"])
                rhs = r.choice(["2", "3", "7", "-3", "-2"]) if op in "/%" else lit()
                return f"({lit()} {op} {rhs})"
            a = lambda: self.expr("i32", d + 1)  # noqa: E731
            if k <= 2:
                return f"({a()} {r.choice(['+', '-', '*'])} {a()})"
            if k == 3:
                b = a()
                return f"({a()} / ((({b}) % 7) * (({b}) % 7) + 1))"
            if k == 4:
                b = a()
                return f"({a()} % ((({b}) % 7) * (({b}) % 7) + 1))"
            if k == 5:
                return f"(-{a()})"
            if k == 6:
                return f"i32({self.expr('f32', d + 1)})"
            if k == 7:
                return f"i32({self.expr('bool', d + 1)})"
            return f"ia[{self.idx(d)}]"
        if t == "f32":
            k = r.randrange(8)
            if k == 7:  # constant sub-expression
                return f"({r.choice(FLT_LITS)} {r.choice(['+', '-', '*', '/'])} {r.choice(FLT_LITS)})"
            a = lambda: self.expr("f32", d + 1)  # noqa: E731
            if k <= 3:
                return f"({a()} {r.choice(['+', '-', '*', '/'])} {a()})"
            if k == 4:
                return f"(-{a()})"
            if k == 5:
                return f"f32({self.expr('i32', d + 1)})"
            return f"fa[{self.idx(d)}]"
        k = r.randrange(4)
        if k == 0:
            return self.compare(d)
        if k == 1:
            return f"({self.expr('bool', d + 1)} && {self.expr('bool', d + 1)})"
        if k == 2:
            return f"({self.expr('bool', d + 1)} || {self.expr('bool', d + 1)})"
        return f"(!{self.expr('bool', d + 1)})"

    # ---------------------------------------------------------------- statements
    def stmts(self, depth, count):
        for _ in range(count):
            if self.budget <= 0:
                return
            self.budget -= 1
            self.stmt(depth)

    def stmt(self, depth):
        r = self.r
        k = r.random()
        assignable = {t: self.vars_of(t, assignable=True) for t in ("i32", "f32", "bool")}
        if k < 0.35 or not any(assignable.values()):
            t = r.choice(["i32", "i32", "f32", "f32", "bool"])
            name = self.fresh()
            self.emit(depth, f"let {name} = {self.expr(t)};")
            self.scopes[-1].append((name, t, True))
        elif k < 0.60:
            t = r.choice([t for t, v in assignable.items() if v])
            self.emit(depth, f"{r.choice(assignable[t])} = {self.expr(t)};")
        elif k < 0.72:
            self.emit(depth, f"if ({self.expr('bool')}) {{")
            self.block(depth + 1, r.randint(1, 3))
            if self.chance(0.5):
                self.emit(depth, "} else {")
                self.block(depth + 1, r.randint(1, 3))
            self.emit(depth, "}")
        elif k < 0.82 and self.loop_depth < 2:
            c = self.fresh("c")
            self.emit(depth, f"let {c} = 0;")
            self.scopes[-1].append((c, "i32", False))  # readable, never assigned by the body
            self.emit(depth, f"while ({c} < {r.randint(1, 4)}) {{")
            self.loop_depth += 1
            self.block(depth + 1, r.randint(1, 3))
            self.loop_depth -= 1
            self.emit(depth + 1, f"{c} = {c} + 1;")
            self.emit(depth, "}")
        elif k < 0.93:
            if self.chance(0.5):
                self.emit(depth, f"io[i] = io[i] * 31 + {self.expr('i32')};")
            else:
                self.emit(depth, f"fo[i] = {self.expr('f32')};")
        elif k < 0.96:
            self.emit(depth, f"if ({self.expr('bool')}) {{ return; }}")
        else:
            t = r.choice(["i32", "f32"])
            name = self.fresh()
            self.emit(depth, f"let {name} = {'io' if t == 'i32' else 'fo'}[i];")
            self.scopes[-1].append((name, t, True))

    def block(self, depth, count):
        self.scopes.append([])
        self.stmts(depth, count)
        self.scopes.pop()

    # ---------------------------------------------------------------- program
    def program(self) -> str:
        self.lines = [SIGNATURE + " {", "    let i = gid();", "    if (i < n) {"]
        self.scopes = [[("i", "i32", False)]]
        self.scopes.append([])
        self.stmts(2, self.budget)
        # fold every live value into the outputs so nothing is trivially dead
        for name, t, _ in self.scopes[-1]:
            if t == "i32":
                self.emit(2, f"io[i] = io[i] * 31 + {name};")
            elif t == "f32":
                self.emit(2, f"io[i] = io[i] * 31 + i32({name});")
                self.emit(2, f"fo[i] = fo[i] + {name};")
            else:
                self.emit(2, f"io[i] = io[i] * 31 + i32({name});")
        self.lines += ["    }", "}"]
        return "\n".join(self.lines) + "\n"

    def launch(self) -> kt.Launch:
        r = self.r
        grid, block = r.randint(1, 3), r.choice([1, 7, 32, 64])
        n = max(1, grid * block - r.randint(0, 3))
        spec_f = r.choice(["random(-10,10)", "random(-1e6,1e6)", "special", "range"])
        spec_i = r.choice(["random(-100,100)", "random(-2147483648,2147483647)", "special", "range"])
        fs = r.choice(kt.SPECIAL_F32 + [r.uniform(-5, 5)])
        is_ = r.choice(kt.SPECIAL_I32 + [r.randint(-50, 50)])
        return kt.Launch("fuzz", grid, block, [
            kt.Arg("f32buf", kt.gen_buffer("f32buf", n, spec_f, r)),
            kt.Arg("i32buf", kt.gen_buffer("i32buf", n, spec_i, r)),
            kt.Arg("f32buf", [0] * n),
            kt.Arg("i32buf", [0] * n),
            kt.Arg("f32", [kt.f2b(fs)]),
            kt.Arg("i32", [kt.i2b(is_)]),
            kt.Arg("i32", [kt.i2b(n)]),
        ])


def generate(seed: int, max_stmts: int = 14):
    g = Gen(seed, max_stmts)
    return g.program(), g.launch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stmts", type=int, default=14)
    a = ap.parse_args()
    prog, launch = generate(a.seed, a.stmts)
    print(prog)
    print("# launch:\n" + launch.text())
