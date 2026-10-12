# The Kern language — specification (v1.0)

Kern is a small, statically typed language for writing GPU kernels. It is
deliberately tiny, but every rule below is **precise**, because the compiler is
verified against this document: the AST interpreter (`runAst`) is a direct
transcription of these rules, and every other execution engine — the IR
interpreter at each optimisation level and the real GPU — must agree with it
bit-for-bit.

## 1. Lexical structure

* Whitespace: space, tab, CR, LF. Comments: `// to end of line` and `/* block */` (not nested).
* Identifiers: `[A-Za-z_][A-Za-z0-9_]*`, excluding keywords.
* Keywords: `kernel let if else while return true false i32 f32 bool`.
* Integer literals: decimal digits, value must be ≤ 2147483647. The single
  exception is `-2147483648`, which is recognised as one literal (INT32_MIN).
* Float literals: `digits.digits` with optional exponent (`1.5`, `2.0e-3`, `1e10`).
  A literal is rounded once, to nearest-even, to the closest `f32`.
* Operators: `+ - * / % = == != < <= > >= && || !`. There are no bitwise
  operators; `&` and `|` are lexical errors.

## 2. Grammar (EBNF)

```
program  := kernel+
kernel   := 'kernel' IDENT '(' [ param { ',' param } ] ')' block
param    := type IDENT
type     := ('i32' | 'f32' | 'bool') [ '*' ]          (bool* is rejected)
block    := '{' { stmt } '}'
stmt     := 'let' IDENT [ ':' type ] '=' expr ';'
          | IDENT '=' expr ';'
          | IDENT '[' expr ']' '=' expr ';'
          | 'if' '(' expr ')' block [ 'else' ( block | if-stmt ) ]
          | 'while' '(' expr ')' block
          | 'return' ';'
expr     := binary expression over the precedence table below
unary    := ( '-' | '!' ) unary | primary
primary  := INT | FLOAT | 'true' | 'false' | IDENT | IDENT '[' expr ']'
          | IDENT '(' [ expr { ',' expr } ] ')' | ('i32'|'f32') '(' expr ')'
          | '(' expr ')'
```

Precedence, lowest to highest (all binary operators are left-associative,
except that comparisons may not be chained: `a < b < c` is an error):

| level | operators |
|---|---|
| 1 | `\|\|` |
| 2 | `&&` |
| 3 | `==` `!=` |
| 4 | `<` `<=` `>` `>=` |
| 5 | `+` `-` |
| 6 | `*` `/` `%` |
| 7 | unary `-` `!` |

Expressions are not statements: `gid();` is a syntax error.

## 3. Types

| type | meaning |
|---|---|
| `i32` | 32-bit two's-complement integer |
| `f32` | IEEE-754 binary32 |
| `bool` | true / false |
| `i32*`, `f32*` | pointer to global device memory — **kernel parameters only** |

* There are **no implicit conversions**. `let x: f32 = 0;` is an error; write `0.0`.
* Pointers cannot be copied into locals, assigned, compared or used in arithmetic;
  they can only be indexed.
* `bool` cannot be a kernel parameter (PTX has no predicate parameters).

## 4. Names and scopes

* Every `block` opens a scope. A name must be declared (`let` or parameter) before use.
* Redeclaring a name in the same scope is an error; shadowing an outer name is allowed.
* Kernel names must be unique in a program; parameter names unique per kernel.
* `let` without a type annotation takes the type of its initialiser.

## 5. Semantics

### 5.1 Integers (`i32`)

* `+ - *` and unary `-` **wrap around** (arithmetic modulo 2³²). This is
  *defined* behaviour, unlike signed overflow in C.
* `/` truncates toward zero; `%` has the sign of the dividend (`-7 / 2 == -3`, `-7 % 2 == -1`).
* **Undefined:** `x / 0`, `x % 0`, `INT32_MIN / -1`, `INT32_MIN % -1`.

### 5.2 Floats (`f32`)

* `+ - * /` are single IEEE-754 operations, rounded to nearest-even.
  Subnormals are preserved (no flush-to-zero). Division by zero gives ±inf or NaN.
* Each operation is rounded separately: the compiler **may not** fuse `a*b+c`
  into an FMA, except when invoked with `--fma`, which explicitly opts out of
  this rule (results may then differ in the last bits).
* Unary `-` flips the sign bit: `-(0.0)` is `-0.0`.
* Comparisons are IEEE: every comparison with NaN is false, **except** `!=`,
  which is true. `-0.0 == 0.0` is true.
* NaN payloads are not specified; any NaN equals any other NaN for testing purposes.

### 5.3 Booleans

* `&&` and `||` **short-circuit**: the right operand is evaluated only if needed.
  `i < n && a[i] > 0` therefore never reads `a[i]` when `i >= n`.
* `==` / `!=` are defined on `bool`; ordering comparisons are not.

### 5.4 Conversions

| expression | result |
|---|---|
| `f32(i)` | nearest `f32`, ties to even (`f32(16777217) == 16777216.0`) |
| `i32(f)` | truncate toward zero; **saturate** to INT32_MIN/MAX when out of range; NaN → 0 |
| `i32(b)` | 1 or 0 |
| `f32(f)`, `i32(i)` | identity |

`i32(f)` is total — it never has undefined behaviour (this matches PTX `cvt.rzi.s32.f32`).

### 5.5 Memory

* `p[i]` reads/writes element `i` of the buffer passed for `p` (4-byte elements).
* **Undefined:** an index outside the buffer.
* **Undefined:** a data race — two threads accessing the same element when at least
  one writes. (There is no shared memory or barrier, so threads are independent.)

### 5.6 Threads

| builtin | meaning | PTX |
|---|---|---|
| `tid()` | thread index in block | `%tid.x` |
| `bid()` | block index | `%ctaid.x` |
| `bdim()` | threads per block | `%ntid.x` |
| `gdim()` | blocks in grid | `%nctaid.x` |
| `gid()` | `bid() * bdim() + tid()` | `mad.lo.s32` |

Launches are one-dimensional. A kernel ends at `return;` or the end of its body.

### 5.7 Undefined behaviour

If a program performs an undefined operation, the whole run has no defined
result: the reference interpreters report `status ub`, and test harnesses
accept any outcome from other engines. A non-terminating loop is treated the
same way (the interpreters stop after 2×10⁸ steps).

## 6. Mapping to PTX (informative)

| Kern | PTX |
|---|---|
| `i32` / `f32` / `bool` / pointer | `%r` (.b32) / `%f` (.f32) / `%p` (.pred) / `%rd` (.b64) |
| `a + b` (f32) | `add.rn.f32` (explicit `.rn` forbids ptxas from forming FMA) |
| `a / b` (f32) | `div.rn.f32` (IEEE-correct division) |
| `a != b` (f32) | `setp.neu.f32` (unordered: true for NaN) |
| `p[i]` | `mul.wide.s32` + `add.s64` + `ld.global` / `st.global` |
| pointer params | `ld.param.u64` + `cvta.to.global.u64` |
