// parser.h - recursive-descent parser with precedence climbing for expressions.
#pragma once
#include <vector>
#include "ast.h"
#include "lexer.h"

namespace kernc {

class Parser {
public:
    explicit Parser(std::vector<Token> toks) : toks_(std::move(toks)) {}
    Program parseProgram();

private:
    std::vector<Token> toks_;
    size_t pos_ = 0;

    const Token& peek(size_t k = 0) const;
    const Token& advance();
    bool check(Tok k) const { return peek().kind == k; }
    bool match(Tok k);
    const Token& expect(Tok k, const char* context);
    [[noreturn]] void error(const Token& t, const std::string& msg);

    Kernel parseKernel();
    Type parseType();
    Block parseBlock();
    StmtPtr parseStmt();
    ExprPtr parseExpr(int minPrec = 1);
    ExprPtr parseUnary();
    ExprPtr parsePrimary();
};

Program parseSource(const std::string& src);  // lex + parse

}  // namespace kernc
