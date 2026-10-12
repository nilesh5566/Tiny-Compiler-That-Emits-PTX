// lexer.h - turns Kern source text into tokens.
#pragma once
#include <string>
#include <vector>
#include "common.h"

namespace kernc {

enum class Tok {
    // keywords
    Kernel, Let, If, Else, While, Return, True, False, KwI32, KwF32, KwBool,
    // literals / names
    Ident, IntLit, FloatLit,
    // punctuation
    LParen, RParen, LBrace, RBrace, LBracket, RBracket, Comma, Semi, Colon,
    // operators
    Plus, Minus, Star, Slash, Percent, Assign,
    Eq, Ne, Lt, Le, Gt, Ge, AndAnd, OrOr, Bang,
    Eof
};

const char* tokName(Tok t);

struct Token {
    Tok kind;
    std::string text;
    int line, col;
};

class Lexer {
public:
    explicit Lexer(std::string src) : src_(std::move(src)) {}
    std::vector<Token> tokenize();

private:
    std::string src_;
    size_t pos_ = 0;
    int line_ = 1, col_ = 1;

    char peek(size_t k = 0) const { return pos_ + k < src_.size() ? src_[pos_ + k] : '\0'; }
    char advance();
    void skipSpaceAndComments();
    Token number(int line, int col);
    Token identOrKeyword(int line, int col);
};

}  // namespace kernc
