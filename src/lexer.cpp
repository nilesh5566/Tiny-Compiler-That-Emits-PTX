// lexer.cpp
#include "lexer.h"
#include <cctype>
#include <unordered_map>

namespace kernc {

const char* tokName(Tok t) {
    switch (t) {
        case Tok::Kernel: return "'kernel'"; case Tok::Let: return "'let'";
        case Tok::If: return "'if'"; case Tok::Else: return "'else'";
        case Tok::While: return "'while'"; case Tok::Return: return "'return'";
        case Tok::True: return "'true'"; case Tok::False: return "'false'";
        case Tok::KwI32: return "'i32'"; case Tok::KwF32: return "'f32'";
        case Tok::KwBool: return "'bool'";
        case Tok::Ident: return "identifier"; case Tok::IntLit: return "integer literal";
        case Tok::FloatLit: return "float literal";
        case Tok::LParen: return "'('"; case Tok::RParen: return "')'";
        case Tok::LBrace: return "'{'"; case Tok::RBrace: return "'}'";
        case Tok::LBracket: return "'['"; case Tok::RBracket: return "']'";
        case Tok::Comma: return "','"; case Tok::Semi: return "';'"; case Tok::Colon: return "':'";
        case Tok::Plus: return "'+'"; case Tok::Minus: return "'-'"; case Tok::Star: return "'*'";
        case Tok::Slash: return "'/'"; case Tok::Percent: return "'%'"; case Tok::Assign: return "'='";
        case Tok::Eq: return "'=='"; case Tok::Ne: return "'!='"; case Tok::Lt: return "'<'";
        case Tok::Le: return "'<='"; case Tok::Gt: return "'>'"; case Tok::Ge: return "'>='";
        case Tok::AndAnd: return "'&&'"; case Tok::OrOr: return "'||'"; case Tok::Bang: return "'!'";
        case Tok::Eof: return "end of file";
    }
    return "?";
}

char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') { line_++; col_ = 1; } else { col_++; }
    return c;
}

void Lexer::skipSpaceAndComments() {
    for (;;) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { advance(); continue; }
        if (c == '/' && peek(1) == '/') {
            while (peek() != '\n' && peek() != '\0') advance();
            continue;
        }
        if (c == '/' && peek(1) == '*') {
            int l = line_, co = col_;
            advance(); advance();
            while (!(peek() == '*' && peek(1) == '/')) {
                if (peek() == '\0') throw CompileError(l, co, "unterminated block comment");
                advance();
            }
            advance(); advance();
            continue;
        }
        return;
    }
}

Token Lexer::number(int line, int col) {
    std::string s;
    bool isFloat = false;
    while (std::isdigit((unsigned char)peek())) s += advance();
    if (peek() == '.' ) {
        if (!std::isdigit((unsigned char)peek(1)))
            throw CompileError(line_, col_, "expected digit after '.' in float literal");
        isFloat = true;
        s += advance();
        while (std::isdigit((unsigned char)peek())) s += advance();
    }
    if (peek() == 'e' || peek() == 'E') {
        isFloat = true;
        s += advance();
        if (peek() == '+' || peek() == '-') s += advance();
        if (!std::isdigit((unsigned char)peek()))
            throw CompileError(line_, col_, "expected exponent digits in float literal");
        while (std::isdigit((unsigned char)peek())) s += advance();
    }
    if (std::isalpha((unsigned char)peek()) || peek() == '_')
        throw CompileError(line_, col_, "invalid character in number literal");
    return {isFloat ? Tok::FloatLit : Tok::IntLit, s, line, col};
}

Token Lexer::identOrKeyword(int line, int col) {
    static const std::unordered_map<std::string, Tok> kw = {
        {"kernel", Tok::Kernel}, {"let", Tok::Let}, {"if", Tok::If}, {"else", Tok::Else},
        {"while", Tok::While}, {"return", Tok::Return}, {"true", Tok::True},
        {"false", Tok::False}, {"i32", Tok::KwI32}, {"f32", Tok::KwF32}, {"bool", Tok::KwBool},
    };
    std::string s;
    while (std::isalnum((unsigned char)peek()) || peek() == '_') s += advance();
    auto it = kw.find(s);
    return {it == kw.end() ? Tok::Ident : it->second, s, line, col};
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> out;
    for (;;) {
        skipSpaceAndComments();
        int l = line_, c = col_;
        char ch = peek();
        if (ch == '\0') { out.push_back({Tok::Eof, "", l, c}); return out; }
        if (std::isdigit((unsigned char)ch)) { out.push_back(number(l, c)); continue; }
        if (std::isalpha((unsigned char)ch) || ch == '_') { out.push_back(identOrKeyword(l, c)); continue; }

        auto two = [&](char next, Tok yes, Tok no) {
            advance();
            if (peek() == next) { advance(); return yes; }
            return no;
        };
        Tok k;
        switch (ch) {
            case '(': advance(); k = Tok::LParen; break;
            case ')': advance(); k = Tok::RParen; break;
            case '{': advance(); k = Tok::LBrace; break;
            case '}': advance(); k = Tok::RBrace; break;
            case '[': advance(); k = Tok::LBracket; break;
            case ']': advance(); k = Tok::RBracket; break;
            case ',': advance(); k = Tok::Comma; break;
            case ';': advance(); k = Tok::Semi; break;
            case ':': advance(); k = Tok::Colon; break;
            case '+': advance(); k = Tok::Plus; break;
            case '-': advance(); k = Tok::Minus; break;
            case '*': advance(); k = Tok::Star; break;
            case '/': advance(); k = Tok::Slash; break;
            case '%': advance(); k = Tok::Percent; break;
            case '=': k = two('=', Tok::Eq, Tok::Assign); break;
            case '!': k = two('=', Tok::Ne, Tok::Bang); break;
            case '<': k = two('=', Tok::Le, Tok::Lt); break;
            case '>': k = two('=', Tok::Ge, Tok::Gt); break;
            case '&':
                advance();
                if (peek() != '&') throw CompileError(l, c, "expected '&&' (bitwise '&' is not supported)");
                advance(); k = Tok::AndAnd; break;
            case '|':
                advance();
                if (peek() != '|') throw CompileError(l, c, "expected '||' (bitwise '|' is not supported)");
                advance(); k = Tok::OrOr; break;
            default:
                throw CompileError(l, c, std::string("unexpected character '") + ch + "'");
        }
        out.push_back({k, src_.substr(0, 0), l, c});
        out.back().text = tokName(k);
    }
}

}  // namespace kernc
