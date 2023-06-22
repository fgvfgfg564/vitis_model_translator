import ply.lex as lex
import logging

__all__ = (
    "tokens",
    "lexer",
)

reserved = {
    "input": "INPUT",
    "output": "OUTPUT",
    "placeholder": "PLACEHOLDER",
    "module": "MODULE",
    "calib": "CALIB",
    "split": "SPLIT",
    "pyexpr": "PYEXPR", 
}

tokens = ["ID", "COMMA", "LPAREN", "RPAREN", "ASSIGN", "SEMICO", "NUMBER", "STR"] + list(
    reserved.values()
)

t_COMMA = r","
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_ASSIGN = r"="
t_SEMICO = r";"


t_ignore_SPACE = "[ \t]"
t_ignore_COMMENT = r"\#[^\n]*"


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_STR(t):
    r'"[^\n"]+"'
    t.value = t.value[1:-1]
    return t

def t_ID(t):
    r"[A-Za-z_]+\w*"
    t.type = reserved.get(t.value, "ID")
    return t


def t_NUMBER(t):
    r"\d+"
    t.value = int(t.value)
    return t


def t_error(t):
    logging.warn("Illegal character '%s' at line %d" % (t.value[0], t.lexer.lineno))
    t.lexer.skip(1)


lexer = lex.lex()
