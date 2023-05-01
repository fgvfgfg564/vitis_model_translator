import ply.yacc as yacc
from .tndast import *

from .lexer import tokens


def p_program_iter(p):
    "program : program expr SEMICO"
    p[1].append(p[2])
    p[0] = p[1]


def p_program_end(p):
    "program : expr SEMICO"
    p[0] = Program(p[1])


def p_expr_define(p):
    """expr : INPUT varlist
    | OUTPUT varlist
    | PLACEHOLDER varlist"""
    p[0] = VarDefinition(p[1], p[2])


def p_expr_module(p):
    "expr : MODULE ID LPAREN NUMBER COMMA NUMBER RPAREN"
    p[0] = ModuleDefinition(p[2], p[4], p[6])


def p_expr_calib(p):
    "expr : CALIB ID LPAREN varlist RPAREN"
    p[0] = Calib(p[2], p[4])


def p_expr_split(p):
    "expr : varlist ASSIGN SPLIT LPAREN ID RPAREN"
    p[0] = Split(p[1], p[5])

def p_expr_pyexpr(p):
    "expr : varlist ASSIGN PYEXPR LPAREN varlist COMMA STR RPAREN"
    p[0] = Pyexpr(p[1], p[5], p[7])

def p_expr_assign(p):
    "expr : varlist ASSIGN ID LPAREN varlist RPAREN"
    p[0] = Assign(p[1], p[3], p[5])


def p_varlist_iter(p):
    """varlist : varlist COMMA ID"""
    p[1].append(p[3])
    p[0] = p[1]


def p_varlist_end(p):
    "varlist : ID"
    p[0] = [p[1]]


def p_error(p):
    print(f"Syntax Error on line: {p.lineno}")


parser = yacc.yacc()
