"""Restricted execution of model-written arithmetic programs.

The program must define a variable `answer`. Before execution the AST is
checked against a whitelist: no imports except `math`, no attribute access
except on `math`, no dunder names, no calls except a fixed set of numeric
builtins and math functions. The program then runs in a separate isolated
interpreter (`python -I`) with a wall-clock timeout. Anything else is
rejected without being run.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import Optional

# print is allowed but replaced by a no-op inside the sandbox.
ALLOWED_CALLS = {"abs", "round", "min", "max", "sum", "int", "float", "len", "range", "divmod", "pow", "print"}
ALLOWED_NODES = (
    ast.Module, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.If, ast.For,
    ast.While, ast.Break, ast.Continue, ast.Call, ast.keyword, ast.List, ast.Tuple, ast.Subscript,
    ast.Slice, ast.Attribute, ast.Import, ast.alias, ast.ListComp, ast.comprehension, ast.GeneratorExp,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context, ast.Pass,
)


class UnsafeProgram(ValueError):
    pass


def check_program(source: str) -> ast.Module:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise UnsafeProgram(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeProgram(f"node not allowed: {type(node).__name__}")
        if isinstance(node, ast.Import):
            if any(a.name != "math" or a.asname not in (None, "math") for a in node.names):
                raise UnsafeProgram("only 'import math' is allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeProgram("dunder names are not allowed")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math") or node.attr.startswith("_"):
                raise UnsafeProgram("attribute access is limited to math functions")
        if isinstance(node, ast.Call):
            f = node.func
            ok = (isinstance(f, ast.Name) and f.id in ALLOWED_CALLS) or \
                 (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "math")
            if not ok:
                raise UnsafeProgram("call not allowed")
    if not any(isinstance(n, ast.Name) and n.id == "answer" and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)):
        raise UnsafeProgram("program must assign 'answer'")
    return tree


_RUNNER = (
    "import json,math,sys\n"
    "src=sys.stdin.read()\n"
    "g={'__builtins__':{k:__builtins__[k] if isinstance(__builtins__,dict) else getattr(__builtins__,k) "
    "for k in ('abs','round','min','max','sum','int','float','len','range','divmod','pow','__import__')},'math':math}\n"
    "g['__builtins__']['print']=lambda *a,**k:None\n"
    "exec(compile(src,'<program>','exec'),g)\n"
    "a=g.get('answer')\n"
    "print(json.dumps(float(a)))\n"
)


def run_program(source: str, timeout: float = 3.0) -> Optional[float]:
    """Return the numeric `answer`, or None if rejected, failing or timing out."""
    try:
        check_program(source)
    except UnsafeProgram:
        return None
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=source, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        value = float(json.loads(proc.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return None
    return value if value == value and abs(value) < 1e15 else None
