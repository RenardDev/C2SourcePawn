import argparse
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from clang import cindex
from clang.cindex import Cursor, CursorKind, TypeKind


def try_set_libclang() -> None:
    lib_file = os.environ.get("LIBCLANG_FILE")
    lib_path = os.environ.get("LIBCLANG_PATH")
    try:
        if lib_file and os.path.exists(lib_file):
            cindex.Config.set_library_file(lib_file)
            return
        if lib_path and os.path.isdir(lib_path):
            cindex.Config.set_library_path(lib_path)
            return
    except Exception:
        pass

    candidates = [
        r"C:\Program Files\LLVM\bin\libclang.dll",
        r"C:\Program Files (x86)\LLVM\bin\libclang.dll",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                cindex.Config.set_library_file(p)
                return
            except Exception:
                pass


@dataclass
class Config:
    char_signed: bool = False


class Out:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.ind: int = 0

    def w(self, s: str = "") -> None:
        self.lines.append(("    " * self.ind) + s)

    def get(self) -> str:
        return "\n".join(self.lines) + "\n"


def cur_tokens(cur: Cursor) -> List[cindex.Token]:
    try:
        return list(cur.get_tokens())
    except Exception:
        return []


def tok_spellings(cur: Cursor) -> List[str]:
    return [t.spelling for t in cur_tokens(cur)]


def tok_join(cur: Cursor, sep: str = " ") -> str:
    return sep.join(tok_spellings(cur))


def start_off(cur: Cursor) -> int:
    try:
        return cur.extent.start.offset
    except Exception:
        return -1


def end_off(cur: Cursor) -> int:
    try:
        return cur.extent.end.offset
    except Exception:
        return -1


def tok_start(t: cindex.Token) -> int:
    try:
        return t.extent.start.offset
    except Exception:
        return -1


def tok_end(t: cindex.Token) -> int:
    try:
        return t.extent.end.offset
    except Exception:
        return -1


def is_definition(cur: Cursor) -> bool:
    try:
        return cur.is_definition()
    except Exception:
        return False


def loc_str(cur: Cursor) -> str:
    try:
        loc = cur.location
        f = str(loc.file) if loc.file else "<unknown>"
        return f"{f}:{loc.line}:{loc.column}"
    except Exception:
        return "<unknown>:0:0"


K_UNION_DECL = getattr(CursorKind, "UNION_DECL", None)
K_SIZEOF_KIND = getattr(CursorKind, "UNARY_EXPR_OR_TYPE_TRAIT_EXPR", None)

IMPLICIT_WRAPPER_KINDS: Set[CursorKind] = {CursorKind.UNEXPOSED_EXPR}
for _name in (
    "PAREN_EXPR",
    "IMPLICIT_CAST_EXPR",
    "CXX_STATIC_CAST_EXPR",
    "CXX_REINTERPRET_CAST_EXPR",
    "CXX_CONST_CAST_EXPR",
    "CXX_DYNAMIC_CAST_EXPR",
    "CXX_FUNCTIONAL_CAST_EXPR",
):
    _k = getattr(CursorKind, _name, None)
    if _k is not None:
        IMPLICIT_WRAPPER_KINDS.add(_k)


def unwrap_expr(cur: Cursor) -> Cursor:
    while cur.kind in IMPLICIT_WRAPPER_KINDS:
        ch = list(cur.get_children())
        if len(ch) != 1:
            break
        cur = ch[0]
    return cur


_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _read_text_guess_encoding(path: str) -> str:
    try:
        data = open(path, "rb").read()
    except Exception:
        return ""

    if data.startswith(b"\xff\xfe"):
        try:
            return data.decode("utf-16-le", errors="ignore")
        except Exception:
            return ""
    if data.startswith(b"\xfe\xff"):
        try:
            return data.decode("utf-16-be", errors="ignore")
        except Exception:
            return ""

    if b"\x00" in data[:256]:
        try:
            return data.decode("utf-16-le", errors="ignore")
        except Exception:
            pass

    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return data.decode("latin-1", errors="ignore")
        except Exception:
            return ""


def load_ignore_symbols(path: str) -> Set[str]:
    text = _read_text_guess_encoding(path)
    if not text:
        return set()

    out: Set[str] = set()

    for m in re.finditer(
        r'symbol\s+already\s+defined:\s*"([^"]+)"', text, flags=re.IGNORECASE
    ):
        out.add(m.group(1))

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";") or s.startswith("//"):
            continue
        if "symbol" in s and "already" in s and "defined" in s:
            continue

        if len(s) >= 2 and (s[0] == s[-1]) and s[0] in ("'", '"'):
            s = s[1:-1].strip()

        if _ID_RE.match(s):
            out.add(s)

    return out


def canonical_kind(t: cindex.Type) -> TypeKind:
    try:
        return t.get_canonical().kind
    except Exception:
        return t.kind


def is_pointer_type(t: cindex.Type) -> bool:
    return canonical_kind(t) == TypeKind.POINTER


def is_function_type(t: cindex.Type) -> bool:
    k = canonical_kind(t)
    return k in (TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO)


def is_function_pointer_type(t: cindex.Type) -> bool:
    if canonical_kind(t) != TypeKind.POINTER:
        return False
    try:
        pt = t.get_canonical().get_pointee()
    except Exception:
        pt = t.get_pointee()
    return is_function_type(pt)


def is_const_qualified(t: cindex.Type) -> bool:
    try:
        return t.is_const_qualified()
    except Exception:
        return False


def pointee_is_const(ptr_t: cindex.Type) -> bool:
    try:
        return is_const_qualified(ptr_t.get_pointee())
    except Exception:
        return False


def is_c_bool_type(t: cindex.Type) -> bool:
    try:
        if (t.spelling or "").strip() == "bool":
            return True
    except Exception:
        pass
    return canonical_kind(t) == TypeKind.BOOL


_INT_KINDS = {
    TypeKind.CHAR_S,
    TypeKind.CHAR_U,
    TypeKind.SCHAR,
    TypeKind.UCHAR,
    TypeKind.SHORT,
    TypeKind.USHORT,
    TypeKind.INT,
    TypeKind.UINT,
    TypeKind.LONG,
    TypeKind.ULONG,
    TypeKind.LONGLONG,
    TypeKind.ULONGLONG,
}


def is_c_intish_type(t: cindex.Type) -> bool:
    k = canonical_kind(t)
    if k in _INT_KINDS:
        return True
    return k == TypeKind.ENUM


def is_c_char_scalar(t: cindex.Type) -> bool:
    k = canonical_kind(t)
    return k in (TypeKind.CHAR_S, TypeKind.CHAR_U, TypeKind.SCHAR, TypeKind.UCHAR)


def char_is_signed_for_type(t: cindex.Type, cfg: Config) -> bool:
    k = canonical_kind(t)
    if k == TypeKind.SCHAR:
        return True
    if k == TypeKind.UCHAR:
        return False
    if k in (TypeKind.CHAR_S, TypeKind.CHAR_U):
        return cfg.char_signed
    return cfg.char_signed


def sp_char_to_int(expr: str, signed: bool) -> str:
    u = f"(view_as<int>({expr}) & 0xFF)"
    if not signed:
        return u
    return f"(({u} ^ 0x80) - 0x80)"


_QUAL_PREFIX = ("const ", "volatile ", "restrict ")
_TAG_PREFIX = ("struct ", "enum ", "union ")


def strip_type_name(spelling: str) -> str:
    s = (spelling or "").strip()
    if "unnamed at" in s:
        return ""

    changed = True
    while changed:
        changed = False
        for q in _QUAL_PREFIX:
            if s.startswith(q):
                s = s[len(q) :].strip()
                changed = True

    for p in _TAG_PREFIX:
        if s.startswith(p):
            s = s[len(p) :].strip()

    changed = True
    while changed:
        changed = False
        for q in _QUAL_PREFIX:
            if s.startswith(q):
                s = s[len(q) :].strip()
                changed = True

    if "unnamed at" in s:
        return ""
    return s


_INT_SUFFIX_CHARS = set("uUlLzZ")


def normalize_int_literal(s: str) -> str:
    if not s:
        return s
    i = len(s)
    while i > 0 and s[i - 1] in _INT_SUFFIX_CHARS:
        i -= 1
    return s[:i]


def normalize_float_literal(s: str) -> str:
    if not s:
        return s
    if s[-1] in ("f", "F", "l", "L"):
        core = s[:-1]
        if any(ch in core for ch in (".", "e", "E", "p", "P")):
            return core
    return s


class SubsetError(Exception):
    def __init__(self, cur: Cursor, msg: str) -> None:
        super().__init__(f"{loc_str(cur)}: {msg}")
        self.cur = cur
        self.msg = msg


BIN_OP_SET = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "<<",
    ">>",
    "&",
    "|",
    "^",
    "&&",
    "||",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
}

ASSIGN_OPS = {
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
}


def extract_infix_op(cur: Cursor, left: Cursor, right: Cursor) -> str:
    le = end_off(left)
    rs = start_off(right)
    if le < 0 or rs < 0:
        for t in tok_spellings(cur):
            if t in BIN_OP_SET:
                return t
        return "?"
    for t in cur_tokens(cur):
        s = tok_start(t)
        e = tok_end(t)
        if s >= le and e <= rs and t.spelling in BIN_OP_SET:
            return t.spelling
    for t in tok_spellings(cur):
        if t in BIN_OP_SET:
            return t
    return "?"


def extract_unary_op(cur: Cursor, operand: Cursor) -> Tuple[str, str]:
    os_ = start_off(operand)
    oe_ = end_off(operand)
    if os_ < 0 or oe_ < 0:
        toks = tok_spellings(cur)
        if len(toks) >= 2:
            if toks[0] in ("++", "--", "!", "~", "&", "*", "+", "-"):
                return (toks[0], "")
            if toks[-1] in ("++", "--"):
                return ("", toks[-1])
        return ("", "")
    prefix: List[str] = []
    suffix: List[str] = []
    for t in cur_tokens(cur):
        s = tok_start(t)
        e = tok_end(t)
        if e <= os_:
            if t.spelling not in ("(", ")"):
                prefix.append(t.spelling)
        elif s >= oe_:
            if t.spelling not in ("(", ")"):
                suffix.append(t.spelling)
    return ("".join(prefix), "".join(suffix))


def validate_subset(tu: cindex.TranslationUnit, strict: bool) -> None:
    if not strict:
        return

    def forbid(cur: Cursor, msg: str) -> None:
        raise SubsetError(cur, msg)

    def walk(cur: Cursor) -> None:
        k = cur.kind

        if (K_UNION_DECL is not None) and (k == K_UNION_DECL) and is_definition(cur):
            forbid(cur, "union is forbidden in this C-subset")

        if k in (CursorKind.GOTO_STMT, CursorKind.LABEL_STMT):
            forbid(cur, "goto/labels are forbidden in this C-subset")

        if k == CursorKind.FUNCTION_DECL:
            try:
                if cur.type.is_function_variadic():
                    forbid(cur, "variadic functions (va_args) are forbidden in this C-subset")
            except Exception:
                pass

        if k in (CursorKind.VAR_DECL, CursorKind.PARM_DECL, CursorKind.FIELD_DECL, CursorKind.TYPEDEF_DECL):
            try:
                if is_function_pointer_type(cur.type):
                    forbid(cur, "function pointers are forbidden in this C-subset")
            except Exception:
                pass

        if ((K_SIZEOF_KIND is not None) and (k == K_SIZEOF_KIND)) or any(
            t == "sizeof" for t in tok_spellings(cur)
        ):
            forbid(cur, "sizeof(...) is forbidden in this C-subset")

        if k in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
            ch = list(cur.get_children())
            if len(ch) == 2:
                op = extract_infix_op(cur, ch[0], ch[1])
                if op in ("+", "-", "+=", "-="):
                    try:
                        if is_pointer_type(ch[0].type) or is_pointer_type(ch[1].type):
                            forbid(cur, f"pointer arithmetic '{op}' is forbidden in this C-subset")
                    except Exception:
                        pass

        if k == CursorKind.UNARY_OPERATOR:
            ch = list(cur.get_children())
            if len(ch) == 1:
                pre, suf = extract_unary_op(cur, ch[0])
                if (pre in ("++", "--") or suf in ("++", "--")) and is_pointer_type(ch[0].type):
                    forbid(cur, "pointer ++/-- is forbidden in this C-subset")

        for sub in cur.get_children():
            walk(sub)

    walk(tu.cursor)


def run_preprocess(clang: str, in_path: str, std: str, extra_args: List[str]) -> str:
    cmd = [clang, "-E", f"-std={std}", "-x", "c", in_path] + extra_args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"clang preprocess failed:\n{p.stderr}")
    return p.stdout


def build_translation_unit(
    idx: cindex.Index,
    input_c: str,
    std: str,
    clang: str,
    clang_args: List[str],
    preprocess: bool,
) -> cindex.TranslationUnit:
    if not preprocess:
        return idx.parse(input_c, args=[f"-std={std}", "-x", "c"] + clang_args)

    pre = run_preprocess(clang, input_c, std, clang_args)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False, encoding="utf-8")
    tmp.write(pre)
    tmp.flush()
    tmp.close()

    try:
        return idx.parse(tmp.name, args=[f"-std={std}", "-x", "c"])
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def find_decl_equals_offset(cur: Cursor) -> Optional[int]:
    for t in cur_tokens(cur):
        if t.spelling == "=":
            return tok_start(t)
    return None


def find_initializer_child(cur: Cursor) -> Optional[Cursor]:
    eq = find_decl_equals_offset(cur)
    if eq is None:
        return None

    candidates: List[Tuple[int, Cursor]] = []
    for ch in cur.get_children():
        so = start_off(ch)
        if so >= 0 and so > eq:
            candidates.append((so, ch))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    chs = list(cur.get_children())
    return chs[-1] if chs else None


def compute_ptr_param_modes(func: Cursor) -> Dict[str, str]:
    modes: Dict[str, str] = {}

    for a in func.get_arguments():
        if canonical_kind(a.type) == TypeKind.POINTER:
            pt = a.type.get_pointee()
            if is_c_char_scalar(pt):
                modes[a.spelling] = "string"
            else:
                modes[a.spelling] = "ref"

    def walk(c: Cursor) -> None:
        if c.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            ch = list(c.get_children())
            if len(ch) == 2:
                base = unwrap_expr(ch[0])
                if base.kind == CursorKind.DECL_REF_EXPR:
                    name = base.spelling
                    if name in modes and modes[name] != "string":
                        modes[name] = "array"
        for sub in c.get_children():
            walk(sub)

    for ch in func.get_children():
        if ch.kind == CursorKind.COMPOUND_STMT:
            walk(ch)

    return modes


@dataclass
class TypeInfo:
    sp_type: str
    name_suffix: str
    mode: str
    kind: str


class TypeMapper:
    def map(self, t: cindex.Type, *, for_param: bool, ptr_param_mode: Optional[str] = None) -> TypeInfo:
        if (t.spelling or "").strip() == "bool":
            return TypeInfo("bool", "", "value", "bool")

        k = canonical_kind(t)

        if t.kind == TypeKind.TYPEDEF:
            name = strip_type_name(t.spelling)
            ck = canonical_kind(t)

            if ck == TypeKind.BOOL:
                return TypeInfo(name or "bool", "", "value", "bool")
            if ck in _INT_KINDS or ck == TypeKind.ENUM:
                return TypeInfo(name or "int", "", "value", "int")
            if ck in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE):
                return TypeInfo(name or "float", "", "value", "float")
            if ck == TypeKind.RECORD:
                return TypeInfo(name or "any", "", "value", "struct" if name else "other")

            return TypeInfo(name or "any", "", "value" if name else "unsupported", "other")

        if k == TypeKind.VOID:
            return TypeInfo("void", "", "value", "void")
        if k == TypeKind.BOOL:
            return TypeInfo("bool", "", "value", "bool")
        if k in _INT_KINDS:
            return TypeInfo("int", "", "value", "int")
        if k in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE):
            return TypeInfo("float", "", "value", "float")

        if k == TypeKind.ENUM:
            name = strip_type_name(t.spelling)
            return TypeInfo(name if name else "int", "", "value", "enum" if name else "int")

        if k == TypeKind.RECORD:
            name = strip_type_name(t.spelling)
            return TypeInfo(name if name else "any", "", "value" if name else "unsupported", "struct" if name else "other")

        if k == TypeKind.CONSTANTARRAY:
            et = self.map(t.element_type, for_param=False)
            return TypeInfo(et.sp_type, f"[{t.element_count}]", "array", et.kind)

        if k == TypeKind.INCOMPLETEARRAY:
            et = self.map(t.element_type, for_param=False)
            return TypeInfo(et.sp_type, "[]", "array", et.kind)

        if k == TypeKind.POINTER:
            pt = t.get_pointee()

            if is_c_char_scalar(pt):
                if for_param:
                    return TypeInfo("char[]", "", "array", "string")
                return TypeInfo("int", "", "unsupported", "other")

            base_ti = self.map(pt, for_param=False)

            if not for_param:
                if base_ti.kind == "struct":
                    return TypeInfo(base_ti.sp_type, "", "ptr_struct", "struct")
                return TypeInfo("int", "", "unsupported", "other")

            mode = ptr_param_mode or "ref"

            if base_ti.kind == "struct":
                if mode == "array":
                    return TypeInfo(f"{base_ti.sp_type}[]", "", "array", "struct")
                return TypeInfo(base_ti.sp_type, "", "ref", "struct")

            if mode == "array":
                return TypeInfo(f"{base_ti.sp_type}[]", "", "array", base_ti.kind)
            if mode == "ref":
                return TypeInfo(f"{base_ti.sp_type}&", "", "ref", base_ti.kind)

            return TypeInfo("any", "", "unsupported", "other")

        sp = strip_type_name(t.spelling) if t.spelling else "any"
        return TypeInfo(sp, "", "value" if sp != "any" else "unsupported", "other")


ASSOCIATIVE_OPS = {"+", "*", "&", "^", "|", "&&", "||"}
BOOLISH_BIN_OPS = {"||", "&&", "==", "!=", "<", "<=", ">", ">="}


def _binop_prec(op: str) -> int:
    if op in ("*", "/", "%"):
        return 70
    if op in ("+", "-"):
        return 60
    if op in ("<<", ">>"):
        return 55
    if op in ("<", "<=", ">", ">="):
        return 50
    if op in ("==", "!="):
        return 45
    if op == "&":
        return 40
    if op == "^":
        return 39
    if op == "|":
        return 38
    if op == "&&":
        return 30
    if op == "||":
        return 29
    if op in ASSIGN_OPS:
        return 10
    return 0


def _node_prec(cur: Cursor) -> int:
    cur = unwrap_expr(cur)
    k = cur.kind
    if k in (
        CursorKind.INTEGER_LITERAL,
        CursorKind.FLOATING_LITERAL,
        CursorKind.STRING_LITERAL,
        CursorKind.CHARACTER_LITERAL,
        CursorKind.DECL_REF_EXPR,
    ):
        return 100
    if k in (CursorKind.CALL_EXPR, CursorKind.ARRAY_SUBSCRIPT_EXPR, CursorKind.MEMBER_REF_EXPR):
        return 90
    if k in (CursorKind.UNARY_OPERATOR, CursorKind.CSTYLE_CAST_EXPR):
        return 80
    if k in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
        ch = list(cur.get_children())
        if len(ch) == 2:
            op = extract_infix_op(cur, ch[0], ch[1])
            return _binop_prec(op)
    if k == CursorKind.CONDITIONAL_OPERATOR:
        return 20
    return 0


class ExprEmitter:
    def __init__(
        self,
        tm: TypeMapper,
        ptr_modes: Dict[str, str],
        alias: Dict[str, str],
        string_vars: Set[str],
        cfg: Config,
    ) -> None:
        self.tm = tm
        self.ptr_modes = ptr_modes
        self.alias = alias
        self.string_vars = string_vars
        self.cfg = cfg

    def unwrap(self, cur: Cursor) -> Cursor:
        return unwrap_expr(cur)

    def _is_boolish_expr(self, cur: Cursor) -> bool:
        cur = self.unwrap(cur)
        if is_c_bool_type(cur.type):
            return True
        if cur.kind == CursorKind.UNARY_OPERATOR:
            ch = list(cur.get_children())
            if len(ch) == 1:
                pre, _ = extract_unary_op(cur, ch[0])
                if pre == "!":
                    return True
        if cur.kind in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
            ch = list(cur.get_children())
            if len(ch) == 2:
                op = extract_infix_op(cur, ch[0], ch[1])
                if op in BOOLISH_BIN_OPS:
                    return True
        return False

    def emit(self, cur: Cursor) -> str:
        return self._emit(cur, parent_prec=0, parent_op=None, is_right_child=False)

    def _is_shift_expr(self, cur: Cursor) -> bool:
        cur = self.unwrap(cur)
        if cur.kind in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
            ch = list(cur.get_children())
            if len(ch) == 2:
                op = extract_infix_op(cur, ch[0], ch[1])
                return op in ("<<", ">>")
        return False

    def _emit(self, cur: Cursor, parent_prec: int, parent_op: Optional[str], is_right_child: bool) -> str:
        cur = self.unwrap(cur)
        k = cur.kind

        if k == CursorKind.INTEGER_LITERAL:
            toks = tok_spellings(cur)
            return normalize_int_literal(toks[0]) if toks else "0"

        if k == CursorKind.FLOATING_LITERAL:
            toks = tok_spellings(cur)
            return normalize_float_literal(toks[0]) if toks else "0.0"

        if k in (CursorKind.STRING_LITERAL, CursorKind.CHARACTER_LITERAL):
            toks = tok_spellings(cur)
            return toks[0] if toks else "0"

        if k == CursorKind.DECL_REF_EXPR:
            if cur.spelling in self.alias:
                return self.alias[cur.spelling]
            return cur.spelling

        if k == CursorKind.CALL_EXPR:
            ch = list(cur.get_children())
            if not ch:
                return "0"
            callee = self._emit(ch[0], parent_prec=90, parent_op=None, is_right_child=False)
            args = [self.emit(x) for x in ch[1:]]
            return f"{callee}({', '.join(args)})"

        if k == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            ch = list(cur.get_children())
            if len(ch) == 2:
                base = self._emit(ch[0], parent_prec=90, parent_op=None, is_right_child=False)
                idx = self.emit(ch[1])
                return f"{base}[{idx}]"
            return tok_join(cur)

        if k == CursorKind.MEMBER_REF_EXPR:
            ch = list(cur.get_children())
            base = self._emit(ch[0], parent_prec=90, parent_op=None, is_right_child=False) if ch else "0"
            return f"{base}.{cur.spelling}"

        if k == CursorKind.CSTYLE_CAST_EXPR:
            ch = list(cur.get_children())
            if not ch:
                return "0"
            expr = self.emit(ch[0])
            src_t = ch[0].type
            return self.emit_cast(cur.type, src_t, expr)

        if k == CursorKind.UNARY_OPERATOR:
            ch = list(cur.get_children())
            if len(ch) != 1:
                return tok_join(cur)

            opnd_cur = self.unwrap(ch[0])
            pre, suf = extract_unary_op(cur, opnd_cur)

            if pre == "&":
                return self.emit(opnd_cur)

            if pre == "*":
                if opnd_cur.kind == CursorKind.DECL_REF_EXPR:
                    name = opnd_cur.spelling
                    if name in self.alias:
                        return self.alias[name]
                    mode = self.ptr_modes.get(name)
                    if mode == "ref":
                        return name
                    if mode == "array":
                        return f"{name}[0]"
                opnd_txt = self.emit(opnd_cur)
                return f"/*deref*/{opnd_txt}"

            if pre:
                opnd = self._emit(opnd_cur, parent_prec=80, parent_op=None, is_right_child=False)
                if _node_prec(opnd_cur) < 80:
                    opnd = f"({opnd})"
                return f"{pre}{opnd}"

            if suf:
                opnd = self._emit(opnd_cur, parent_prec=80, parent_op=None, is_right_child=False)
                if _node_prec(opnd_cur) < 80:
                    opnd = f"({opnd})"
                return f"{opnd}{suf}"

            return self.emit(opnd_cur)

        if k in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
            ch = list(cur.get_children())
            if len(ch) != 2:
                return tok_join(cur)

            op = extract_infix_op(cur, ch[0], ch[1])
            my_prec = _binop_prec(op)

            if op in ("==", "!="):
                lhs = unwrap_expr(ch[0])
                rhs = unwrap_expr(ch[1])

                def is_zero_intlit(x: Cursor) -> bool:
                    if x.kind == CursorKind.INTEGER_LITERAL:
                        v = normalize_int_literal(tok_spellings(x)[0]) if tok_spellings(x) else "0"
                        return v == "0"
                    return False

                if lhs.kind == CursorKind.DECL_REF_EXPR and lhs.spelling in self.string_vars and is_zero_intlit(rhs):
                    s = f"{lhs.spelling}[0] {op} 0"
                    if my_prec < parent_prec:
                        return f"({s})"
                    return s
                if rhs.kind == CursorKind.DECL_REF_EXPR and rhs.spelling in self.string_vars and is_zero_intlit(lhs):
                    s = f"{rhs.spelling}[0] {op} 0"
                    if my_prec < parent_prec:
                        return f"({s})"
                    return s

            left = self._emit(ch[0], parent_prec=my_prec, parent_op=op, is_right_child=False)
            right = self._emit(ch[1], parent_prec=my_prec, parent_op=op, is_right_child=True)

            if op in ("&", "|", "^"):
                if self._is_shift_expr(ch[0]):
                    left = f"({left})"
                if self._is_shift_expr(ch[1]):
                    right = f"({right})"

            s = f"{left} {op} {right}"

            if my_prec < parent_prec:
                return f"({s})"
            if my_prec == parent_prec and is_right_child and parent_op is not None:
                if not (parent_op == op and op in ASSOCIATIVE_OPS):
                    return f"({s})"
            return s

        if k == CursorKind.CONDITIONAL_OPERATOR:
            ch = list(cur.get_children())
            if len(ch) != 3:
                return tok_join(cur)

            cond = self.emit(ch[0])

            a_cur = unwrap_expr(ch[1])
            b_cur = unwrap_expr(ch[2])
            a_s = self.emit(a_cur)
            b_s = self.emit(b_cur)

            a_is_char = is_c_char_scalar(a_cur.type)
            b_is_char = is_c_char_scalar(b_cur.type)
            if a_is_char and not b_is_char:
                a_s = sp_char_to_int(a_s, signed=char_is_signed_for_type(a_cur.type, self.cfg))
            elif b_is_char and not a_is_char:
                b_s = sp_char_to_int(b_s, signed=char_is_signed_for_type(b_cur.type, self.cfg))

            s = f"{cond} ? {a_s} : {b_s}"
            if 20 < parent_prec:
                return f"({s})"
            return s

        return tok_join(cur) or "0"

    def emit_cast(self, dst: cindex.Type, src: Optional[cindex.Type], expr: str) -> str:
        dst_k = canonical_kind(dst)
        src_k = canonical_kind(src) if src is not None else None
        dst_sp = self.tm.map(dst, for_param=False).sp_type

        if dst_k == TypeKind.BOOL or dst_sp == "bool":
            if src is not None and is_c_bool_type(src):
                return expr
            return f"({expr} != 0)"

        if dst_k in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE):
            return f"float({expr})"

        if dst_k in _INT_KINDS or dst_k == TypeKind.ENUM:
            if src_k in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE):
                return f"RoundToZero({expr})"
            if src is not None and is_c_char_scalar(src):
                signed = char_is_signed_for_type(src, self.cfg)
                return sp_char_to_int(expr, signed=signed)

            if dst_k == TypeKind.ENUM and dst_sp and dst_sp != "int":
                return f"view_as<{dst_sp}>({expr})"
            return expr

        return expr


class StmtEmitter:
    def __init__(self, tm: TypeMapper, ex: ExprEmitter, cfg: Config, ret_is_bool: bool) -> None:
        self.tm = tm
        self.ex = ex
        self.cfg = cfg
        self.ret_is_bool = ret_is_bool
        self.alias = ex.alias

    def _emit_bool_rhs(self, rhs_cur: Cursor) -> str:
        rhs_cur = unwrap_expr(rhs_cur)

        if rhs_cur.kind == CursorKind.CONDITIONAL_OPERATOR:
            ch = list(rhs_cur.get_children())
            if len(ch) == 3:
                cond = self.ex.emit(ch[0])
                a = self._emit_bool_rhs(ch[1])
                b = self._emit_bool_rhs(ch[2])
                return f"({cond} ? {a} : {b})"

        if rhs_cur.kind == CursorKind.INTEGER_LITERAL:
            s = self.ex.emit(rhs_cur).strip()
            if s == "0":
                return "false"
            if s == "1":
                return "true"

        if is_c_bool_type(rhs_cur.type) or self.ex._is_boolish_expr(rhs_cur):
            return self.ex.emit(rhs_cur)

        s = self.ex.emit(rhs_cur)
        return f"({s} != 0)"

    def _emit_rhs_for_lhs(self, lhs_cur: Cursor, rhs_cur: Cursor) -> str:
        lhs_cur = unwrap_expr(lhs_cur)
        rhs_cur = unwrap_expr(rhs_cur)

        if is_c_bool_type(lhs_cur.type):
            return self._emit_bool_rhs(rhs_cur)

        rhs_s = self.ex.emit(rhs_cur)

        if is_c_intish_type(lhs_cur.type) and is_c_char_scalar(rhs_cur.type):
            signed = char_is_signed_for_type(rhs_cur.type, self.cfg)
            return sp_char_to_int(rhs_s, signed=signed)

        return rhs_s

    def emit_stmt(self, cur: Cursor, out: Out) -> None:
        k = cur.kind

        if k == CursorKind.COMPOUND_STMT:
            out.w("{")
            out.ind += 1
            for ch in cur.get_children():
                self.emit_stmt(ch, out)
            out.ind -= 1
            out.w("}")
            return

        if k == CursorKind.DECL_STMT:
            for ch in cur.get_children():
                if ch.kind == CursorKind.VAR_DECL:
                    self.emit_vardecl(ch, out, is_global=False)
            return

        if k == CursorKind.RETURN_STMT:
            ch = list(cur.get_children())
            if not ch:
                out.w("return;")
                return
            if self.ret_is_bool:
                out.w(f"return {self._emit_bool_rhs(ch[0])};")
            else:
                out.w(f"return {self.ex.emit(ch[0])};")
            return

        if k == CursorKind.IF_STMT:
            ch = list(cur.get_children())
            cond = self.ex.emit(ch[0]) if len(ch) >= 1 else "0"
            out.w(f"if ({cond})")
            if len(ch) >= 2:
                self.emit_as_block(ch[1], out)
            else:
                out.w("{ }")
            if len(ch) >= 3:
                out.w("else")
                self.emit_as_block(ch[2], out)
            return

        if k == CursorKind.WHILE_STMT:
            ch = list(cur.get_children())
            cond = self.ex.emit(ch[0]) if len(ch) >= 1 else "0"
            out.w(f"while ({cond})")
            if len(ch) >= 2:
                self.emit_as_block(ch[1], out)
            else:
                out.w("{ }")
            return

        if k == CursorKind.FOR_STMT:
            ch = list(cur.get_children())
            init = self.emit_for_part(ch[0]) if len(ch) >= 1 else ""
            cond = self.ex.emit(ch[1]) if len(ch) >= 2 and ch[1].kind != CursorKind.NULL_STMT else ""
            inc = self.emit_for_part(ch[2]) if len(ch) >= 3 else ""
            body = ch[3] if len(ch) >= 4 else None

            out.w(f"for ({init}; {cond}; {inc})")
            if body is not None:
                self.emit_as_block(body, out)
            else:
                out.w("{ }")
            return

        if k == CursorKind.DO_STMT:
            ch = list(cur.get_children())
            body = ch[0] if len(ch) >= 1 else None
            cond = ch[1] if len(ch) >= 2 else None
            out.w("do")
            if body is not None:
                self.emit_as_block(body, out)
            else:
                out.w("{ }")
            out.w(f"while ({self.ex.emit(cond)});" if cond is not None else "while (0);")
            return

        if k == CursorKind.SWITCH_STMT:
            ch = list(cur.get_children())
            expr_cur = ch[0] if len(ch) >= 1 else None
            body_cur = ch[1] if len(ch) >= 2 else None
            self.emit_switch_as_if(expr_cur, body_cur, out)
            return

        if k == CursorKind.BREAK_STMT:
            out.w("break;")
            return

        if k == CursorKind.CONTINUE_STMT:
            out.w("continue;")
            return

        if k == CursorKind.NULL_STMT:
            out.w(";")
            return

        if k in (CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
            ch = list(cur.get_children())
            if len(ch) == 2:
                op = extract_infix_op(cur, ch[0], ch[1])
                if op in ASSIGN_OPS:
                    lhs = ch[0]
                    rhs = ch[1]
                    rhs_s = self._emit_rhs_for_lhs(lhs, rhs)
                    out.w(f"{self.ex.emit(lhs)} {op} {rhs_s};")
                    return
            out.w(self.ex.emit(cur) + ";")
            return

        if k == CursorKind.UNARY_OPERATOR:
            ch = list(cur.get_children())
            if len(ch) == 1:
                pre, suf = extract_unary_op(cur, ch[0])
                if pre in ("++", "--") or suf in ("++", "--"):
                    out.w(f"{pre}{self.ex.emit(ch[0])}{suf};")
                    return
            out.w(self.ex.emit(cur) + ";")
            return

        if k in (
            CursorKind.CALL_EXPR,
            CursorKind.CONDITIONAL_OPERATOR,
            CursorKind.CSTYLE_CAST_EXPR,
            CursorKind.ARRAY_SUBSCRIPT_EXPR,
            CursorKind.MEMBER_REF_EXPR,
        ):
            out.w(self.ex.emit(cur) + ";")
            return

        joined = tok_join(cur)
        out.w((joined + ";") if joined else "/* stmt unsupported */")

    def emit_as_block(self, cur: Cursor, out: Out) -> None:
        if cur.kind == CursorKind.COMPOUND_STMT:
            self.emit_stmt(cur, out)
            return
        out.w("{")
        out.ind += 1
        self.emit_stmt(cur, out)
        out.ind -= 1
        out.w("}")

    def emit_for_part(self, cur: Cursor) -> str:
        if cur.kind == CursorKind.NULL_STMT:
            return ""
        if cur.kind == CursorKind.DECL_STMT:
            parts: List[str] = []
            for ch in cur.get_children():
                if ch.kind == CursorKind.VAR_DECL:
                    ti = self.tm.map(ch.type, for_param=False)
                    name = ch.spelling
                    init_cur = find_initializer_child(ch)
                    if init_cur is not None:
                        init = self._emit_init_for_type(ch.type, ti, init_cur)
                        parts.append(f"{ti.sp_type} {name}{ti.name_suffix} = {init}")
                    else:
                        parts.append(f"{ti.sp_type} {name}{ti.name_suffix}")
            return ", ".join(parts)
        return self.ex.emit(cur)

    def _emit_init_for_type(self, decl_type: cindex.Type, ti: TypeInfo, init_cur: Cursor) -> str:
        if ti.sp_type == "bool" or is_c_bool_type(decl_type):
            return self._emit_bool_rhs(init_cur)

        if ti.sp_type == "int":
            u = unwrap_expr(init_cur)
            if is_c_char_scalar(u.type):
                signed = char_is_signed_for_type(u.type, self.cfg)
                return sp_char_to_int(self.ex.emit(u), signed=signed)

        return self.ex.emit(init_cur)

    def emit_vardecl(self, cur: Cursor, out: Out, is_global: bool) -> None:
        name = cur.spelling
        ti = self.tm.map(cur.type, for_param=False)

        prefix = ""
        try:
            if cur.storage_class == cindex.StorageClass.STATIC:
                prefix += "static "
        except Exception:
            pass

        init_cur = find_initializer_child(cur)

        if not is_global and canonical_kind(cur.type) == TypeKind.POINTER:
            base = self.tm.map(cur.type, for_param=False)
            if base.mode == "ptr_struct":
                if init_cur is not None and unwrap_expr(init_cur).kind == CursorKind.UNARY_OPERATOR:
                    u = unwrap_expr(init_cur)
                    ch = list(u.get_children())
                    if len(ch) == 1:
                        pre, _ = extract_unary_op(u, ch[0])
                        if pre == "&":
                            self.alias[name] = self.ex.emit(ch[0])
                            return

        if init_cur is not None:
            init_u = unwrap_expr(init_cur)

            if (not is_global) and ti.kind == "struct" and init_u.kind != CursorKind.INIT_LIST_EXPR:
                out.w(f"{prefix}{ti.sp_type} {name}{ti.name_suffix};")
                init = self._emit_init_for_type(cur.type, ti, init_u)
                out.w(f"{name} = {init};")
                return

            init = self._emit_init_for_type(cur.type, ti, init_u)
            out.w(f"{prefix}{ti.sp_type} {name}{ti.name_suffix} = {init};")
        else:
            out.w(f"{prefix}{ti.sp_type} {name}{ti.name_suffix};")

    def emit_switch_as_if(self, expr_cur: Optional[Cursor], body_cur: Optional[Cursor], out: Out) -> None:
        sw_expr = self.ex.emit(expr_cur) if expr_cur is not None else "0"

        cases: List[Tuple[List[str], List[Cursor]]] = []
        default_stmts: Optional[List[Cursor]] = None

        body_children = list(body_cur.get_children()) if body_cur is not None else []

        for node in body_children:
            if node.kind == CursorKind.CASE_STMT:
                vals, stmts = self._collect_case_group(node)
                cases.append((vals, stmts))
            elif node.kind == CursorKind.DEFAULT_STMT:
                default_stmts = self._collect_default_stmts(node)

        first = True
        for vals, stmts in cases:
            cond = " || ".join([f"{sw_expr} == {v}" for v in vals])
            if first:
                out.w(f"if ({cond})")
                first = False
            else:
                out.w(f"else if ({cond})")
            out.w("{")
            out.ind += 1
            for s in stmts:
                if s.kind == CursorKind.BREAK_STMT:
                    continue
                self.emit_stmt(s, out)
            out.ind -= 1
            out.w("}")

        if default_stmts is not None:
            out.w("else")
            out.w("{")
            out.ind += 1
            for s in default_stmts:
                if s.kind == CursorKind.BREAK_STMT:
                    continue
                self.emit_stmt(s, out)
            out.ind -= 1
            out.w("}")

        if not cases and default_stmts is None:
            out.w("{")
            out.w("}")

    def _collect_case_group(self, case_cur: Cursor) -> Tuple[List[str], List[Cursor]]:
        ch = list(case_cur.get_children())
        if not ch:
            return (["0"], [])

        vals: List[str] = [self.ex.emit(ch[0])]
        rest: List[Cursor] = ch[1:]

        while len(rest) == 1 and rest[0].kind == CursorKind.CASE_STMT:
            nxt = rest[0]
            nch = list(nxt.get_children())
            if not nch:
                break
            vals.append(self.ex.emit(nch[0]))
            rest = nch[1:]

        if len(rest) == 1 and rest[0].kind == CursorKind.COMPOUND_STMT:
            rest = list(rest[0].get_children())

        return (vals, rest)

    def _collect_default_stmts(self, def_cur: Cursor) -> List[Cursor]:
        stmts = list(def_cur.get_children())
        if len(stmts) == 1 and stmts[0].kind == CursorKind.COMPOUND_STMT:
            return list(stmts[0].get_children())
        return stmts


def emit_enum(cur: Cursor, ignore_symbols: Set[str]) -> List[str]:
    name = strip_type_name(cur.spelling)
    if not name:
        return []
    if name in ignore_symbols:
        return []

    consts = [ch for ch in cur.get_children() if ch.kind == CursorKind.ENUM_CONSTANT_DECL]
    consts = [ch for ch in consts if (ch.spelling and ch.spelling not in ignore_symbols)]
    if not consts:
        return []

    o = Out()
    o.w(f"enum {name}")
    o.w("{")
    o.ind += 1

    for i, ch in enumerate(consts):
        try:
            v = ch.enum_value
            line = f"{ch.spelling} = {v}"
        except Exception:
            line = f"{ch.spelling}"
        if i != (len(consts) - 1):
            o.w(line + ",\n") # NOTE: Minor bug but newline is needed here
        else:
            o.w(line)

    o.ind -= 1
    o.w("}")
    return o.lines


def emit_struct(cur: Cursor, tm: TypeMapper, ignore_symbols: Set[str]) -> List[str]:
    name = strip_type_name(cur.spelling)
    if not name:
        return []
    if name in ignore_symbols:
        return []

    o = Out()
    o.w(f"enum struct {name}")
    o.w("{")
    o.ind += 1
    for ch in cur.get_children():
        if ch.kind == CursorKind.FIELD_DECL:
            ti = tm.map(ch.type, for_param=False)
            o.w(f"{ti.sp_type} {ch.spelling}{ti.name_suffix};")
    o.ind -= 1
    o.w("}")
    return o.lines


def emit_function(cur: Cursor, tm: TypeMapper, cfg: Config) -> List[str]:
    ptr_modes = compute_ptr_param_modes(cur)

    alias: Dict[str, str] = {}
    string_vars: Set[str] = set()
    for a in cur.get_arguments():
        if canonical_kind(a.type) == TypeKind.POINTER and is_c_char_scalar(a.type.get_pointee()):
            string_vars.add(a.spelling)

    ex = ExprEmitter(tm, ptr_modes, alias, string_vars, cfg)

    rt = tm.map(cur.result_type, for_param=False).sp_type
    name = cur.spelling

    ret_is_bool = is_c_bool_type(cur.result_type) or (rt == "bool")
    st = StmtEmitter(tm, ex, cfg, ret_is_bool=ret_is_bool)

    params: List[str] = []
    for a in cur.get_arguments():
        pmode = None
        if canonical_kind(a.type) == TypeKind.POINTER:
            pmode = ptr_modes.get(a.spelling, "ref")

        ti = tm.map(a.type, for_param=True, ptr_param_mode=pmode)

        prefix = ""
        if canonical_kind(a.type) == TypeKind.POINTER:
            if pointee_is_const(a.type):
                prefix = "const "
        else:
            if is_const_qualified(a.type):
                prefix = "const "

        params.append(f"{prefix}{ti.sp_type} {a.spelling}{ti.name_suffix}".rstrip())

    is_static = False
    try:
        is_static = (cur.storage_class == cindex.StorageClass.STATIC)
    except Exception:
        pass

    fn_prefix = "static " if is_static else "public "

    o = Out()
    o.w(f"{fn_prefix}{rt} {name}({', '.join(params)})")
    o.w("{")
    o.ind += 1

    body = None
    for ch in cur.get_children():
        if ch.kind == CursorKind.COMPOUND_STMT:
            body = ch
            break

    if body is not None:
        for s in body.get_children():
            st.emit_stmt(s, o)

    o.ind -= 1
    o.w("}")
    return o.lines


def _extract_literals(line: str) -> Tuple[str, List[str]]:
    out = []
    lits: List[str] = []
    i = 0
    n = len(line)

    def take_quoted(q: str, start: int) -> Tuple[str, int]:
        j = start + 1
        while j < n:
            c = line[j]
            if c == "\\":
                j += 2
                continue
            if c == q:
                return (line[start : j + 1], j + 1)
            j += 1
        return (line[start:], n)

    while i < n:
        c = line[i]
        if c in ("'", '"'):
            lit, i2 = take_quoted(c, i)
            key = f"@@LIT{len(lits)}@@"
            lits.append(lit)
            out.append(key)
            i = i2
            continue
        out.append(c)
        i += 1

    return ("".join(out), lits)


def _restore_literals(line: str, lits: List[str]) -> str:
    for i, lit in enumerate(lits):
        line = line.replace(f"@@LIT{i}@@", lit)
    return line


def format_sourcepawn(text: str) -> str:
    lines = text.splitlines(True)
    out_lines: List[str] = []
    in_block = False

    for raw in lines:
        line = raw

        if in_block:
            out_lines.append(line)
            if "*/" in line:
                in_block = False
            continue

        if "/*" in line:
            out_lines.append(line)
            if "*/" not in line:
                in_block = True
            continue

        masked, lits = _extract_literals(line)
        idx = masked.find("//")
        if idx >= 0:
            code = masked[:idx]
            comment = masked[idx:]
        else:
            code = masked
            comment = ""

        code = re.sub(r"\s*\.\s*", ".", code)
        code = re.sub(r"!\s+", "!", code)
        code = re.sub(r"~\s+", "~", code)
        code = re.sub(r"\(\s+", "(", code)
        code = re.sub(r"\s+\)", ")", code)
        code = re.sub(r"\[\s+", "[", code)
        code = re.sub(r"\s+\]", "]", code)
        code = re.sub(r"\s+;", ";", code)
        code = re.sub(r"\s*,\s*", ", ", code)

        _kw = {"if", "for", "while", "switch", "catch"}

        def _fix_call_space(m: re.Match) -> str:
            name = m.group(1)
            if name in _kw:
                return m.group(0)
            return f"{name}("

        code = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*(?:<[^>]+>)?)\s+\(", _fix_call_space, code)

        code = re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:<[^>]+>)?)\s+&\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\1& \2",
            code,
        )

        code = re.sub(r"(=\s*)-\s+(\d)", r"\1-\2", code)
        code = re.sub(r"(\(\s*)-\s+(\d)", r"\1-\2", code)
        code = re.sub(r"(,\s*)-\s+(\d)", r"\1-\2", code)

        merged = code + comment
        merged = _restore_literals(merged, lits)
        out_lines.append(merged)

    return "".join(out_lines)


def main() -> None:
    try_set_libclang()

    ap = argparse.ArgumentParser()
    ap.add_argument("input_c", help="input .c file")
    ap.add_argument("-o", "--output", default="out.sp", help="output .sp")
    ap.add_argument("--clang", default="clang", help="clang executable")
    ap.add_argument("--std", default="c17", help="C standard")
    ap.add_argument("--clang-arg", action="append", default=[], help="extra clang args (repeatable)")
    ap.add_argument("--include", action="append", default=[], help="SP include lines (repeatable)")
    ap.add_argument("--strict", action="store_true", help="strict subset validation")
    ap.add_argument("--no-preprocess", action="store_true", help="parse original file (no clang -E stage)")
    ap.add_argument("--ignore-symbols", default="", help="path to compiler log or list of symbols to ignore")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--char-unsigned", action="store_true", help="plain C `char` promoted as unsigned 0..255 (default)")
    g.add_argument("--char-signed", action="store_true", help="plain C `char` promoted as signed -128..127")

    args = ap.parse_args()

    cfg = Config(char_signed=False)
    if args.char_signed:
        cfg.char_signed = True

    ignore_symbols: Set[str] = set()
    if args.ignore_symbols:
        ignore_symbols |= load_ignore_symbols(args.ignore_symbols)

    idx = cindex.Index.create()
    tu = build_translation_unit(
        idx=idx,
        input_c=args.input_c,
        std=args.std,
        clang=args.clang,
        clang_args=args.clang_arg,
        preprocess=(not args.no_preprocess),
    )

    validate_subset(tu, strict=args.strict)

    tm = TypeMapper()
    out = Out()

    out.w("#pragma semicolon 1")
    out.w("#pragma newdecls required")
    for inc in args.include:
        out.w(inc)
    out.w("")

    best: Dict[str, Cursor] = {}
    for c in tu.cursor.get_children():
        try:
            usr = c.get_usr()
        except Exception:
            usr = ""
        key = usr or f"{c.kind}:{c.spelling}:{start_off(c)}"

        if key not in best:
            best[key] = c
            continue

        old = best[key]
        if is_definition(c) and not is_definition(old):
            best[key] = c
            continue
        if is_definition(c) and is_definition(old):
            if start_off(c) > start_off(old):
                best[key] = c
            continue

    enums: List[Cursor] = []
    structs: List[Cursor] = []
    globals_: List[Cursor] = []
    funcs: List[Cursor] = []

    for c in best.values():
        if c.kind == CursorKind.ENUM_DECL and is_definition(c):
            enums.append(c)
        elif c.kind == CursorKind.STRUCT_DECL and is_definition(c):
            structs.append(c)
        elif c.kind == CursorKind.VAR_DECL:
            if c.spelling and c.spelling in ignore_symbols:
                continue
            globals_.append(c)
        elif c.kind == CursorKind.FUNCTION_DECL and is_definition(c):
            if c.spelling and c.spelling in ignore_symbols:
                continue
            funcs.append(c)

    enums.sort(key=start_off)
    structs.sort(key=start_off)
    globals_.sort(key=start_off)
    funcs.sort(key=start_off)

    for e in enums:
        lines = emit_enum(e, ignore_symbols)
        for ln in lines:
            out.w(ln)
        if lines:
            out.w("")

    for s in structs:
        lines = emit_struct(s, tm, ignore_symbols)
        for ln in lines:
            out.w(ln)
        if lines:
            out.w("")

    if globals_:
        ex = ExprEmitter(tm, {}, {}, set(), cfg)
        st = StmtEmitter(tm, ex, cfg, ret_is_bool=False)
        for gcur in globals_:
            if gcur.spelling and gcur.spelling in ignore_symbols:
                continue
            st.emit_vardecl(gcur, out, is_global=True)
        out.w("")

    for fn in funcs:
        if fn.spelling and fn.spelling in ignore_symbols:
            continue
        for ln in emit_function(fn, tm, cfg):
            out.w(ln)
        out.w("")

    text = format_sourcepawn(out.get())

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
