#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RE_LINE_COMMENT = re.compile(r"//.*?$", re.M)

RE_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$")
RE_CONST = re.compile(r"^\s*const\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*;\s*$")
RE_ENUM_STRUCT = re.compile(r"^\s*enum\s+struct\s+([A-Za-z_]\w*)\b")
RE_ENUM = re.compile(r"^\s*enum(?:\s+([A-Za-z_]\w*))?\b")
RE_METHODMAP = re.compile(r"^\s*methodmap\s+([A-Za-z_]\w*)\b")
RE_PROP_HEAD = re.compile(r"^\s*property\s+(.+?)\s+([A-Za-z_]\w*)\b")
RE_PROTO_START = re.compile(r"^\s*(native|forward)\b")

_SP_TAG_MAP: Dict[str, str] = {
    "Float": "float",
    "bool": "bool",
    "Bool": "bool",
    "Action": "Action",
}

_BASE_MAP: Dict[str, str] = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "void": "void",
    "char": "char",
    "any": "int",
    "Handle": "Handle",
    "Function": "int",
    "Address": "uintptr_t",
}


@dataclass
class EnumDecl:
    name: Optional[str]
    items: List[str]


@dataclass
class StructDecl:
    name: str
    fields: List[str]


@dataclass
class MethodmapDecl:
    name: str
    props: Dict[str, str] = field(default_factory=dict)


@dataclass
class ParamDecl:
    cdecl: str
    name: str
    default: Optional[str] = None


@dataclass
class FuncDecl:
    ret: str
    name: str
    params: List[ParamDecl]


@dataclass
class ConstDecl:
    ctype: str
    name: str
    expr: str


def strip_comments(text: str) -> str:
    text = _RE_BLOCK_COMMENT.sub("", text)
    text = _RE_LINE_COMMENT.sub("", text)
    return text


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth_par = depth_br = depth_sq = 0
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            cur.append(ch)
            continue

        if ch == "(":
            depth_par += 1
        elif ch == ")":
            depth_par = max(0, depth_par - 1)
        elif ch == "{":
            depth_br += 1
        elif ch == "}":
            depth_br = max(0, depth_br - 1)
        elif ch == "[":
            depth_sq += 1
        elif ch == "]":
            depth_sq = max(0, depth_sq - 1)

        if ch == "," and depth_par == 0 and depth_br == 0 and depth_sq == 0:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)

    last = "".join(cur).strip()
    if last:
        parts.append(last)
    return parts


def split_top_level_default(s: str) -> Tuple[str, Optional[str]]:
    s = s.strip()
    if not s:
        return s, None

    depth_par = depth_br = depth_sq = 0
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "(":
            depth_par += 1
            continue
        if ch == ")":
            depth_par = max(0, depth_par - 1)
            continue
        if ch == "{":
            depth_br += 1
            continue
        if ch == "}":
            depth_br = max(0, depth_br - 1)
            continue
        if ch == "[":
            depth_sq += 1
            continue
        if ch == "]":
            depth_sq = max(0, depth_sq - 1)
            continue

        if ch == "=" and depth_par == 0 and depth_br == 0 and depth_sq == 0:
            prev = s[i - 1] if i > 0 else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt == "=":
                continue
            if prev in ("!", "<", ">", "="):
                continue

            lhs = s[:i].rstrip()
            rhs = s[i + 1:].strip()
            return lhs, rhs if rhs != "" else None

    return s, None


def collect_brace_block(lines: List[str], start_i: int) -> Tuple[str, int]:
    buf: List[str] = []
    depth = 0
    started = False

    i = start_i
    while i < len(lines):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
        buf.append(line)
        if started and depth == 0:
            return "\n".join(buf), i + 1
        i += 1

    return "\n".join(buf), i


def normalize_sp_type(sp_t: str) -> str:
    sp_t = norm_ws(sp_t)
    sp_t = sp_t.replace("const ", "").strip()

    if sp_t.endswith(":"):
        tag = sp_t[:-1].strip()
        return _SP_TAG_MAP.get(tag, "int")

    sp_t = re.sub(
        r"\b([A-Za-z_]\w*):\b",
        lambda m: _SP_TAG_MAP.get(m.group(1), "int") + " ",
        sp_t,
    )
    sp_t = norm_ws(sp_t)
    return _BASE_MAP.get(sp_t, sp_t)


def _maybe_float_suffix(expr: str) -> str:
    e = expr.strip()
    if e.lower().endswith("f"):
        return e
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", e):
        if "." in e or "e" in e.lower():
            return e + "f"
    return e


def sp_param_to_c(param: str) -> Optional[ParamDecl]:
    param = param.strip()
    if not param or param == "void" or param == "...":
        return None

    param_no_def, def_expr = split_top_level_default(param)

    is_const = bool(re.search(r"\bconst\b", param_no_def))
    is_ref = "&" in param_no_def

    param_no_def = param_no_def.replace("&", " ").strip()
    param_no_def = norm_ws(param_no_def)

    m = re.match(
        r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_]\w*)(?P<arr>\s*\[[^\]]*\])?\s*$",
        param_no_def,
    )
    if not m:
        return None

    sp_type = m.group("type").strip()
    name = m.group("name").strip()
    arr = (m.group("arr") or "").strip()

    if sp_type.endswith("[]"):
        sp_type = sp_type[:-2].strip()
        arr = "[]"

    c_type = normalize_sp_type(sp_type)

    if c_type == "char" and arr:
        cdecl = f"{'const char*' if is_const else 'char*'} {name}"
        return ParamDecl(cdecl=cdecl, name=name, default=def_expr)

    if is_ref or arr:
        cdecl = f"{'const ' if is_const else ''}{c_type}* {name}".strip()
        return ParamDecl(cdecl=cdecl, name=name, default=def_expr)

    cdecl = f"{'const ' if is_const else ''}{c_type} {name}".strip()

    if def_expr is not None and c_type == "float":
        def_expr = _maybe_float_suffix(def_expr)

    return ParamDecl(cdecl=cdecl, name=name, default=def_expr)


def sp_field_to_c(line: str) -> Optional[str]:
    line = line.strip().rstrip(";").strip()
    if not line:
        return None

    m = re.match(
        r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_]\w*)(?P<arr>\s*\[[^\]]*\])?\s*$",
        line,
    )
    if not m:
        return None

    sp_type = m.group("type").strip()
    name = m.group("name").strip()
    arr = (m.group("arr") or "").strip()

    if sp_type.endswith("[]"):
        sp_type = sp_type[:-2].strip()
        arr = "[]"

    c_type = normalize_sp_type(sp_type)

    if c_type == "char" and arr:
        if arr == "[]":
            return f"char* {name};"
        return f"char {name}{arr};"

    if arr:
        if arr == "[]":
            return f"{c_type}* {name};"
        return f"{c_type} {name}{arr};"

    return f"{c_type} {name};"


def parse_enum_block(block_text: str, name: Optional[str]) -> EnumDecl:
    b0 = block_text.find("{")
    b1 = block_text.rfind("}")
    inner = block_text[b0 + 1 : b1].strip() if (b0 >= 0 and b1 > b0) else ""
    raw_items = split_top_level_commas(inner)

    items: List[str] = []
    for it in raw_items:
        it = norm_ws(it).rstrip(",").strip()
        if it:
            items.append(it)

    return EnumDecl(name=name, items=items)


def parse_enum_struct_block(block_text: str, name: str) -> StructDecl:
    b0 = block_text.find("{")
    b1 = block_text.rfind("}")
    inner = block_text[b0 + 1 : b1].strip() if (b0 >= 0 and b1 > b0) else ""

    fields: List[str] = []
    for ln in inner.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "{" in ln:
            ln = ln.split("{", 1)[0].strip()
        if not ln.endswith(";"):
            continue
        fld = sp_field_to_c(ln)
        if fld:
            fields.append(fld)

    return StructDecl(name=name, fields=fields)


def parse_methodmap_block(block_text: str, name: str) -> MethodmapDecl:
    mm = MethodmapDecl(name=name)
    for ln in block_text.splitlines():
        ln = ln.strip()
        m = RE_PROP_HEAD.match(ln)
        if not m:
            continue
        sp_type = norm_ws(m.group(1))
        prop = m.group(2)
        c_type = normalize_sp_type(sp_type)
        if c_type.endswith("[]"):
            c_type = c_type[:-2].strip() + "*"
        if c_type == "char":
            c_type = "char*"
        mm.props[prop] = c_type
    return mm


def parse_proto_stmt(stmt: str) -> Optional[FuncDecl]:
    stmt = stmt.strip()
    if not stmt.endswith(";"):
        return None

    stmt = re.sub(r"^\s*(native|forward)\s+", "", stmt).strip()

    m = re.match(
        r"^(?P<ret>.+?)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*)\)\s*;\s*$",
        stmt,
        re.S,
    )
    if not m:
        return None

    sp_ret = norm_ws(m.group("ret"))
    name = m.group("name")
    args = m.group("args").strip()

    c_ret = normalize_sp_type(sp_ret)
    params: List[ParamDecl] = []

    if args and args != "void":
        for p in split_top_level_commas(args):
            cp = sp_param_to_c(p)
            if cp:
                params.append(cp)

    return FuncDecl(ret=c_ret, name=name, params=params)


def parse_text(
    text: str,
) -> Tuple[
    Dict[str, str],
    List[ConstDecl],
    Dict[str, EnumDecl],
    Dict[str, StructDecl],
    Dict[str, MethodmapDecl],
    Dict[str, FuncDecl],
]:
    defines: Dict[str, str] = {}
    consts: List[ConstDecl] = []
    enums: Dict[str, EnumDecl] = {}
    structs: Dict[str, StructDecl] = {}
    methodmaps: Dict[str, MethodmapDecl] = {}
    funcs: Dict[str, FuncDecl] = {}

    text = strip_comments(text)
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()

        md = RE_DEFINE.match(ln)
        if md:
            name = md.group(1)
            val = md.group(2).strip()
            if "\\" not in ln:
                defines[name] = val
            i += 1
            continue

        mc = RE_CONST.match(ln)
        if mc:
            sp_t, name, expr = mc.group(1), mc.group(2), mc.group(3).strip()
            consts.append(ConstDecl(ctype=normalize_sp_type(sp_t), name=name, expr=expr))
            i += 1
            continue

        ms = RE_ENUM_STRUCT.match(ln)
        if ms:
            name = ms.group(1)
            k = i
            while k < len(lines) and "{" not in lines[k]:
                k += 1
            if k < len(lines):
                block, j = collect_brace_block(lines, k)
                full = ("\n".join(lines[i:k]) + "\n" + block) if k != i else block
                structs[name] = parse_enum_struct_block(full, name)
                i = j
                continue

        me = RE_ENUM.match(ln)
        if me and not ln.strip().startswith("enum struct"):
            name = me.group(1)
            k = i
            while k < len(lines) and "{" not in lines[k]:
                if ";" in lines[k]:
                    break
                k += 1
            if k < len(lines) and "{" in lines[k]:
                block, j = collect_brace_block(lines, k)
                full = ("\n".join(lines[i:k]) + "\n" + block) if k != i else block
                key = name or f"__anon_enum_{len(enums) + 1}"
                enums[key] = parse_enum_block(full, name)
                i = j
                continue

        mm = RE_METHODMAP.match(ln)
        if mm:
            name = mm.group(1)
            k = i
            while k < len(lines) and "{" not in lines[k]:
                if ";" in lines[k]:
                    break
                k += 1
            if k < len(lines) and "{" in lines[k]:
                block, j = collect_brace_block(lines, k)
                full = ("\n".join(lines[i:k]) + "\n" + block) if k != i else block
                d = parse_methodmap_block(full, name)
                prev = methodmaps.get(name)
                if prev:
                    prev.props.update(d.props)
                else:
                    methodmaps[name] = d
                i = j
                continue

        if RE_PROTO_START.match(ln):
            stmt_lines = [ln]
            j = i + 1
            while j < len(lines) and ";" not in stmt_lines[-1]:
                stmt_lines.append(lines[j].rstrip())
                j += 1
            stmt = "\n".join(stmt_lines)

            fd = parse_proto_stmt(stmt)
            if fd:
                funcs[fd.name] = fd

            i = j
            continue

        i += 1

    return defines, consts, enums, structs, methodmaps, funcs


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def collect_input_paths(inp: str) -> List[str]:
    inp = os.path.abspath(inp)
    if os.path.isdir(inp):
        acc: List[str] = []
        for root, _, names in os.walk(inp):
            for n in names:
                if n.lower().endswith(".inc"):
                    acc.append(os.path.join(root, n))
        acc.sort()
        return acc
    return [inp]


def render_header(
    defines: Dict[str, str],
    consts: List[ConstDecl],
    enums: Dict[str, EnumDecl],
    structs: Dict[str, StructDecl],
    methodmaps: Dict[str, MethodmapDecl],
    funcs: Dict[str, FuncDecl],
    source_name: str,
) -> str:
    out: List[str] = []
    out.append("#pragma once")
    out.append("")
    out.append("/* minimal fixed-width helpers */")
    out.append("#if defined(__UINTPTR_TYPE__)")
    out.append("typedef __UINTPTR_TYPE__ uintptr_t;")
    out.append("#elif defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)")
    out.append("typedef unsigned long long uintptr_t;")
    out.append("#else")
    out.append("typedef unsigned long uintptr_t;")
    out.append("#endif")
    out.append("")
    out.append("typedef unsigned char bool;")
    out.append("#ifndef true")
    out.append("#define true 1")
    out.append("#endif")
    out.append("#ifndef false")
    out.append("#define false 0")
    out.append("#endif")
    out.append("")
    out.append("typedef void* Handle;")
    out.append("typedef uintptr_t Address;")
    out.append("typedef int any;")
    out.append("")

    if defines:
        out.append("/* ---------------------------- */")
        out.append("/* #define */")
        out.append("/* ---------------------------- */")
        for k in sorted(defines.keys()):
            out.append(f"#define {k} {defines[k]}")
        out.append("")

    if consts:
        out.append("/* ---------------------------- */")
        out.append("/* const */")
        out.append("/* ---------------------------- */")
        for c in consts:
            out.append(f"static const {c.ctype} {c.name} = {c.expr};")
        out.append("")

    if enums:
        out.append("/* ---------------------------- */")
        out.append("/* enums */")
        out.append("/* ---------------------------- */")
        for key in sorted(enums.keys()):
            e = enums[key]
            if e.name:
                out.append(f"typedef enum {e.name}")
            else:
                out.append("enum")
            out.append("{")
            for idx, it in enumerate(e.items):
                comma = "," if idx != (len(e.items) - 1) else ""
                out.append(f"    {it}{comma}")
            out.append("}" + (f" {e.name};" if e.name else ";"))
            out.append("")
        out.append("")

    if structs:
        out.append("/* ---------------------------- */")
        out.append("/* enum struct -> struct */")
        out.append("/* ---------------------------- */")
        for name in sorted(structs.keys()):
            s = structs[name]
            out.append(f"typedef struct {s.name}")
            out.append("{")
            if s.fields:
                for f in s.fields:
                    out.append(f"    {f}")
            else:
                out.append("    int __dummy;")
            out.append(f"}} {s.name};")
            out.append("")
        out.append("")

    if methodmaps:
        out.append("/* ---------------------------- */")
        out.append("/* methodmap -> struct (properties only) */")
        out.append("/* ---------------------------- */")
        for name in sorted(methodmaps.keys()):
            mm = methodmaps[name]
            out.append(f"typedef struct {mm.name}")
            out.append("{")
            if mm.props:
                for prop in sorted(mm.props.keys()):
                    out.append(f"    {mm.props[prop]} {prop};")
            else:
                out.append("    int __dummy;")
            out.append(f"}} {mm.name};")
            out.append("")
        out.append("")

    if funcs:
        out.append("/* ---------------------------- */")
        out.append("/* prototypes (native/forward only) */")
        out.append("/* ---------------------------- */")
        for name in sorted(funcs.keys()):
            f = funcs[name]
            defaults = [p for p in f.params if p.default is not None]
            if defaults:
                pairs = ", ".join([f"{p.name}={p.default}" for p in defaults])
                out.append(f"/* defaults: {pairs} */")
            params = ", ".join([p.cdecl for p in f.params]) if f.params else "void"
            out.append(f"{f.ret} {f.name}({params});")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to .inc file OR include directory")
    ap.add_argument("-o", "--output", default="slim.h", help="output header path")
    args = ap.parse_args()

    paths = collect_input_paths(args.input)

    merged_defines: Dict[str, str] = {}
    merged_consts: List[ConstDecl] = []
    merged_enums: Dict[str, EnumDecl] = {}
    merged_structs: Dict[str, StructDecl] = {}
    merged_methodmaps: Dict[str, MethodmapDecl] = {}
    merged_funcs: Dict[str, FuncDecl] = {}

    for p in paths:
        text = read_file(p)
        defines, consts, enums, structs, methodmaps, funcs = parse_text(text)

        merged_defines.update(defines)
        merged_consts.extend(consts)

        for k, v in enums.items():
            merged_enums[k] = v
        for k, v in structs.items():
            merged_structs[k] = v

        for k, v in methodmaps.items():
            prev = merged_methodmaps.get(k)
            if prev:
                prev.props.update(v.props)
            else:
                merged_methodmaps[k] = v

        for k, v in funcs.items():
            merged_funcs[k] = v

    hdr = render_header(
        merged_defines,
        merged_consts,
        merged_enums,
        merged_structs,
        merged_methodmaps,
        merged_funcs,
        source_name=os.path.abspath(args.input),
    )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(hdr)

    print(f"[ok] wrote {out_path}")
    print(f"  defines:   {len(merged_defines)}")
    print(f"  consts:    {len(merged_consts)}")
    print(f"  enums:     {len(merged_enums)}")
    print(f"  structs:   {len(merged_structs)}")
    print(f"  methodmap: {len(merged_methodmaps)}")
    print(f"  funcs:     {len(merged_funcs)}")


if __name__ == "__main__":
    main()
