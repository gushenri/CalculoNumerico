

from __future__ import annotations
import math
from typing import Callable, Dict, Any, List, Tuple, Optional
import re
import sys
import csv
import os

ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
ALLOWED_NAMES.update({
    "abs": abs
})

def safe_eval(expr: str, x: float) -> float:
    """Avalia expr(x) com dicionário de nomes permitido."""
    # Permite usar ^ como potência, trocando por **
    expr = expr.replace("^", "**")
    try:
        return float(eval(expr, {"__builtins__": {}}, {**ALLOWED_NAMES, "x": x}))
    except Exception as e:
        raise ValueError(f"Erro ao avaliar expressão '{expr}' em x={x}: {e}")

def parse_input_txt(path: str) -> Dict[str, Any]:
   
    params: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = [s.strip() for s in line.split("=", 1)]
            # Tenta converter números automaticamente
            if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", v):
                try:
                    params[k] = float(v)
                    continue
                except:
                    pass
            # Vetores simples [a,b]
            if v.startswith("[") and v.endswith("]"):
                inside = v[1:-1].strip()
                parts = [p.strip() for p in inside.split(",")]
                arr = []
                for p in parts:
                    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", p):
                        arr.append(float(p))
                    else:
                        arr.append(p)
                params[k] = arr
                continue
            params[k] = v
    return params

def bisseccao(f_expr: str, a: float, b: float, tol: float, max_iter: int) -> List[Dict[str, float]]:
    rows = []
    fa = safe_eval(f_expr, a)
    fb = safe_eval(f_expr, b)
    if fa * fb > 0:
        raise ValueError("Bissecção requer f(a) e f(b) de sinais opostos.")
    x_prev = None
    for k in range(1, int(max_iter) + 1):
        c = (a + b) / 2.0
        fc = safe_eval(f_expr, c)
        err = abs(c - x_prev) if x_prev is not None else float("nan")
        rows.append({"iter": k, "a": a, "b": b, "x": c, "f(x)": fc, "erro": err})
        if abs(fc) < tol or (x_prev is not None and abs(c - x_prev) < tol):
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        x_prev = c
    return rows

def regula_falsi(f_expr: str, a: float, b: float, tol: float, max_iter: int) -> List[Dict[str, float]]:
    rows = []
    fa = safe_eval(f_expr, a)
    fb = safe_eval(f_expr, b)
    if fa * fb > 0:
        raise ValueError("Regula Falsi requer f(a) e f(b) de sinais opostos.")
    x_prev = None
    for k in range(1, int(max_iter) + 1):
        # Interpolação linear
        c = b - fb * (b - a) / (fb - fa)
        fc = safe_eval(f_expr, c)
        err = abs(c - x_prev) if x_prev is not None else float("nan")
        rows.append({"iter": k, "a": a, "b": b, "x": c, "f(x)": fc, "erro": err})
        if abs(fc) < tol or (x_prev is not None and abs(c - x_prev) < tol):
            break
        # Atualiza mantendo a mudança de sinal
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        x_prev = c
    return rows

def iterativo_linear(g_expr: str, x0: float, tol: float, max_iter: int) -> List[Dict[str, float]]:
    rows = []
    x_prev = x0
    for k in range(1, int(max_iter) + 1):
        x = safe_eval(g_expr, x_prev)
        err = abs(x - x_prev)
        rows.append({"iter": k, "x": x, "g(x_prev)": x, "erro": err})
        if err < tol:
            break
        x_prev = x
    return rows

def newton(f_expr: str, df_expr: str, x0: float, tol: float, max_iter: int) -> List[Dict[str, float]]:
    rows = []
    x = x0
    for k in range(1, int(max_iter) + 1):
        fx = safe_eval(f_expr, x)
        dfx = safe_eval(df_expr, x)
        if dfx == 0:
            raise ZeroDivisionError("Derivada zero no método de Newton.")
        x_new = x - fx / dfx
        err = abs(x_new - x)
        rows.append({"iter": k, "x": x_new, "f(x)": fx, "f'(x)": dfx, "erro": err})
        if abs(fx) < tol or err < tol:
            x = x_new
            break
        x = x_new
    return rows

def secante(f_expr: str, x0: float, x1: float, tol: float, max_iter: int) -> List[Dict[str, float]]:
    rows = []
    x_prev, x = x0, x1
    f_prev = safe_eval(f_expr, x_prev)
    fx = safe_eval(f_expr, x)
    for k in range(1, int(max_iter) + 1):
        denom = (fx - f_prev)
        if denom == 0:
            raise ZeroDivisionError("Denominador zero no método da Secante.")
        x_new = x - fx*(x - x_prev)/denom
        err = abs(x_new - x)
        rows.append({"iter": k, "x_{k-1}": x_prev, "x_k": x, "x_{k+1}": x_new, "f(x_k)": fx, "erro": err})
        if abs(fx) < tol or err < tol:
            x = x_new
            break
        x_prev, f_prev = x, fx
        x, fx = x_new, safe_eval(f_expr, x_new)
    return rows

def format_table(rows: List[Dict[str, float]], headers: List[str]) -> str:
    if not rows:
        return "(sem iterações)\n"
    # Garante colunas
    headers = headers or sorted({k for r in rows for k in r.keys()})
    # Larguras
    widths = {h: max(len(h), *(len(f"{r.get(h, ''):.10g}") if isinstance(r.get(h, ""), (int, float)) else len(str(r.get(h, ""))) for r in rows)) for h in headers}
    # Monta linhas
    sep = "+".join("-"*(widths[h]+2) for h in headers)
    lines = []
    # Header
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-+-".join("-"*widths[h] for h in headers))
    # Rows
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                sval = f"{v:.10g}"
            else:
                sval = str(v)
            vals.append(sval.rjust(widths[h]) if isinstance(v, (int, float)) else sval.ljust(widths[h]))
        lines.append(" | ".join(vals))
    return "\n".join(lines) + "\n"

def write_txt_and_csv(out_txt: str, out_csv: str, all_results: Dict[str, List[Dict[str, float]]]):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Resultados dos Métodos de Raízes\n\n")
        for method, rows in all_results.items():
            f.write(f"## {method}\n")
            if rows:
                headers = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x!="iter", x))
                table = format_table(rows, headers=headers)
                f.write(table + "\n")
            else:
                f.write("(método não executado ou sem iterações)\n\n")
    # CSV consolidado
    # Normaliza chaves por linha
    fieldnames = sorted({k for rows in all_results.values() for r in rows for k in r.keys()} | {"method"} , key=lambda x: (x!="method" and x!="iter", x))
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for method, rows in all_results.items():
            for r in rows:
                rr = {"method": method}
                rr.update(r)
                w.writerow(rr)

def main():
    if len(sys.argv) < 2:
        print("Uso: python metodos_cap2.py /caminho/para/dados.txt")
        sys.exit(1)
    in_path = sys.argv[1]
    params = parse_input_txt(in_path)

    # Parâmetros globais
    f_expr = params.get("function", None)
    df_expr = params.get("derivative", None)
    g_expr = params.get("g", None)
    a = params.get("a", None)
    b = params.get("b", None)
    x0 = params.get("x0", None)
    x1 = params.get("x1", None)
    tol = float(params.get("tol", 1e-6))
    max_iter = int(params.get("max_iter", 100))

    results: Dict[str, List[Dict[str, float]]] = {}

    # Bissecção
    try:
        if f_expr is not None and a is not None and b is not None:
            results["Bissecção"] = bisseccao(f_expr, float(a), float(b), tol, max_iter)
        else:
            results["Bissecção"] = []
    except Exception as e:
        results["Bissecção"] = [{"iter": 0, "erro_msg": str(e)}]

    # Iterativo Linear (Ponto Fixo)
    try:
        if g_expr is not None and x0 is not None:
            results["Iterativo Linear (Ponto Fixo)"] = iterativo_linear(g_expr, float(x0), tol, max_iter)
        else:
            results["Iterativo Linear (Ponto Fixo)"] = []
    except Exception as e:
        results["Iterativo Linear (Ponto Fixo)"] = [{"iter": 0, "erro_msg": str(e)}]

    # Newton
    try:
        if f_expr is not None and df_expr is not None and x0 is not None:
            results["Newton"] = newton(f_expr, df_expr, float(x0), tol, max_iter)
        else:
            results["Newton"] = []
    except Exception as e:
        results["Newton"] = [{"iter": 0, "erro_msg": str(e)}]

    # Secante
    try:
        if f_expr is not None and x0 is not None and x1 is not None:
            results["Secante"] = secante(f_expr, float(x0), float(x1), tol, max_iter)
        else:
            results["Secante"] = []
    except Exception as e:
        results["Secante"] = [{"iter": 0, "erro_msg": str(e)}]

    # Regula Falsi
    try:
        if f_expr is not None and a is not None and b is not None:
            results["Regula Falsi"] = regula_falsi(f_expr, float(a), float(b), tol, max_iter)
        else:
            results["Regula Falsi"] = []
    except Exception as e:
        results["Regula Falsi"] = [{"iter": 0, "erro_msg": str(e)}]

    # Saídas
    out_dir = os.path.dirname(os.path.abspath(in_path))
    out_txt = os.path.join(out_dir, "resultados.txt")
    out_csv = os.path.join(out_dir, "resultados.csv")
    write_txt_and_csv(out_txt, out_csv, results)
    print(f"OK\n{out_txt}\n{out_csv}")

if __name__ == "__main__":
    main()
