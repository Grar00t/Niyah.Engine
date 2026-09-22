#!/usr/bin/env python3
"""Generate a deterministic, mechanically-verifiable Arabic math corpus.

The generator deliberately avoids model-written free-form answers. Every record is
constructed from exact integer/rational arithmetic and carries a verification
payload that can be recomputed independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

SCHEMA = "niyah.verified_math.v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_record(fp, rec: dict) -> None:
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fp.write(raw + "\n")


def base_record(idx: int, task: str, instruction: str, response: str, verification: dict, tags: list[str]) -> dict:
    content = {
        "instruction": instruction,
        "response": response,
        "verification": verification,
    }
    return {
        "schema": SCHEMA,
        "id": f"math-{idx:08d}",
        "domain": "mathematics",
        "task": task,
        "language": "ar",
        "instruction": instruction,
        "response": response,
        "verification": verification,
        "tags": tags,
        "provenance": {
            "source": "deterministic-generator",
            "generator": "tools/generate_verified_math.py",
            "status": "MECHANICALLY_VERIFIABLE",
            "content_sha256": sha256_text(json.dumps(content, ensure_ascii=False, sort_keys=True)),
        },
    }


def make_arithmetic(rng: random.Random, idx: int) -> dict:
    a = rng.randint(-5000, 5000)
    b = rng.randint(-5000, 5000)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
        arabic_op = "+"
    elif op == "-":
        result = a - b
        arabic_op = "−"
    else:
        result = a * b
        arabic_op = "×"
    instruction = f"احسب بدقة: {a} {arabic_op} {b}."
    response = f"الناتج الدقيق هو {result}."
    return base_record(idx, "integer_arithmetic", instruction, response,
                       {"type": "integer_exact", "a": a, "b": b, "op": op, "answer": result},
                       ["arithmetic", "exact"])


def make_fraction(rng: random.Random, idx: int) -> dict:
    a, b = rng.randint(-40, 40), rng.randint(1, 40)
    c, d = rng.randint(-40, 40), rng.randint(1, 40)
    op = rng.choice(["+", "-"])
    x, y = Fraction(a, b), Fraction(c, d)
    z = x + y if op == "+" else x - y
    symbol = "+" if op == "+" else "−"
    instruction = f"بسّط الكسر الناتج عن ({a}/{b}) {symbol} ({c}/{d}) إلى أبسط صورة."
    response = f"بعد توحيد المقامات والاختزال يكون الناتج {z.numerator}/{z.denominator}."
    return base_record(idx, "rational_arithmetic", instruction, response,
                       {"type": "fraction_exact", "a": [a, b], "b": [c, d], "op": op,
                        "answer": [z.numerator, z.denominator]},
                       ["fractions", "exact"])


def make_linear(rng: random.Random, idx: int) -> dict:
    x = rng.randint(-100, 100)
    a = rng.choice([v for v in range(-20, 21) if v != 0])
    b = rng.randint(-100, 100)
    c = a * x + b
    instruction = f"حل المعادلة الخطية {a}x + ({b}) = {c} وأعط قيمة x فقط بعد التحقق."
    response = f"بنقل الحد الثابت ثم القسمة على {a} نحصل على x = {x}. وبالتعويض: {a}×({x}) + ({b}) = {c}."
    return base_record(idx, "linear_equation", instruction, response,
                       {"type": "linear_exact", "a": a, "b": b, "c": c, "answer": x},
                       ["algebra", "linear", "exact"])


def make_gcd(rng: random.Random, idx: int) -> dict:
    a = rng.randint(2, 100000)
    b = rng.randint(2, 100000)
    g = math.gcd(a, b)
    instruction = f"احسب القاسم المشترك الأكبر للعددين {a} و{b}."
    response = f"بتطبيق خوارزمية إقليدس يكون القاسم المشترك الأكبر هو {g}."
    return base_record(idx, "gcd", instruction, response,
                       {"type": "gcd_exact", "a": a, "b": b, "answer": g},
                       ["number_theory", "euclid", "exact"])


def make_mod_inverse(rng: random.Random, idx: int) -> dict:
    while True:
        m = rng.randint(3, 1000)
        a = rng.randint(2, m - 1)
        if math.gcd(a, m) == 1:
            break
    inv = pow(a, -1, m)
    instruction = f"أوجد المعكوس الضربي للعدد {a} بترديد {m}."
    response = f"المعكوس هو {inv} لأن ({a}×{inv}) mod {m} = 1."
    return base_record(idx, "modular_inverse", instruction, response,
                       {"type": "mod_inverse_exact", "a": a, "modulus": m, "answer": inv},
                       ["number_theory", "modular_arithmetic", "exact"])


def make_combinations(rng: random.Random, idx: int) -> dict:
    n = rng.randint(4, 30)
    k = rng.randint(0, n)
    ans = math.comb(n, k)
    instruction = f"كم طريقة لاختيار {k} عناصر من {n} عنصرًا متميزًا عندما لا يهم الترتيب؟"
    response = f"العدد هو C({n},{k}) = {ans}."
    return base_record(idx, "combinations", instruction, response,
                       {"type": "comb_exact", "n": n, "k": k, "answer": ans},
                       ["combinatorics", "exact"])

GENERATORS = [make_arithmetic, make_fraction, make_linear, make_gcd, make_mod_inverse, make_combinations]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--records", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1448)
    args = ap.parse_args()
    if args.records <= 0:
        raise SystemExit("--records must be > 0")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.out.open("w", encoding="utf-8", newline="\n") as fp:
        for i in range(1, args.records + 1):
            gen = GENERATORS[(i - 1) % len(GENERATORS)]
            dump_record(fp, gen(rng, i))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"RECORDS={args.records}")
    print(f"SEED={args.seed}")
    print(f"SHA256={digest}")
    print(f"OUT={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
