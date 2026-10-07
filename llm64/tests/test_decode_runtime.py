import ctypes
import math
import struct
import sys

lib = ctypes.CDLL(sys.argv[1])

class Runtime(ctypes.Structure):
    _fields_ = [
        ("keys", ctypes.c_void_p),
        ("values", ctypes.c_void_p),
        ("kv_bytes", ctypes.c_size_t),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_size_t),
        ("logits", ctypes.c_void_p),
        ("logits_bytes", ctypes.c_size_t),
    ]

create = lib.runtime_memory_create
create.argtypes = [ctypes.c_void_p, ctypes.POINTER(Runtime)]
create.restype = ctypes.c_int

destroy = lib.runtime_memory_destroy
destroy.argtypes = [ctypes.POINTER(Runtime)]
destroy.restype = ctypes.c_int

decode = lib.decode_token_f32
decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(Runtime),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint32,
]
decode.restype = ctypes.c_int

argmax = lib.argmax_f32
argmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
argmax.restype = ctypes.c_uint32


VOCAB = 8
CONTEXT = 4
DIM = 4
LAYERS = 1
HEADS = 2
KV_HEADS = 1
FFN = 8
EPS = 1.0e-5
TIE = 1
SEGMENTS = 0

HEAD_DIM = DIM // HEADS
KV_DIM = HEAD_DIM * KV_HEADS


def f32(x):
    return ctypes.c_float(float(x)).value


def layout():
    token = 0
    segment = token + VOCAB * DIM
    layers = segment + SEGMENTS * DIM

    q_count = DIM * DIM
    kv_count = KV_DIM * DIM
    fm = FFN * DIM

    stride = (
        DIM +
        q_count +
        kv_count +
        kv_count +
        q_count +
        DIM +
        fm + fm + fm
    )

    final_norm = layers + LAYERS * stride
    lm_head = token if TIE else final_norm + DIM

    total = final_norm + DIM
    if not TIE:
        total += VOCAB * DIM

    return {
        "token": token,
        "layers": layers,
        "stride": stride,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


L = layout()


def layer_layout(index):
    x = L["layers"] + index * L["stride"]
    out = {}

    out["attn_norm"] = x
    x += DIM

    out["wq"] = x
    x += DIM * DIM

    out["wk"] = x
    x += KV_DIM * DIM

    out["wv"] = x
    x += KV_DIM * DIM

    out["wo"] = x
    x += DIM * DIM

    out["ffn_norm"] = x
    x += DIM

    out["w_gate"] = x
    x += FFN * DIM

    out["w_up"] = x
    x += FFN * DIM

    out["w_down"] = x
    x += DIM * FFN

    assert x == L["layers"] + (index + 1) * L["stride"]
    return out


LL = layer_layout(0)

weights = [
    f32((((i * 17) % 29) - 14) * 0.0125)
    for i in range(L["total"])
]

# Keep norm scales well-conditioned.
for i in range(DIM):
    weights[LL["attn_norm"] + i] = f32(0.90 + 0.03 * i)
    weights[LL["ffn_norm"] + i]  = f32(1.05 - 0.02 * i)
    weights[L["final_norm"] + i] = f32(0.95 + 0.01 * i)

W = (ctypes.c_float * len(weights))(*weights)


def ff_add(a, b):
    return f32(f32(a) + f32(b))


def ff_mul(a, b):
    return f32(f32(a) * f32(b))


def matvec(matrix_off, x, rows, cols):
    out = []
    for r in range(rows):
        acc = f32(0.0)
        base = matrix_off + r * cols
        for c in range(cols):
            acc = ff_add(acc, ff_mul(weights[base + c], x[c]))
        out.append(acc)
    return out


def rmsnorm(x, off):
    ss = 0.0
    for v in x:
        z = float(f32(v))
        ss += z * z

    mean = f32(ss / len(x))
    denom = f32(math.sqrt(f32(mean + f32(EPS))))
    inv = f32(f32(1.0) / denom)

    return [
        ff_mul(ff_mul(v, inv), weights[off + i])
        for i, v in enumerate(x)
    ]


def rope(vec, n_heads, position):
    out = list(vec)
    freq_mult = f32(math.exp(f32(-18.420680743952367 / HEAD_DIM)))
    pos = f32(position)

    for h in range(n_heads):
        inv = f32(1.0)
        base = h * HEAD_DIM

        for i in range(0, HEAD_DIM, 2):
            angle = ff_mul(pos, inv)
            c = f32(math.cos(angle))
            s = f32(math.sin(angle))

            x0 = out[base + i]
            x1 = out[base + i + 1]

            y0 = ff_add(ff_mul(x0, c), -ff_mul(x1, s))
            y1 = ff_add(ff_mul(x0, s),  ff_mul(x1, c))

            out[base + i] = y0
            out[base + i + 1] = y1

            inv = ff_mul(inv, freq_mult)

    return out


def softmax(xs):
    m = max(xs)
    ex = []
    total = f32(0.0)

    for x in xs:
        e = f32(math.exp(f32(x - m)))
        ex.append(e)
        total = ff_add(total, e)

    return [f32(e / total) for e in ex]


def attention(q, keys, values, position):
    group = HEADS // KV_HEADS
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    out = [f32(0.0)] * DIM

    for head in range(HEADS):
        kv_head = head // group
        qh = q[head * HEAD_DIM:(head + 1) * HEAD_DIM]

        scores = []

        for source in range(position + 1):
            kb = source * KV_DIM + kv_head * HEAD_DIM
            acc = f32(0.0)

            for d in range(HEAD_DIM):
                acc = ff_add(
                    acc,
                    ff_mul(qh[d], keys[kb + d])
                )

            scores.append(ff_mul(acc, scale))

        probs = softmax(scores)

        for d in range(HEAD_DIM):
            acc = f32(0.0)

            for source in range(position + 1):
                vb = source * KV_DIM + kv_head * HEAD_DIM

                acc = ff_add(
                    acc,
                    ff_mul(
                        probs[source],
                        values[vb + d]
                    )
                )

            out[head * HEAD_DIM + d] = acc

    return out


def silu_mul(g, u):
    e = f32(math.exp(f32(-g)))
    den = ff_add(1.0, e)
    sig = f32(g / den)
    return ff_mul(sig, u)


# Header with canonical count.
header = struct.pack(
    "<8s" + "I"*11 + "fIIQ",
    b"NIYAHCKP",
    2, 0, 5, 0,
    VOCAB, CONTEXT, DIM, LAYERS,
    HEADS, KV_HEADS, FFN,
    EPS, TIE, SEGMENTS,
    L["total"],
)

H = ctypes.create_string_buffer(header)

rt = Runtime()
assert create(ctypes.addressof(H), ctypes.byref(rt)) == 0

position = ctypes.c_uint64(0)
token = 3

rc = decode(
    ctypes.addressof(H),
    W,
    ctypes.byref(rt),
    ctypes.byref(position),
    token,
)

assert rc == 0, rc
assert position.value == 1


# ------------------------------------------------------------------
# Python reference for position 0
# ------------------------------------------------------------------

hidden = list(
    weights[
        L["token"] + token * DIM:
        L["token"] + (token + 1) * DIM
    ]
)

norm = rmsnorm(hidden, LL["attn_norm"])

q = matvec(LL["wq"], norm, DIM, DIM)
k = matvec(LL["wk"], norm, KV_DIM, DIM)
v = matvec(LL["wv"], norm, KV_DIM, DIM)

q = rope(q, HEADS, 0)
k = rope(k, KV_HEADS, 0)

keys = list(k)
values = list(v)

att = attention(q, keys, values, 0)

proj = matvec(LL["wo"], att, DIM, DIM)
hidden = [ff_add(a, b) for a, b in zip(hidden, proj)]

norm = rmsnorm(hidden, LL["ffn_norm"])

gate = matvec(LL["w_gate"], norm, FFN, DIM)
up   = matvec(LL["w_up"],   norm, FFN, DIM)

gate = [
    silu_mul(g, u)
    for g, u in zip(gate, up)
]

proj = matvec(LL["w_down"], gate, DIM, FFN)
hidden = [ff_add(a, b) for a, b in zip(hidden, proj)]

norm = rmsnorm(hidden, L["final_norm"])
ref_logits = matvec(L["lm_head"], norm, VOCAB, DIM)

got_logits = list(
    (ctypes.c_float * VOCAB).from_address(rt.logits)
)

max_abs = max(
    abs(float(a) - float(b))
    for a, b in zip(got_logits, ref_logits)
)

assert max_abs < 2.0e-4, (max_abs, got_logits, ref_logits)

got_argmax = int(
    argmax(
        ctypes.cast(rt.logits, ctypes.POINTER(ctypes.c_float)),
        VOCAB,
    )
)

ref_argmax = max(
    range(VOCAB),
    key=lambda i: ref_logits[i],
)

assert got_argmax == ref_argmax, (
    got_argmax,
    ref_argmax,
    got_logits,
    ref_logits,
)

# Compare written KV position 0.
got_k = list(
    (ctypes.c_float * KV_DIM).from_address(rt.keys)
)

got_v = list(
    (ctypes.c_float * KV_DIM).from_address(rt.values)
)

kv_err = max(
    max(abs(a-b) for a,b in zip(got_k, k)),
    max(abs(a-b) for a,b in zip(got_v, v)),
)

assert kv_err < 2.0e-5, kv_err

print("DECODE_TOKEN_F32=PASS")
print(f"DECODE_LOGITS_MAX_ABS={max_abs:.9g}")
print(f"DECODE_KV_MAX_ABS={kv_err:.9g}")
print(f"ARGMAX_TOKEN={got_argmax}")
print("DECODE_ARGMAX=PASS")

assert destroy(ctypes.byref(rt)) == 0

print("DECODE_RUNTIME=PASS")
