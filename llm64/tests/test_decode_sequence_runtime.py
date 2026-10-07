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

VOCAB = 9
CONTEXT = 4
DIM = 4
LAYERS = 2
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


def ff_add(a, b):
    return f32(f32(a) + f32(b))


def ff_mul(a, b):
    return f32(f32(a) * f32(b))


def model_layout():
    token = 0
    segment = token + VOCAB * DIM
    layers = segment + SEGMENTS * DIM
    q_count = DIM * DIM
    kv_count = KV_DIM * DIM
    fm = FFN * DIM
    stride = DIM + q_count + kv_count + kv_count + q_count + DIM + fm + fm + fm
    final_norm = layers + LAYERS * stride
    lm_head = token if TIE else final_norm + DIM
    total = final_norm + DIM
    if not TIE:
        total += VOCAB * DIM
    return {
        "token": token, "layers": layers, "stride": stride,
        "final_norm": final_norm, "lm_head": lm_head, "total": total,
    }


L = model_layout()


def layer_layout(index):
    x = L["layers"] + index * L["stride"]
    out = {"attn_norm": x}
    x += DIM
    out["wq"] = x; x += DIM * DIM
    out["wk"] = x; x += KV_DIM * DIM
    out["wv"] = x; x += KV_DIM * DIM
    out["wo"] = x; x += DIM * DIM
    out["ffn_norm"] = x; x += DIM
    out["w_gate"] = x; x += FFN * DIM
    out["w_up"] = x; x += FFN * DIM
    out["w_down"] = x; x += DIM * FFN
    assert x == L["layers"] + (index + 1) * L["stride"]
    return out


LAYOUTS = [layer_layout(i) for i in range(LAYERS)]
weights = [f32((((i * 19) % 37) - 18) * 0.009) for i in range(L["total"])]

for layer_index, ll in enumerate(LAYOUTS):
    for i in range(DIM):
        weights[ll["attn_norm"] + i] = f32(0.91 + 0.02 * i + 0.01 * layer_index)
        weights[ll["ffn_norm"] + i] = f32(1.03 - 0.015 * i + 0.005 * layer_index)

for i in range(DIM):
    weights[L["final_norm"] + i] = f32(0.96 + 0.01 * i)

W = (ctypes.c_float * len(weights))(*weights)


def matvec(off, x, rows, cols):
    out = []
    for r in range(rows):
        acc = f32(0.0)
        base = off + r * cols
        for col in range(cols):
            acc = ff_add(acc, ff_mul(weights[base + col], x[col]))
        out.append(acc)
    return out


def rmsnorm(x, off):
    ss = 0.0
    for value in x:
        z = float(f32(value))
        ss += z * z
    mean = f32(ss / len(x))
    denom = f32(math.sqrt(f32(mean + f32(EPS))))
    inv = f32(f32(1.0) / denom)
    return [ff_mul(ff_mul(value, inv), weights[off + i]) for i, value in enumerate(x)]


def rope(vec, n_heads, position):
    out = list(vec)
    freq_mult = f32(math.exp(f32(-18.420680743952367 / HEAD_DIM)))
    pos = f32(position)
    for head in range(n_heads):
        inv = f32(1.0)
        base = head * HEAD_DIM
        for i in range(0, HEAD_DIM, 2):
            angle = ff_mul(pos, inv)
            c = f32(math.cos(angle))
            s = f32(math.sin(angle))
            x0 = out[base + i]
            x1 = out[base + i + 1]
            out[base + i] = ff_add(ff_mul(x0, c), -ff_mul(x1, s))
            out[base + i + 1] = ff_add(ff_mul(x0, s), ff_mul(x1, c))
            inv = ff_mul(inv, freq_mult)
    return out


def softmax(xs):
    maximum = max(xs)
    exp_values = []
    total = f32(0.0)
    for value in xs:
        e = f32(math.exp(f32(value - maximum)))
        exp_values.append(e)
        total = ff_add(total, e)
    return [f32(e / total) for e in exp_values]


def attention(q, keys, values, position):
    group = HEADS // KV_HEADS
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    out = [f32(0.0)] * DIM
    for head in range(HEADS):
        kv_head = head // group
        qh = q[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        scores = []
        for source in range(position + 1):
            base = source * KV_DIM + kv_head * HEAD_DIM
            dot = f32(0.0)
            for d in range(HEAD_DIM):
                dot = ff_add(dot, ff_mul(qh[d], keys[base + d]))
            scores.append(ff_mul(dot, scale))
        probs = softmax(scores)
        for d in range(HEAD_DIM):
            value = f32(0.0)
            for source in range(position + 1):
                base = source * KV_DIM + kv_head * HEAD_DIM
                value = ff_add(value, ff_mul(probs[source], values[base + d]))
            out[head * HEAD_DIM + d] = value
    return out


def silu_mul(gate, up):
    exp_value = f32(math.exp(f32(-gate)))
    return ff_mul(f32(gate / ff_add(1.0, exp_value)), up)


reference_keys = [[f32(0.0)] * (CONTEXT * KV_DIM) for _ in range(LAYERS)]
reference_values = [[f32(0.0)] * (CONTEXT * KV_DIM) for _ in range(LAYERS)]


def reference_decode(token, position):
    hidden = list(weights[L["token"] + token * DIM:L["token"] + (token + 1) * DIM])

    for layer_index, ll in enumerate(LAYOUTS):
        norm = rmsnorm(hidden, ll["attn_norm"])
        q = rope(matvec(ll["wq"], norm, DIM, DIM), HEADS, position)
        k = rope(matvec(ll["wk"], norm, KV_DIM, DIM), KV_HEADS, position)
        v = matvec(ll["wv"], norm, KV_DIM, DIM)

        base = position * KV_DIM
        reference_keys[layer_index][base:base + KV_DIM] = k
        reference_values[layer_index][base:base + KV_DIM] = v

        attn = attention(
            q,
            reference_keys[layer_index],
            reference_values[layer_index],
            position,
        )
        proj = matvec(ll["wo"], attn, DIM, DIM)
        hidden = [ff_add(a, b) for a, b in zip(hidden, proj)]

        norm = rmsnorm(hidden, ll["ffn_norm"])
        gate = matvec(ll["w_gate"], norm, FFN, DIM)
        up = matvec(ll["w_up"], norm, FFN, DIM)
        gate = [silu_mul(a, b) for a, b in zip(gate, up)]
        proj = matvec(ll["w_down"], gate, DIM, FFN)
        hidden = [ff_add(a, b) for a, b in zip(hidden, proj)]

    norm = rmsnorm(hidden, L["final_norm"])
    return matvec(L["lm_head"], norm, VOCAB, DIM)


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

tokens = [3, 5, 2]
max_logit_error = 0.0
max_cache_error = 0.0

for expected_position, token in enumerate(tokens):
    reference_logits = reference_decode(token, expected_position)
    rc = decode(
        ctypes.addressof(H),
        W,
        ctypes.byref(rt),
        ctypes.byref(position),
        token,
    )
    assert rc == 0, (expected_position, rc)
    assert position.value == expected_position + 1, position.value

    actual_logits = list((ctypes.c_float * VOCAB).from_address(rt.logits))
    logit_error = max(abs(float(a) - float(b)) for a, b in zip(actual_logits, reference_logits))
    max_logit_error = max(max_logit_error, logit_error)
    assert logit_error < 4.0e-4, (expected_position, logit_error, actual_logits, reference_logits)

    total_cache = LAYERS * CONTEXT * KV_DIM
    actual_keys = list((ctypes.c_float * total_cache).from_address(rt.keys))
    actual_values = list((ctypes.c_float * total_cache).from_address(rt.values))
    for layer_index in range(LAYERS):
        layer_base = layer_index * CONTEXT * KV_DIM
        used = (expected_position + 1) * KV_DIM
        for offset in range(used):
            key_error = abs(actual_keys[layer_base + offset] - reference_keys[layer_index][offset])
            value_error = abs(actual_values[layer_base + offset] - reference_values[layer_index][offset])
            max_cache_error = max(max_cache_error, key_error, value_error)

assert max_cache_error < 3.0e-5, max_cache_error

position.value = CONTEXT
assert decode(
    ctypes.addressof(H),
    W,
    ctypes.byref(rt),
    ctypes.byref(position),
    1,
) == 1
assert position.value == CONTEXT

assert destroy(ctypes.byref(rt)) == 0

print("DECODE_SEQUENCE=PASS")
print("DECODE_KV_REUSE=PASS")
print("DECODE_CONTEXT_BOUNDARY=PASS")
print(f"DECODE_SEQUENCE_LOGITS_MAX_ABS={max_logit_error:.9g}")
print(f"DECODE_SEQUENCE_CACHE_MAX_ABS={max_cache_error:.9g}")
