import ctypes
import math
import sys

lib = ctypes.CDLL(sys.argv[1])

kv_store = lib.kv_store_f32
kv_store.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
kv_store.restype = ctypes.c_int

attn = lib.attention_gqa_f32
attn.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
]
attn.restype = ctypes.c_int


n_heads = 4
n_kv_heads = 2
head_dim = 2

dim = n_heads * head_dim
kv_dim = n_kv_heads * head_dim

context = 3
position = 2

keys = (ctypes.c_float * (context * kv_dim))()
vals = (ctypes.c_float * (context * kv_dim))()

K = [
    [0.10,  0.20,  0.30,  0.40],
    [0.50, -0.20,  0.10,  0.70],
    [0.25,  0.15, -0.35,  0.45],
]

V = [
    [1.00,  2.00,  3.00,  4.00],
    [2.00, -1.00,  0.50,  1.50],
    [0.25,  0.75, -2.00,  1.00],
]

for pos in range(context):
    k = (ctypes.c_float * kv_dim)(*K[pos])
    v = (ctypes.c_float * kv_dim)(*V[pos])

    rc = kv_store(
        keys,
        vals,
        context,
        kv_dim,
        pos,
        k,
        v,
    )

    assert rc == 0, ("kv_store", pos, rc)

for pos in range(context):
    got_k = [
        float(keys[pos * kv_dim + i])
        for i in range(kv_dim)
    ]
    got_v = [
        float(vals[pos * kv_dim + i])
        for i in range(kv_dim)
    ]

    k_err = max(abs(a - b) for a, b in zip(got_k, K[pos]))
    v_err = max(abs(a - b) for a, b in zip(got_v, V[pos]))

    assert k_err < 1e-6, (pos, k_err, got_k, K[pos])
    assert v_err < 1e-6, (pos, v_err, got_v, V[pos])

print("KV_STORE_F32=PASS")


Q = [
     0.4, -0.1,
     0.7,  0.2,
    -0.3,  0.8,
     0.6, -0.5,
]

q = (ctypes.c_float * dim)(*Q)
out = (ctypes.c_float * dim)()
scores = (ctypes.c_float * context)()

rc = attn(
    out,
    q,
    keys,
    vals,
    n_heads,
    n_kv_heads,
    head_dim,
    position,
    scores,
)

assert rc == 0, ("attention", rc)


# Python reference matching Niyah GQA semantics.
group_size = n_heads // n_kv_heads
scale = 1.0 / math.sqrt(head_dim)

ref = [0.0] * dim

for head in range(n_heads):
    kv_head = head // group_size

    qh = Q[
        head * head_dim:
        (head + 1) * head_dim
    ]

    logits = []

    for source in range(position + 1):
        kh = K[source][
            kv_head * head_dim:
            (kv_head + 1) * head_dim
        ]

        dot = sum(
            qh[d] * kh[d]
            for d in range(head_dim)
        )

        logits.append(dot * scale)

    m = max(logits)
    probs = [math.exp(x - m) for x in logits]
    z = sum(probs)
    probs = [x / z for x in probs]

    for d in range(head_dim):
        value = 0.0

        for source in range(position + 1):
            vh = V[source][
                kv_head * head_dim:
                (kv_head + 1) * head_dim
            ]

            value += probs[source] * vh[d]

        ref[head * head_dim + d] = value


got = [float(x) for x in out]

max_abs = max(
    abs(a - b)
    for a, b in zip(got, ref)
)

assert max_abs < 1e-5, (max_abs, got, ref)

print("ATTENTION_GQA_F32=PASS")
print(f"ATTENTION_MAX_ABS={max_abs:.9g}")
print("ATTENTION_RUNTIME=PASS")


# Geometry rejection.
bad = attn(
    out,
    q,
    keys,
    vals,
    3,             # not divisible by n_kv_heads=2
    2,
    head_dim,
    position,
    scores,
)

assert bad == 1

print("ATTENTION_INVALID_GEOMETRY=PASS")
