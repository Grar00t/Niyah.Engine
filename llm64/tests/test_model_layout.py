import ctypes
import struct
import sys

lib = ctypes.CDLL(sys.argv[1])
layout_fn = lib.model_layout_from_header
layout_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
layout_fn.restype = ctypes.c_int
layer_fn = lib.model_layer_layout_from_header
layer_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
layer_fn.restype = ctypes.c_int

def header(vocab, context, dim, layers, heads, kv_heads, ffn, eps, tie, segments):
    b = bytearray(72)
    b[0:8] = b'NIYAHCKP'
    struct.pack_into('<I', b, 8, 2)
    struct.pack_into('<I', b, 24, vocab)
    struct.pack_into('<I', b, 28, context)
    struct.pack_into('<I', b, 32, dim)
    struct.pack_into('<I', b, 36, layers)
    struct.pack_into('<I', b, 40, heads)
    struct.pack_into('<I', b, 44, kv_heads)
    struct.pack_into('<I', b, 48, ffn)
    struct.pack_into('<f', b, 52, eps)
    struct.pack_into('<I', b, 56, tie)
    struct.pack_into('<I', b, 60, segments)
    return b

def ref(cfg):
    vocab, context, dim, layers, heads, kv_heads, ffn, eps, tie, segments = cfg
    head_dim = dim // heads
    kv_dim = head_dim * kv_heads
    token = vocab * dim
    segment = segments * dim
    q = dim * dim
    kv = kv_dim * dim
    fu = ffn * dim
    fd = dim * ffn
    layer_stride = dim + q + kv + kv + q + dim + fu + fu + fd
    layer_base = token + segment
    final_norm = layer_base + layer_stride * layers
    lm_head = 0 if tie else final_norm + dim
    total = final_norm + dim + (0 if tie else vocab * dim)
    return [0, token, layer_base, layer_stride, final_norm, lm_head, total, head_dim, kv_dim]

def layer_ref(cfg, L, i):
    dim = cfg[2]
    ffn = cfg[6]
    q = dim * dim
    kv = L[8] * dim
    fu = ffn * dim
    fd = dim * ffn
    c = L[2] + i * L[3]
    out = []
    for n in (dim, q, kv, kv, q, dim, fu, fu, fd):
        out.append(c)
        c += n
    return out

cases = [
    (258, 128, 16, 2, 4, 2, 32, 1e-5, 1, 0),
    (1024, 512, 64, 3, 8, 2, 192, 1e-6, 0, 4),
    (1756, 1024, 128, 6, 8, 4, 384, 1e-5, 1, 2),
]

for cfg in cases:
    hb = ctypes.create_string_buffer(bytes(header(*cfg)))
    out = (ctypes.c_uint64 * 9)()
    rc = layout_fn(ctypes.addressof(hb), out)
    assert rc == 0, (cfg, rc)
    got = list(out)
    want = ref(cfg)
    assert got == want, (cfg, got, want)
    for i in range(cfg[3]):
        lo = (ctypes.c_uint64 * 9)()
        rc = layer_fn(ctypes.addressof(hb), out, i, lo)
        assert rc == 0, (cfg, i, rc)
        expected = layer_ref(cfg, want, i)
        assert list(lo) == expected, (cfg, i, list(lo), expected)

bad = header(258, 128, 18, 2, 4, 2, 32, 1e-5, 1, 0)
hb = ctypes.create_string_buffer(bytes(bad))
out = (ctypes.c_uint64 * 9)()
assert layout_fn(ctypes.addressof(hb), out) == 1

good = cases[0]
hb = ctypes.create_string_buffer(bytes(header(*good)))
assert layout_fn(ctypes.addressof(hb), out) == 0
lo = (ctypes.c_uint64 * 9)()
assert layer_fn(ctypes.addressof(hb), out, good[3], lo) == 1

print('MODEL_LAYOUT=PASS')
print('MODEL_LAYER_LAYOUT=PASS')
print('LAYOUT_CASES=%d' % len(cases))
