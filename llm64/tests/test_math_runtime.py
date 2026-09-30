import ctypes
import math
import sys

lib = ctypes.CDLL(sys.argv[1])

expf = lib.exp_f32
expf.argtypes = [ctypes.c_float]
expf.restype = ctypes.c_float

softmax = lib.softmax_f32
softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
softmax.restype = ctypes.c_int

rope = lib.rope_f32
rope.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
rope.restype = ctypes.c_int


# ---- exp parity ------------------------------------------------------------

for x in (-10.0, -2.0, -0.1, 0.0, 0.1, 2.0, 10.0):
    got = float(expf(x))
    ref = math.exp(x)

    abs_err = abs(got - ref)
    rel_err = abs_err / max(abs(ref), 1e-30)

    assert rel_err < 3e-6, (x, got, ref, rel_err)

print("EXP_F32=PASS")


# ---- softmax parity --------------------------------------------------------

src = [1.0, 2.0, 3.0, -2.0]
arr = (ctypes.c_float * len(src))(*src)

assert softmax(arr, len(src)) == 0

m = max(src)
tmp = [math.exp(x - m) for x in src]
z = sum(tmp)
ref = [x / z for x in tmp]
got = [float(x) for x in arr]

for i, (a, b) in enumerate(zip(got, ref)):
    assert abs(a - b) < 3e-6, (i, a, b)

assert abs(sum(got) - 1.0) < 3e-6

print("SOFTMAX_F32=PASS")


# ---- RoPE parity -----------------------------------------------------------

n_heads = 2
head_dim = 8
position = 3

src = [
     0.1, -0.2,  0.3, -0.4,  0.5, -0.6,  0.7, -0.8,
    -0.9,  1.0, -1.1,  1.2, -1.3,  1.4, -1.5,  1.6,
]

arr = (ctypes.c_float * len(src))(*src)

assert rope(arr, n_heads, head_dim, position) == 0

ref = list(src)

for h in range(n_heads):
    base = h * head_dim

    for i in range(0, head_dim, 2):
        inv_freq = 10000.0 ** (-(i / head_dim))
        angle = float(position) * inv_freq
        c = math.cos(angle)
        ss = math.sin(angle)

        x0 = ref[base + i]
        x1 = ref[base + i + 1]

        ref[base + i]     = x0 * c - x1 * ss
        ref[base + i + 1] = x0 * ss + x1 * c

got = [float(x) for x in arr]

max_abs = max(abs(a - b) for a, b in zip(got, ref))
assert max_abs < 1e-5, max_abs

print("ROPE_F32=PASS")
print(f"ROPE_MAX_ABS={max_abs:.9g}")

# ---- SwiGLU parity ---------------------------------------------------------
swiglu = lib.swiglu_f32
swiglu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
swiglu.restype = ctypes.c_int
gate_src = [-4.0, -1.0, 0.0, 0.5, 2.0, 5.0]
up_src   = [0.5, 2.0, 3.0, -1.0, 0.25, 1.5]
gate = (ctypes.c_float * len(gate_src))(*gate_src)
up = (ctypes.c_float * len(up_src))(*up_src)
assert swiglu(gate, up, len(gate_src)) == 0
ref = [(x / (1.0 + math.exp(-x))) * u for x, u in zip(gate_src, up_src)]
got = [float(x) for x in gate]
max_abs = max(abs(a-b) for a,b in zip(got, ref))
assert max_abs < 2e-5, (max_abs, got, ref)
assert swiglu(None, up, len(gate_src)) == 1
assert swiglu(gate, None, len(gate_src)) == 1
assert swiglu(gate, up, 0) == 1
print("SWIGLU_F32=PASS")
print(f"SWIGLU_MAX_ABS={max_abs:.9g}")
print("MATH_RUNTIME=PASS")
