import ctypes
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

VOCAB=258
CONTEXT=8
DIM=4
LAYERS=1
HEADS=2
KV_HEADS=1
FFN=8

header = struct.pack(
    "<8s" + "I"*11 + "fIIQ",
    b"NIYAHCKP",
    2,0,5,0,
    VOCAB,CONTEXT,DIM,LAYERS,
    HEADS,KV_HEADS,FFN,
    1e-5,1,0,0,
)

buf = ctypes.create_string_buffer(header)
rt = Runtime()

assert create(ctypes.addressof(buf), ctypes.byref(rt)) == 0

head_dim = DIM // HEADS
kv_dim = head_dim * KV_HEADS

expected_kv = LAYERS * CONTEXT * kv_dim * 4
expected_ws = (5*DIM + 2*kv_dim + 2*FFN + CONTEXT) * 4
expected_logits = VOCAB * 4

assert rt.keys
assert rt.values
assert rt.workspace
assert rt.logits

assert rt.kv_bytes == expected_kv, (rt.kv_bytes, expected_kv)
assert rt.workspace_bytes == expected_ws, (rt.workspace_bytes, expected_ws)
assert rt.logits_bytes == expected_logits, (rt.logits_bytes, expected_logits)

# Anonymous mmap must be writable and initially zero.
x = (ctypes.c_float * (rt.kv_bytes // 4)).from_address(rt.keys)
assert all(v == 0.0 for v in x)
x[0] = 3.25
assert x[0] == 3.25

print("RUNTIME_MEMORY_CREATE=PASS")
print(f"KV_BYTES={rt.kv_bytes}")
print(f"WORKSPACE_BYTES={rt.workspace_bytes}")
print(f"LOGITS_BYTES={rt.logits_bytes}")

assert destroy(ctypes.byref(rt)) == 0
assert not rt.keys
assert not rt.values
assert not rt.workspace
assert not rt.logits

print("RUNTIME_MEMORY_DESTROY=PASS")
print("RUNTIME_MEMORY=PASS")
