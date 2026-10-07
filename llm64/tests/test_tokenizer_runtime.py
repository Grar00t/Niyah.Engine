import ctypes
import struct
import sys

so = sys.argv[1]
lib = ctypes.CDLL(so)

enc = lib.tokenizer_encode_mapped
enc.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
enc.restype = ctypes.c_ssize_t

dec = lib.tokenizer_decode_one_mapped
dec.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
dec.restype = ctypes.c_ssize_t

decoded_len = lib.tokenizer_decoded_length_mapped
decoded_len.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
decoded_len.restype = ctypes.c_ssize_t

BASE = 258
BOS = 256
EOS = 257

# Ordered BPE:
# 258 = "ab"
# 259 = "abc"
# 260 = "xy"
merges = [
    (ord("a"), ord("b"), 258),
    (258, ord("c"), 259),
    (ord("x"), ord("y"), 260),
]

header = b"NIYAHTOK" + struct.pack(
    "<6I",
    1,                  # version
    0,                  # flags
    BASE,
    BASE + len(merges),
    len(merges),
    0,                  # reserved
)

triples = b"".join(struct.pack("<III", *m) for m in merges)
tok_blob = ctypes.create_string_buffer(header + triples)

src = b"abc ab xy!"
src_buf = ctypes.create_string_buffer(src, len(src))
tokens = (ctypes.c_uint32 * len(src))()

count = enc(
    ctypes.addressof(tok_blob),
    ctypes.addressof(src_buf),
    len(src),
    tokens,
    len(tokens),
)

expected = [259, 32, 258, 32, 260, 33]
got = list(tokens[:count])

assert count == len(expected), (count, expected)
assert got == expected, (got, expected)

decoded = bytearray()

for token in got:
    out = (ctypes.c_uint8 * 64)()
    stack = (ctypes.c_uint32 * 64)()

    n = dec(
        ctypes.addressof(tok_blob),
        token,
        out,
        len(out),
        stack,
        len(stack),
    )

    assert n >= 0, (token, n)
    decoded.extend(bytes(out[:n]))

assert bytes(decoded) == src, (bytes(decoded), src)

tmp = (ctypes.c_uint8 * 8)()
stk = (ctypes.c_uint32 * 8)()

assert dec(ctypes.addressof(tok_blob), BOS, tmp, 8, stk, 8) == 0
assert dec(ctypes.addressof(tok_blob), EOS, tmp, 8, stk, 8) == 0
assert dec(ctypes.addressof(tok_blob), 999999, tmp, 8, stk, 8) == -1

assert enc(
    ctypes.addressof(tok_blob),
    None,
    0,
    None,
    0,
) == 0

# A legal merge tree can expand far beyond 4*vocab bytes. Build a chain that
# doubles "A" eleven times: final token expands to 2048 bytes.
deep_merges = []
left = ord("A")
for index in range(11):
    output = BASE + index
    deep_merges.append((left, left, output))
    left = output

deep_header = b"NIYAHTOK" + struct.pack(
    "<6I", 1, 0, BASE, BASE + len(deep_merges), len(deep_merges), 0
)
deep_triples = b"".join(struct.pack("<III", *m) for m in deep_merges)
deep_blob = ctypes.create_string_buffer(deep_header + deep_triples)
deep_stack = (ctypes.c_uint32 * (BASE + len(deep_merges)))()

n = decoded_len(
    ctypes.addressof(deep_blob),
    left,
    deep_stack,
    len(deep_stack),
)
assert n == 2048, n
assert n > 4 * (BASE + len(deep_merges)), n

deep_out = (ctypes.c_uint8 * n)()
decoded = dec(
    ctypes.addressof(deep_blob),
    left,
    deep_out,
    n,
    deep_stack,
    len(deep_stack),
)
assert decoded == n, decoded
assert bytes(deep_out) == b"A" * n

print("TOKENIZER_RUNTIME=PASS")
print("TOKENIZER_DECODE_LENGTH=PASS")
print("ENCODED_IDS=" + ",".join(map(str, got)))
print("ROUNDTRIP=" + decoded.decode("ascii"))
