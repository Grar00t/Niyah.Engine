import ctypes
import hashlib
import struct
import sys

lib = ctypes.CDLL(sys.argv[1])
identity = lib.tokenizer_identity_sha256_mapped
identity.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
identity.restype = ctypes.c_int

BASE = 258
merges = [
    (ord("a"), ord("b"), 258),
    (258, ord("c"), 259),
    (ord("x"), ord("y"), 260),
]
header = b"NIYAHTOK" + struct.pack("<6I", 1, 0, BASE, BASE + len(merges), len(merges), 0)
triples = b"".join(struct.pack("<III", *m) for m in merges)
tok = ctypes.create_string_buffer(header + triples)
out = (ctypes.c_ubyte * 32)()

rc = identity(ctypes.addressof(tok), out)
assert rc == 0, rc

semantic = b"NIYAH-TOKENIZER-V1" + struct.pack(
    "<III", BASE, BASE + len(merges), len(merges)
) + triples
expected = hashlib.sha256(semantic).digest()
assert bytes(out) == expected, (bytes(out).hex(), expected.hex())

print("TOKENIZER_IDENTITY=PASS")
print("SHA256=" + expected.hex())
