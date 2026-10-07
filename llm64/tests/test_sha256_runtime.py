import ctypes
import hashlib
import os
import sys

lib = ctypes.CDLL(sys.argv[1])
sha = lib.sha256_bytes
sha.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
sha.restype = ctypes.c_int

vectors = [
    b"",
    b"abc",
    b"a" * 55,
    b"a" * 56,
    b"a" * 64,
    bytes(range(256)) * 3,
]

for payload in vectors:
    buf = ctypes.create_string_buffer(payload if payload else b"\0", max(1, len(payload)))
    out = (ctypes.c_ubyte * 32)()
    rc = sha(ctypes.addressof(buf), len(payload), out)
    assert rc == 0, (len(payload), rc)
    got = bytes(out)
    expected = hashlib.sha256(payload).digest()
    assert got == expected, (len(payload), got.hex(), expected.hex())

print("SHA256_RUNTIME=PASS")
