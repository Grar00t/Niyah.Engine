#!/usr/bin/env python3
"""Gate 6: Test all 8 unsupported GGUF types fail loudly."""
import sys, os, tempfile, struct
sys.path.insert(0, 'tools')
sys.path.insert(0, 'tools/tests')
import gguf_fixture as fx
import convert_gguf_to_niyah as conv

tmp = tempfile.mkdtemp()

def write_good_model(path):
    writer, order = fx.build_model(types=fx.KQUANT_TYPES)
    writer.write(path)
    return order

def find_and_patch_tensor_type(data, old_type, new_type):
    magic = data[:4]
    assert magic == b'GGUF', repr(magic)
    pos = 4
    version = struct.unpack_from('<I', data, pos)[0]; pos += 4
    tensor_count = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    kv_count     = struct.unpack_from('<Q', data, pos)[0]; pos += 8

    for _ in range(kv_count):
        klen = struct.unpack_from('<Q', data, pos)[0]; pos += 8 + klen
        vtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        if vtype == 8:
            slen = struct.unpack_from('<Q', data, pos)[0]; pos += 8 + slen
        elif vtype in (0,1,7): pos += 1
        elif vtype in (2,3): pos += 2
        elif vtype in (4,5,6): pos += 4
        elif vtype in (10,11,12): pos += 8
        elif vtype == 9:
            etype = struct.unpack_from('<I', data, pos)[0]; pos += 4
            alen  = struct.unpack_from('<Q', data, pos)[0]; pos += 8
            for _ in range(alen):
                if etype == 8:
                    slen = struct.unpack_from('<Q', data, pos)[0]; pos += 8 + slen
                elif etype in (10,11,12): pos += 8
                else: pos += 4
        else:
            pos += 4

    patched = bytearray(data)
    patched_count = 0
    for ti in range(tensor_count):
        nlen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        name = data[pos:pos+nlen]; pos += nlen
        n_dims = struct.unpack_from('<I', data, pos)[0]; pos += 4
        pos += n_dims * 8
        type_pos = pos
        ttype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        pos += 8
        if ttype == old_type and patched_count == 0:
            struct.pack_into('<I', patched, type_pos, new_type)
            patched_count += 1
            nm = name.decode()
            print("  patched tensor '" + nm + "' type " + str(old_type) + "->" + str(new_type))

    return bytes(patched), patched_count

base_path = os.path.join(tmp, 'base.gguf')
write_good_model(base_path)
with open(base_path, 'rb') as f:
    base_data = f.read()
print("Base GGUF size: " + str(len(base_data)) + " bytes")

# Q5_K via fixture
print("\n--- Q5_K (fixture-based) ---")
types = dict(fx.KQUANT_TYPES)
types['token_embd'] = fx.GGML_Q5_K
writer, _ = fx.build_model(types=types)
gpath = os.path.join(tmp, 'Q5_K.gguf')
bpath = os.path.join(tmp, 'Q5_K.bin')
writer.write(gpath)
msg = None
try:
    conv.convert(gpath, bpath, None)
except RuntimeError as e:
    msg = str(e)
rc = 0 if msg is None else 1
s = str(msg)[:120] if msg else "NONE"
exists = os.path.exists(bpath)
print("Q5_K(type=13): rc=" + str(rc))
print("  error: " + repr(s))
print("  output_left_behind: " + str(exists))

# Binary-patch for remaining types
BASE_TYPE = 12  # Q4_K
print()
for tname, new_type in [
    ('Q2_K', 10), ('Q3_K', 11), ('Q8_K', 15),
    ('Q5_0', 6),  ('Q5_1', 7),  ('Q8_0', 8), ('Q8_1', 9)
]:
    patched, n = find_and_patch_tensor_type(base_data, BASE_TYPE, new_type)
    if n == 0:
        print(tname + ": no tensor with type " + str(BASE_TYPE) + " found to patch")
        continue
    gpath = os.path.join(tmp, tname + '.gguf')
    bpath = os.path.join(tmp, tname + '.bin')
    with open(gpath, 'wb') as f:
        f.write(patched)
    msg = None
    try:
        conv.convert(gpath, bpath, None)
    except RuntimeError as e:
        msg = str(e)
    rc = 0 if msg is None else 1
    s = str(msg)[:120] if msg else "NONE"
    exists = os.path.exists(bpath)
    print(tname + "(type=" + str(new_type) + "): rc=" + str(rc))
    print("  error: " + repr(s))
    print("  output_left_behind: " + str(exists))
    print()
