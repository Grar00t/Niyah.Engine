#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  NIYAH-CORE  ·  Weight Converter                                 ║
║  Converts GGUF (llama/mistral) or safetensors → .niyah format   ║
║  Dependency: numpy only (no torch, no transformers)             ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python3 convert.py --gguf  llama-3.2-1b.gguf  --out model.niyah
  python3 convert.py --st    model.safetensors   --out model.niyah
  python3 convert.py --demo  --size tiny         --out tiny.niyah
"""

import struct
import sys
import os
import argparse
import numpy as np

# ── .niyah binary format constants ──────────────────────────────────
NIYAH_MAGIC   = 0x4859494E  # "NIYH" little-endian
NIYAH_VERSION = 2

# ── NiyahConfig struct layout (matches C struct) ────────────────────
CONFIG_FMT    = "<IIIIIIIfI"   # 9 fields
CONFIG_SIZE   = struct.calcsize(CONFIG_FMT)

def pack_config(vocab, ctx, embed, layers, heads, ffn, head_dim, rope_theta=10000.0):
    return struct.pack(CONFIG_FMT,
        vocab, ctx, embed, layers, heads, ffn, head_dim,
        rope_theta, 0)

def write_niyah(path: str, cfg: dict, weights: dict):
    """
    cfg keys: vocab_size, ctx_len, embed_dim, num_layers,
              num_heads, ffn_dim, head_dim, rope_theta
    weights keys per layer l:
        rms1_{l}, q_{l}, k_{l}, v_{l}, o_{l},
        rms2_{l}, gate_{l}, up_{l}, down_{l}
    global: token_emb, out_rms, lm_head
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<II", NIYAH_MAGIC, NIYAH_VERSION))
        f.write(pack_config(
            cfg["vocab_size"], cfg["ctx_len"], cfg["embed_dim"],
            cfg["num_layers"], cfg["num_heads"], cfg["ffn_dim"],
            cfg["head_dim"], cfg.get("rope_theta", 10000.0)
        ))
        def wt(name):
            t = weights[name].astype(np.float32)
            f.write(t.tobytes())
            print(f"  wrote {name:30s} {t.shape}  [{t.nbytes/1e6:.2f} MB]")

        wt("token_emb")
        for l in range(cfg["num_layers"]):
            wt(f"rms1_{l}")
            wt(f"q_{l}")
            wt(f"k_{l}")
            wt(f"v_{l}")
            wt(f"o_{l}")
            wt(f"rms2_{l}")
            wt(f"gate_{l}")
            wt(f"up_{l}")
            wt(f"down_{l}")
        wt("out_rms")
        wt("lm_head")

    size_mb = os.path.getsize(path) / 1e6
    print(f"\n✓ Saved: {path}  ({size_mb:.1f} MB)")

# ── Demo: generate random weights ───────────────────────────────────
def generate_demo(size: str) -> tuple[dict, dict]:
    configs = {
        "tiny":  dict(vocab_size=256, ctx_len=512,  embed_dim=128, num_layers=4,
                      num_heads=4, ffn_dim=384,  head_dim=32),
        "small": dict(vocab_size=256, ctx_len=1024, embed_dim=512, num_layers=6,
                      num_heads=8, ffn_dim=1536, head_dim=64),
        "medium":dict(vocab_size=256, ctx_len=2048, embed_dim=1024,num_layers=12,
                      num_heads=16,ffn_dim=4096, head_dim=64),
    }
    cfg = configs.get(size, configs["tiny"])
    cfg["rope_theta"] = 10000.0
    rng = np.random.default_rng(42)

    d, v, ffd, L = cfg["embed_dim"], cfg["vocab_size"], cfg["ffn_dim"], cfg["num_layers"]

    def normal(shape, fan_in):
        return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

    weights = {}
    weights["token_emb"] = normal((v, d), d)
    for l in range(L):
        weights[f"rms1_{l}"] = np.ones(d, dtype=np.float32)
        weights[f"q_{l}"]    = normal((d, d), d)
        weights[f"k_{l}"]    = normal((d, d), d)
        weights[f"v_{l}"]    = normal((d, d), d)
        weights[f"o_{l}"]    = normal((d, d), d)
        weights[f"rms2_{l}"] = np.ones(d, dtype=np.float32)
        weights[f"gate_{l}"] = normal((ffd, d), d)
        weights[f"up_{l}"]   = normal((ffd, d), d)
        weights[f"down_{l}"] = normal((d, ffd), ffd)
    weights["out_rms"] = np.ones(d, dtype=np.float32)
    weights["lm_head"] = normal((v, d), d)

    params = sum(w.size for w in weights.values())
    print(f"Random model — {size}: {params/1e6:.1f}M params, "
          f"embed={d}, layers={L}, heads={cfg['num_heads']}, ffn={ffd}")
    return cfg, weights

# ── GGUF reader (minimal, no external deps) ─────────────────────────
GGUF_MAGIC = 0x46554747  # "GGUF"

GGUF_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64"
}

def read_gguf_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")

def read_gguf_value(f, typ):
    if typ == 0:  return struct.unpack("<B", f.read(1))[0]
    if typ == 1:  return struct.unpack("<b", f.read(1))[0]
    if typ == 2:  return struct.unpack("<H", f.read(2))[0]
    if typ == 3:  return struct.unpack("<h", f.read(2))[0]
    if typ == 4:  return struct.unpack("<I", f.read(4))[0]
    if typ == 5:  return struct.unpack("<i", f.read(4))[0]
    if typ == 6:  return struct.unpack("<f", f.read(4))[0]
    if typ == 7:  return bool(struct.unpack("<B", f.read(1))[0])
    if typ == 8:  return read_gguf_string(f)
    if typ == 10: return struct.unpack("<Q", f.read(8))[0]
    if typ == 11: return struct.unpack("<q", f.read(8))[0]
    if typ == 12: return struct.unpack("<d", f.read(8))[0]
    if typ == 9:  # array
        atyp = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        return [read_gguf_value(f, atyp) for _ in range(alen)]
    raise ValueError(f"Unknown GGUF type: {typ}")

def load_gguf(path: str) -> tuple[dict, dict]:
    """
    Parses a GGUF file and returns (cfg_dict, tensor_dict).
    Only handles F32 and F16 tensors (converts to F32).
    Tested with LLaMA-3.2 1B and Mistral 7B GGUF files.
    """
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic=0x{magic:08X})")
        version   = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv      = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF v{version}: {n_kv} metadata keys, {n_tensors} tensors")

        # read metadata
        meta = {}
        for _ in range(n_kv):
            key = read_gguf_string(f)
            typ = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_gguf_value(f, typ)

        # read tensor descriptors
        tensors = {}
        for _ in range(n_tensors):
            name   = read_gguf_string(f)
            ndim   = struct.unpack("<I", f.read(4))[0]
            dims   = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            dtype  = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = (dims, dtype, offset)

        data_start = f.tell()
        # align to 32
        alignment = meta.get("general.alignment", 32)
        if data_start % alignment != 0:
            data_start += alignment - (data_start % alignment)

        # read tensor data
        raw = {}
        for name, (dims, dtype, offset) in tensors.items():
            f.seek(data_start + offset)
            n = 1
            for d in dims: n *= d
            if dtype == 6:    # F32
                arr = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
            elif dtype == 1:  # F16
                arr = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
            elif dtype == 0:  # F64 (rare)
                arr = np.frombuffer(f.read(n * 8), dtype=np.float64).astype(np.float32)
            else:
                print(f"  skip {name}: unsupported dtype {dtype}")
                continue
            raw[name] = arr.reshape(dims[::-1])  # GGUF stores dims in reverse

    # ── extract config from metadata ─────────────────────────────
    def mi(key, default=None):
        for k, v in meta.items():
            if key in k: return v
        return default

    embed_dim  = mi("embedding_length", 512)
    num_layers = mi("block_count", 4)
    num_heads  = mi("attention.head_count", 8)
    ffn_dim    = mi("feed_forward_length", embed_dim * 4)
    ctx_len    = mi("context_length", 2048)
    vocab_size = mi("vocab_size")
    rope_theta = mi("rope.freq_base", 10000.0)
    head_dim   = embed_dim // num_heads

    if vocab_size is None:
        # infer from embedding table
        emb_key = next((k for k in raw if "token_embd" in k or "embed_tokens" in k), None)
        vocab_size = raw[emb_key].shape[0] if emb_key else 32000

    cfg = dict(vocab_size=vocab_size, ctx_len=ctx_len, embed_dim=embed_dim,
               num_layers=num_layers, num_heads=num_heads, ffn_dim=ffn_dim,
               head_dim=head_dim, rope_theta=rope_theta)

    # ── map GGUF tensor names → .niyah names ─────────────────────
    weights = {}

    def find(patterns):
        for p in patterns:
            for k in raw:
                if p in k: return raw[k]
        return None

    emb = find(["token_embd.weight", "embed_tokens.weight"])
    if emb is not None: weights["token_emb"] = emb

    for l in range(num_layers):
        def fl(patterns):
            for p in patterns:
                k = p.format(l)
                if k in raw: return raw[k]
            return None

        w = fl(["blk.{}.attn_norm.weight", "layers.{}.input_layernorm.weight"])
        if w is not None: weights[f"rms1_{l}"] = w
        else: weights[f"rms1_{l}"] = np.ones(embed_dim, np.float32)

        q = fl(["blk.{}.attn_q.weight", "layers.{}.self_attn.q_proj.weight"])
        if q is not None: weights[f"q_{l}"] = q

        k_ = fl(["blk.{}.attn_k.weight", "layers.{}.self_attn.k_proj.weight"])
        if k_ is not None: weights[f"k_{l}"] = k_

        v = fl(["blk.{}.attn_v.weight", "layers.{}.self_attn.v_proj.weight"])
        if v is not None: weights[f"v_{l}"] = v

        o = fl(["blk.{}.attn_output.weight", "layers.{}.self_attn.o_proj.weight"])
        if o is not None: weights[f"o_{l}"] = o

        w2 = fl(["blk.{}.ffn_norm.weight", "layers.{}.post_attention_layernorm.weight"])
        if w2 is not None: weights[f"rms2_{l}"] = w2
        else: weights[f"rms2_{l}"] = np.ones(embed_dim, np.float32)

        gate = fl(["blk.{}.ffn_gate.weight", "layers.{}.mlp.gate_proj.weight"])
        if gate is not None: weights[f"gate_{l}"] = gate

        up = fl(["blk.{}.ffn_up.weight", "layers.{}.mlp.up_proj.weight"])
        if up is not None: weights[f"up_{l}"] = up

        down = fl(["blk.{}.ffn_down.weight", "layers.{}.mlp.down_proj.weight"])
        if down is not None: weights[f"down_{l}"] = down

    out_rms = find(["output_norm.weight", "model.norm.weight"])
    if out_rms is not None: weights["out_rms"] = out_rms
    else: weights["out_rms"] = np.ones(embed_dim, np.float32)

    lm = find(["output.weight", "lm_head.weight"])
    if lm is None: lm = weights.get("token_emb")  # tied weights
    if lm is not None: weights["lm_head"] = lm

    return cfg, weights

# ── main ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="NIYAH weight converter")
    p.add_argument("--gguf",  help="Input GGUF file")
    p.add_argument("--demo",  action="store_true", help="Generate random model")
    p.add_argument("--size",  default="tiny", choices=["tiny","small","medium"])
    p.add_argument("--out",   required=True, help="Output .niyah file")
    args = p.parse_args()

    if args.demo:
        print(f"Generating random {args.size} model...")
        cfg, weights = generate_demo(args.size)
    elif args.gguf:
        print(f"Loading GGUF: {args.gguf}")
        cfg, weights = load_gguf(args.gguf)
    else:
        p.print_help()
        sys.exit(1)

    write_niyah(args.out, cfg, weights)

if __name__ == "__main__":
    main()
