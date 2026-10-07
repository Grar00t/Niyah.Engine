#!/usr/bin/env python3
from __future__ import annotations

import binascii
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

VOCAB = 258
CONTEXT = 8
DIM = 4
LAYERS = 1
HEADS = 2
KV_HEADS = 1
FFN = 8
EPS = 1.0e-5
TIE = 1
SEGMENTS = 0
REQUIRED = 1


def crc_footer(data: bytes) -> bytes:
    return struct.pack("<II", 1, binascii.crc32(data) & 0xFFFFFFFF)


def section(section_id: int, payload: bytes) -> bytes:
    return struct.pack("<IIQ", section_id, REQUIRED, len(payload)) + payload


def model_layout() -> tuple[int, int, int, int, int, int]:
    head_dim = DIM // HEADS
    kv_dim = head_dim * KV_HEADS
    token_embedding = 0
    layers = VOCAB * DIM
    stride = (
        2 * DIM
        + 2 * DIM * DIM
        + 2 * kv_dim * DIM
        + 3 * FFN * DIM
    )
    final_norm = layers + LAYERS * stride
    total = final_norm + DIM
    return head_dim, kv_dim, token_embedding, layers, final_norm, total


def checkpoint(identity: bytes) -> bytes:
    _, kv_dim, token_embedding, layers, final_norm, total = model_layout()
    weights = [0.0] * total

    # Transition design under tied embeddings with all Transformer matrices zero:
    #   prompt x -> raw byte 'A' -> EOS.
    # x/BOS point along +X; A=(2,1); EOS=(0,6).
    vectors = {
        256: (1.0, 0.0, 0.0, 0.0),
        ord("x"): (1.0, 0.0, 0.0, 0.0),
        ord("A"): (2.0, 1.0, 0.0, 0.0),
        257: (0.0, 6.0, 0.0, 0.0),
    }
    for token, vector in vectors.items():
        base = token_embedding + token * DIM
        for i, value in enumerate(vector):
            weights[base + i] = value

    cursor = layers

    for i in range(DIM):
        weights[cursor + i] = 1.0
    cursor += DIM

    cursor += DIM * DIM
    cursor += kv_dim * DIM
    cursor += kv_dim * DIM
    cursor += DIM * DIM

    for i in range(DIM):
        weights[cursor + i] = 1.0
    cursor += DIM

    cursor += FFN * DIM
    cursor += FFN * DIM
    cursor += DIM * FFN

    assert cursor == final_norm

    for i in range(DIM):
        weights[final_norm + i] = 1.0

    tensor = struct.pack(f"<{len(weights)}f", *weights)
    zero_state = b"\0" * len(tensor)
    meta = struct.pack("<ffffffQ", 1e-3, 0.9, 0.999, 1e-8, 0.0, 1.0, 1)
    assert len(identity) == 32

    sections = [
        section(1, tensor),
        section(2, zero_state),
        section(3, zero_state),
        section(4, meta),
        section(5, identity),
    ]

    header = struct.pack(
        "<8s" + "I" * 11 + "fIIQ",
        b"NIYAHCKP",
        2,
        0,
        len(sections),
        0,
        VOCAB,
        CONTEXT,
        DIM,
        LAYERS,
        HEADS,
        KV_HEADS,
        FFN,
        EPS,
        TIE,
        SEGMENTS,
        total,
    )

    body = header + b"".join(sections)
    return body + crc_footer(body)


def tokenizer() -> bytes:
    header = struct.pack(
        "<8sIIIIII",
        b"NIYAHTOK",
        1,
        0,
        258,
        258,
        0,
        0,
    )
    return header + crc_footer(header)


def tokenizer_identity(tok: bytes) -> bytes:
    base_vocab, vocab, merge_count = struct.unpack_from("<III", tok, 16)
    triples = tok[32:32 + merge_count * 12]
    semantic = (
        b"NIYAH-TOKENIZER-V1"
        + struct.pack("<III", base_vocab, vocab, merge_count)
        + triples
    )
    return hashlib.sha256(semantic).digest()


def main() -> int:
    binary = Path(sys.argv[1]).resolve()

    with tempfile.TemporaryDirectory(prefix="niyah-gen-") as tmp:
        root = Path(tmp)
        ckpt = root / "model.ckpt"
        bad_identity_ckpt = root / "bad-identity.ckpt"
        tok = root / "tok.bin"
        tok_bytes = tokenizer()
        identity = tokenizer_identity(tok_bytes)
        ckpt.write_bytes(checkpoint(identity))
        bad_identity_ckpt.write_bytes(checkpoint(bytes(32)))
        tok.write_bytes(tok_bytes)

        proc = subprocess.run(
            [str(binary), str(ckpt), str(tok), "x"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert proc.returncode == 0, (
            proc.returncode,
            proc.stdout,
            proc.stderr,
        )
        assert proc.stdout == b"A\n", (proc.stdout, proc.stderr)

        limited = subprocess.run(
            [str(binary), str(ckpt), str(tok), "x", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert limited.returncode == 0, (limited.returncode, limited.stderr)
        assert limited.stdout == b"A\n", limited.stdout

        overflow = subprocess.run(
            [str(binary), str(ckpt), str(tok), "x", "7"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert overflow.returncode == 1, (overflow.returncode, overflow.stdout, overflow.stderr)
        assert overflow.stdout == b"", overflow.stdout

        for empty_args in (
            [str(binary), str(ckpt), str(tok), ""],
            [str(binary), str(ckpt), str(tok), "", "1"],
        ):
            empty = subprocess.run(
                empty_args,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            assert empty.returncode == 1, (empty.returncode, empty.stdout, empty.stderr)
            assert empty.stdout == b"", empty.stdout

        identity_mismatch = subprocess.run(
            [str(binary), str(bad_identity_ckpt), str(tok), "x", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert identity_mismatch.returncode == 1, (
            identity_mismatch.returncode,
            identity_mismatch.stdout,
            identity_mismatch.stderr,
        )
        assert identity_mismatch.stdout == b"", identity_mismatch.stdout

    print("NIYAH_ASM_MAX_NEW_TOKENS=PASS")
    print("NIYAH_ASM_CONTEXT_PREFLIGHT=PASS")
    print("NIYAH_ASM_GREEDY_LOOP=PASS")
    print("NIYAH_ASM_EMPTY_PROMPT_REJECT=PASS")
    print("NIYAH_ASM_TOKENIZER_IDENTITY=PASS")
    print("PROMPT=x")
    print("GENERATED_TOKEN_1=65")
    print("GENERATED_BYTES=A")
    print("NEXT_TOKEN=EOS")
    print("GREEDY_EOS_STOP=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
