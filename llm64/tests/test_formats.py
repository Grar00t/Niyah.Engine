#!/usr/bin/env python3
"""Black-box format tests for the syscall-only llm64 loader.

The runtime under test remains pure NASM/ld. Python is used only by the test
harness to build deterministic binary fixtures and invoke the ELF.
"""

from __future__ import annotations

import binascii
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

CKPT_MAGIC = b"NIYAHCKP"
TOK_MAGIC = b"NIYAHTOK"
CRC32_KIND = 1
REQUIRED = 1

VOCAB = 258
CONTEXT = 8
DIM = 4
LAYERS = 1
HEADS = 2
KV_HEADS = 1
FFN = 8
RMS_EPS = 1.0e-5
TIE = 1
SEGMENTS = 0


def crc_footer(data: bytes) -> bytes:
    return struct.pack("<II", CRC32_KIND, binascii.crc32(data) & 0xFFFFFFFF)


def weight_count() -> int:
    head_dim = DIM // HEADS
    token_embedding = VOCAB * DIM
    segment_embedding = SEGMENTS * DIM
    q_count = DIM * DIM
    kv_count = head_dim * KV_HEADS * DIM
    ffn_matrix = FFN * DIM
    layer_stride = 2 * DIM + 2 * q_count + 2 * kv_count + 3 * ffn_matrix
    total = token_embedding + segment_embedding + LAYERS * layer_stride + DIM
    if not TIE:
        total += token_embedding
    return total


def section(section_id: int, payload: bytes, flags: int = REQUIRED) -> bytes:
    return struct.pack("<IIQ", section_id, flags, len(payload)) + payload


def checkpoint(sections: list[bytes]) -> bytes:
    count = weight_count()
    header = struct.pack(
        "<8s" + "I" * 11 + "fIIQ",
        CKPT_MAGIC,
        2,  # version
        0,  # flags
        len(sections),
        0,  # reserved
        VOCAB,
        CONTEXT,
        DIM,
        LAYERS,
        HEADS,
        KV_HEADS,
        FFN,
        RMS_EPS,
        TIE,
        SEGMENTS,
        count,
    )
    assert len(header) == 72
    body = header + b"".join(sections)
    return body + crc_footer(body)


def tokenizer() -> bytes:
    header = struct.pack(
        "<8sIIIIII",
        TOK_MAGIC,
        1,  # version
        0,  # flags
        258,
        258,
        0,  # merge_count
        0,  # reserved
    )
    assert len(header) == 32
    return header + crc_footer(header)


def base_sections(*, identity_id: int = 5, identity_flags: int = REQUIRED) -> list[bytes]:
    tensor = b"\x00" * (weight_count() * 4)
    meta = struct.pack("<ffffffQ", 1.0e-3, 0.9, 0.999, 1.0e-8, 0.0, 1.0, 1)
    identity = bytes(range(32))
    return [
        section(1, tensor),
        section(2, tensor),
        section(3, tensor),
        section(4, meta),
        section(identity_id, identity, identity_flags),
    ]


def run(binary: Path, ckpt: Path, tok: Path, expected: int) -> None:
    proc = subprocess.run(
        [str(binary), str(ckpt), str(tok)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != expected:
        raise AssertionError(
            f"{ckpt.name}: expected rc={expected}, got rc={proc.returncode}; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )


def main() -> int:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "./niyah-asm").resolve()
    if not binary.is_file():
        raise SystemExit(f"missing loader: {binary}")

    with tempfile.TemporaryDirectory(prefix="niyah-llm64-") as tmp:
        root = Path(tmp)
        tok = root / "tok.bin"
        tok.write_bytes(tokenizer())

        valid = root / "valid.ckpt"
        valid.write_bytes(checkpoint(base_sections()))
        run(binary, valid, tok, 0)

        valid_schedule = root / "valid-schedule.ckpt"
        valid_schedule.write_bytes(checkpoint(base_sections() + [section(6, struct.pack("<Q", 1))]))
        run(binary, valid_schedule, tok, 0)

        unknown_required = root / "unknown-required.ckpt"
        unknown_required.write_bytes(checkpoint(base_sections(identity_id=99)))
        run(binary, unknown_required, tok, 65)

        missing_required_identity = root / "missing-identity.ckpt"
        missing_required_identity.write_bytes(checkpoint(base_sections(identity_id=99, identity_flags=0)))
        run(binary, missing_required_identity, tok, 65)

        nonrequired_known = root / "identity-not-required.ckpt"
        nonrequired_known.write_bytes(checkpoint(base_sections(identity_flags=0)))
        run(binary, nonrequired_known, tok, 65)

        zero_schedule = root / "zero-schedule.ckpt"
        zero_schedule.write_bytes(checkpoint(base_sections() + [section(6, struct.pack("<Q", 0))]))
        run(binary, zero_schedule, tok, 65)

        bad_weight = bytearray(b"\x00" * (weight_count() * 4))
        bad_weight[:4] = struct.pack("<f", math.inf)
        sections = base_sections()
        sections[0] = section(1, bytes(bad_weight))
        nonfinite = root / "nonfinite-model.ckpt"
        nonfinite.write_bytes(checkpoint(sections))
        run(binary, nonfinite, tok, 65)

    print("llm64 format regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
