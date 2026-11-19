#!/usr/bin/env python3
"""Prepare a GTE-Qwen2 GGUF for Ollama embeddings.

Ollama only treats a GGUF as an embedding model when the metadata
contains ``<arch>.pooling_type``.  Several public GTE-Qwen2 GGUFs omit
that key, so Ollama exposes the model as completion-only and
``/api/embed`` fails with "this model does not support embeddings".
This script adds the key (default: 3 = last-token pooling, which is what
GTE-Qwen2 was trained with).

The ``rjmalagon/gte-qwen2-1.5b-instruct-embed-f16`` upload has a second
defect: its data section starts with two zero bytes, so every tensor is
read two bytes early.  F32 tensors (RMSNorm gains, biases) then decode
as denormals and llama.cpp returns all-zero embeddings.  The script
detects that exact layout (file is two bytes longer than the last tensor
*and* F32 samples are denormal at the declared offset but normal two
bytes later) and removes the two-byte gap.  The correctly converted
``second-state/gte-Qwen2-1.5B-instruct-GGUF`` upload does not need this
repair.

Usage::

    # Preferred: a correctly converted file (only pooling is missing).
    curl -L -o gte-qwen2-f16.gguf \\
      https://huggingface.co/second-state/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf

    python scripts/prepare_gte_qwen2_gguf.py \\
        gte-qwen2-f16.gguf gte-qwen2-pooled-f16.gguf

    # Modelfile:
    #   FROM /absolute/path/gte-qwen2-pooled-f16.gguf
    #   PARAMETER num_thread 8
    ollama create gte-qwen2:8t -f Modelfile

    curl -s http://localhost:11434/api/embed \\
      -d '{"model":"gte-qwen2:8t","input":"dimension test"}' \\
      | python3 -c 'import json, math, sys; \\
e = json.load(sys.stdin)["embeddings"][0]; \\
print(len(e), math.sqrt(sum(x * x for x in e)))'
    # -> 1536 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import sys
import tempfile
from pathlib import Path

# GGUF metadata value sizes (fixed-width types).
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# GGML tensor type -> (elements per block, bytes per block).
TENSOR_SIZES = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
}


def skip_value(f, vtype: int) -> None:
    if vtype in FIXED:
        f.seek(FIXED[vtype], 1)
        return
    if vtype == 8:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
        return
    if vtype == 9:
        (atype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, atype)
        return
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_string(f) -> bytes:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def tensor_nbytes(dims: list[int], ttype: int) -> int:
    if ttype not in TENSOR_SIZES:
        raise ValueError(f"unknown GGML tensor type {ttype}")
    block_elems, block_bytes = TENSOR_SIZES[ttype]
    count = 1
    for d in dims:
        count *= d
    if count % block_elems:
        raise ValueError(f"tensor size {count} not divisible by block {block_elems}")
    return count // block_elems * block_bytes


def f32_words(f, offset: int, count: int) -> tuple[float, ...]:
    f.seek(offset)
    raw = f.read(count * 4)
    if len(raw) != count * 4:
        return ()
    return struct.unpack(f"<{count}f", raw)


def detect_data_shift(path: Path, data_start: int, last_end: int, file_size: int) -> int:
    """Detect the ``rjmalagon`` 2-byte leading-gap defect.

    The bad upload is exactly two bytes longer than its declared tensors
    and its F32 data decodes as denormals at the declared offset while
    the same words two bytes later decode as ordinary floats.
    """
    if file_size != last_end + 2:
        return 0

    with path.open("rb") as f:
        checked = 0
        for offset, count, ttype in _f32_tensors(path):
            if count < 4:
                continue
            early = f32_words(f, data_start + offset, min(count, 8))
            late = f32_words(f, data_start + offset + 2, min(count, 8))
            if not early or not late:
                return 0
            if not all(abs(v) < 1e-30 for v in early):
                return 0
            if not all(1e-12 < abs(v) < 1e6 for v in late):
                return 0
            checked += 1
        return 2 if checked else 0


def _f32_tensors(path: Path) -> list[tuple[int, int, int]]:
    """Return ``(offset, element_count, type)`` for every F32 tensor."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        struct.unpack("<I", f.read(4))  # version
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            key = read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == b"general.alignment" and vtype == 4:
                f.seek(4, 1)
            else:
                skip_value(f, vtype)
        result = []
        for _ in range(n_tensors):
            read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            if ttype == 0:
                count = 1
                for d in dims:
                    count *= d
                result.append((offset, count, ttype))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", type=Path, help="source GGUF")
    parser.add_argument("dst", type=Path, help="output GGUF")
    parser.add_argument(
        "--pooling-type",
        type=int,
        default=3,
        help="llama.cpp pooling type to add when missing (3 = last token)",
    )
    parser.add_argument(
        "--no-pooling",
        action="store_true",
        help="do not add the pooling metadata key",
    )
    args = parser.parse_args()

    with args.src.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        kv_start = f.tell()

        alignment = 32
        architecture = None
        existing_pooling = None
        for _ in range(n_kv):
            key = read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == b"general.alignment" and vtype == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            elif key == b"general.architecture" and vtype == 8:
                (slen,) = struct.unpack("<Q", f.read(8))
                architecture = f.read(slen).decode()
            elif key.endswith(b".pooling_type") and vtype == 4:
                (existing_pooling,) = struct.unpack("<I", f.read(4))
            else:
                skip_value(f, vtype)
        kv_end = f.tell()

        tensors = []
        for _ in range(n_tensors):
            name = read_string(f).decode()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, ttype, offset))
        meta_end = f.tell()

    if architecture is None:
        raise SystemExit("general.architecture missing")

    data_start = (meta_end + alignment - 1) // alignment * alignment
    file_size = args.src.stat().st_size
    last_end = data_start + max(
        offset + tensor_nbytes(dims, ttype)
        for _name, dims, ttype, offset in tensors
    )
    if file_size < last_end:
        raise SystemExit(
            f"truncated file: size {file_size} < last tensor end {last_end}"
        )

    data_shift = detect_data_shift(args.src, data_start, last_end, file_size)
    if data_shift:
        print(f"detected {data_shift}-byte leading gap in tensor data; repairing")

    add_pooling = not args.no_pooling and existing_pooling is None
    new_key = f"{architecture}.pooling_type".encode()

    with args.src.open("rb") as src, tempfile.NamedTemporaryFile(
        dir=args.dst.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(
            b"GGUF"
            + struct.pack("<I", version)
            + struct.pack("<Q", n_tensors)
            + struct.pack("<Q", n_kv + (1 if add_pooling else 0))
        )

        src.seek(kv_start)
        remaining = kv_end - kv_start
        while remaining:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                raise SystemExit("unexpected EOF in metadata")
            tmp.write(chunk)
            remaining -= len(chunk)

        if add_pooling:
            tmp.write(
                struct.pack("<Q", len(new_key))
                + new_key
                + struct.pack("<I", 4)
                + struct.pack("<I", args.pooling_type)
            )

        # Tensor info entries follow the key/value block and must stay
        # between the metadata and the aligned data section.
        src.seek(kv_end)
        remaining = meta_end - kv_end
        while remaining:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                raise SystemExit("unexpected EOF in tensor info")
            tmp.write(chunk)
            remaining -= len(chunk)

        new_meta_end = tmp.tell()
        new_data_start = (new_meta_end + alignment - 1) // alignment * alignment
        tmp.write(b"\0" * (new_data_start - new_meta_end))

        src.seek(data_start + data_shift)
        shutil.copyfileobj(src, tmp, length=1 << 20)
        tmp.close()

    digest = hashlib.sha256()
    with tmp_path.open("rb") as out:
        while chunk := out.read(1 << 20):
            digest.update(chunk)
    tmp_path.replace(args.dst)

    pooling_text = (
        f"{args.pooling_type} (added)"
        if add_pooling
        else (f"{existing_pooling} (kept)" if existing_pooling is not None else "none")
    )
    print(f"wrote {args.dst}")
    print(
        f"architecture={architecture} pooling_type={pooling_text} "
        f"data_shift={data_shift} tensors={n_tensors}"
    )
    print(f"sha256 {digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
