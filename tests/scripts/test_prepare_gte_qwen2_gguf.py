"""Tests for ``scripts/prepare_gte_qwen2_gguf.py``.

The script adds the ``qwen2.pooling_type`` metadata Ollama needs and
repairs the two-byte leading gap in the ``rjmalagon`` GTE upload.  These
tests build tiny synthetic GGUFs so they exercise the metadata/tensor-info
layout without a multi-gigabyte model file.
"""
from __future__ import annotations

import pathlib
import struct
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_gte_qwen2_gguf.py"

_ALIGN = 32
_VALUES = (1.5, -0.25, 0.125, 3.0)


def _gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _build_gguf(path: pathlib.Path, *, shifted: bool) -> None:
    metadata = (
        _gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + _gguf_string("qwen2")
        + _gguf_string("general.alignment")
        + struct.pack("<I", 4)
        + struct.pack("<I", _ALIGN)
    )
    tensor_info = (
        _gguf_string("output_norm.weight")
        + struct.pack("<I", 1)  # n_dims
        + struct.pack("<Q", len(_VALUES))
        + struct.pack("<I", 0)  # GGML_TYPE_F32
        + struct.pack("<Q", 0)  # offset
    )
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)  # n_tensors
        + struct.pack("<Q", 2)  # n_kv
    )
    prefix = header + metadata + tensor_info
    data_start = (len(prefix) + _ALIGN - 1) // _ALIGN * _ALIGN
    payload = struct.pack(f"<{len(_VALUES)}f", *_VALUES)
    if shifted:
        # The rjmalagon layout: two zero bytes before the tensor data,
        # so every tensor is read two bytes early.
        payload = b"\0\0" + payload
    path.write_bytes(prefix + b"\0" * (data_start - len(prefix)) + payload)


def _parse_gguf(path: pathlib.Path) -> tuple[dict[bytes, tuple[int, object]], tuple]:
    """Return ``(metadata, tensor_info)`` for a single-tensor GGUF."""
    f = path.open("rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))  # version
    (n_tensors,) = struct.unpack("<Q", f.read(8))
    (n_kv,) = struct.unpack("<Q", f.read(8))

    metadata: dict[bytes, tuple[int, object]] = {}
    for _ in range(n_kv):
        (klen,) = struct.unpack("<Q", f.read(8))
        key = f.read(klen)
        (vtype,) = struct.unpack("<I", f.read(4))
        if vtype == 8:
            (slen,) = struct.unpack("<Q", f.read(8))
            value: object = f.read(slen)
        elif vtype == 4:
            (value,) = struct.unpack("<I", f.read(4))
        else:
            raise AssertionError(f"unexpected fixture type {vtype} for {key}")
        metadata[key] = (vtype, value)

    tensors = []
    for _ in range(n_tensors):
        (klen,) = struct.unpack("<Q", f.read(8))
        name = f.read(klen)
        (nd,) = struct.unpack("<I", f.read(4))
        dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
        (ttype,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        tensors.append((name, dims, ttype, offset))

    meta_end = f.tell()
    data_start = (meta_end + _ALIGN - 1) // _ALIGN * _ALIGN
    f.seek(data_start + tensors[0][3])
    values = struct.unpack(f"<{len(_VALUES)}f", f.read(len(_VALUES) * 4))
    f.close()
    return metadata, (tensors[0], values)


def _run(src: pathlib.Path, dst: pathlib.Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(src), str(dst)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_adds_pooling_and_keeps_tensor_info(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src.gguf"
    dst = tmp_path / "dst.gguf"
    _build_gguf(src, shifted=False)

    output = _run(src, dst)

    assert "pooling_type=3 (added)" in output
    assert "data_shift=0" in output
    metadata, (tensor, values) = _parse_gguf(dst)
    assert metadata[b"qwen2.pooling_type"] == (4, 3)
    assert metadata[b"general.architecture"] == (8, b"qwen2")
    assert tensor[0] == b"output_norm.weight"
    assert tensor[1] == (len(_VALUES),)
    assert tensor[2] == 0
    assert values == _VALUES


def test_repairs_two_byte_data_gap(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "shifted.gguf"
    dst = tmp_path / "repaired.gguf"
    _build_gguf(src, shifted=True)

    output = _run(src, dst)

    assert "data_shift=2" in output
    assert "detected 2-byte leading gap" in output
    metadata, (tensor, values) = _parse_gguf(dst)
    assert metadata[b"qwen2.pooling_type"] == (4, 3)
    assert tensor[0] == b"output_norm.weight"
    assert values == _VALUES
