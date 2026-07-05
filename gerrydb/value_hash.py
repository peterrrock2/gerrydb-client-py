"""Order-independent content fingerprints for column values.

Byte-for-byte mirror of gerrydb_meta/value_hash.py on the server; the two
implementations are pinned to each other by identical golden-vector tests.
Changing any byte of the encoding breaks duplicate detection.
"""

import hashlib
import json
import struct

import pandas as pd

from gerrydb.schemas import ColumnType

_TYPE_TAGS = {
    ColumnType.INT: b"i:",
    ColumnType.FLOAT: b"f:",
    ColumnType.BOOL: b"b:",
    ColumnType.STR: b"s:",
    ColumnType.JSON: b"j:",
}


def encode_value(col_type: ColumnType, value) -> bytes:
    """Canonical byte encoding of a value, keyed by the column's type."""
    tag = _TYPE_TAGS[col_type]
    if col_type == ColumnType.INT:
        return tag + str(int(value)).encode()
    if col_type == ColumnType.FLOAT:
        return tag + struct.pack(">d", float(value))
    if col_type == ColumnType.BOOL:
        return tag + (b"1" if value else b"0")
    if col_type == ColumnType.STR:
        return tag + str(value).encode()
    return tag + json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def pair_digest(path: str, col_type: ColumnType, value) -> tuple[int, int]:
    """Digest of one (geo path, value) pair as signed (hi, lo) 64-bit halves."""
    d = hashlib.md5(path.encode() + b"\x00" + encode_value(col_type, value)).digest()
    return (
        int.from_bytes(d[:8], "big", signed=True),
        int.from_bytes(d[8:], "big", signed=True),
    )


def infer_column_type(series: pd.Series) -> ColumnType | None:
    """Maps a pandas dtype to the ColumnType used for hashing.

    Must agree with the matched column's declared server-side type for a
    fingerprint to match; a wrong guess just means no dedup (the safe
    direction). Columns with NaNs are not fingerprinted.
    """
    if series.isna().any():
        return None
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOL
    if pd.api.types.is_integer_dtype(series):
        return ColumnType.INT
    if pd.api.types.is_float_dtype(series):
        return ColumnType.FLOAT
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return ColumnType.STR
    return None


def column_fingerprint(series: pd.Series, col_type: ColumnType) -> tuple[int, int]:
    """Fingerprint of a DataFrame column indexed by geography path."""
    hi = lo = 0
    for path, value in series.items():
        h, l = pair_digest(str(path), col_type, value)
        hi ^= h
        lo ^= l
    return hi, lo
