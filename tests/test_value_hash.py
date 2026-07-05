"""Golden vectors pinning the client hash encoding to the server's."""

import hashlib

from gerrydb.schemas import ColumnType
from gerrydb.value_hash import column_fingerprint, encode_value, pair_digest

import pandas as pd


def test_encoding_golden_vectors():
    # Identical to gerrydb-meta's test_value_hash vectors: the two
    # implementations must produce the same bytes forever.
    assert encode_value(ColumnType.INT, 5) == b"i:5"
    assert encode_value(ColumnType.INT, -12) == b"i:-12"
    assert encode_value(ColumnType.FLOAT, 1.0) == b"f:" + bytes.fromhex("3ff0000000000000")
    assert encode_value(ColumnType.BOOL, True) == b"b:1"
    assert encode_value(ColumnType.STR, "x") == b"s:x"
    assert encode_value(ColumnType.FLOAT, 1) == encode_value(ColumnType.FLOAT, 1.0)


def test_pair_digest_golden_vector():
    d = hashlib.md5(b"geo:1\x00i:5").digest()
    expect = (
        int.from_bytes(d[:8], "big", signed=True),
        int.from_bytes(d[8:], "big", signed=True),
    )
    assert pair_digest("geo:1", ColumnType.INT, 5) == expect


def test_column_fingerprint_order_independent():
    s1 = pd.Series([1, 2], index=["a", "b"])
    s2 = pd.Series([2, 1], index=["b", "a"])
    assert column_fingerprint(s1, ColumnType.INT) == column_fingerprint(s2, ColumnType.INT)
