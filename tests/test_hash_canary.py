"""Cross-stack WKB hash canary.

geo_bin dedup on the server hashes the canonical (grid-snapped) WKB bytes the
client uploads; the constants here are shared with gerrydb-meta's server-side
canary. If this test fails, a shapely/GEOS upgrade changed WKB serialization
or precision reduction, and uploads would stop deduplicating against
previously stored geometries.
"""

import hashlib

import shapely.wkb
from shapely.geometry import Polygon

from gerrydb.repos.geography import canonicalize_geo

EMPTY_POLYGON_MD5 = "75b6f320f5eb33d79cbcd9cf62be5a83"
AWKWARD_POLYGON_MD5 = "f6b505d3b022c9826c36a5c1527d63b0"
CANONICAL_AWKWARD_POLYGON_MD5 = "f05ebab893ed6babfebd1bbbe1693be9"


def awkward_polygon():
    """7-decimal coordinates: off the canonical 1e-6 degree grid."""
    return Polygon(
        [(-71.1234567, 42.7654321), (-70.9876543, 42.7654329), (-70.9876543, 43.0123456)]
    )


def test_wkb_hash_canary():
    assert hashlib.md5(Polygon().wkb).hexdigest() == EMPTY_POLYGON_MD5
    assert hashlib.md5(awkward_polygon().wkb).hexdigest() == AWKWARD_POLYGON_MD5


def test_canonical_wkb_hash_canary():
    """What the serializer actually uploads: grid-snapped geometry bytes."""
    empty_canonical = canonicalize_geo(Polygon())
    assert hashlib.md5(shapely.wkb.dumps(empty_canonical)).hexdigest() == EMPTY_POLYGON_MD5

    awkward_canonical = canonicalize_geo(awkward_polygon())
    assert (
        hashlib.md5(shapely.wkb.dumps(awkward_canonical)).hexdigest()
        == CANONICAL_AWKWARD_POLYGON_MD5
    )
