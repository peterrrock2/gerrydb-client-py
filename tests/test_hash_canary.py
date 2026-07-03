"""Cross-stack WKB hash canary.

geo_bin dedup on the server hashes the WKB bytes the client uploads; the
constants here are shared with gerrydb-meta's server-side canary. If this test
fails, a shapely/GEOS upgrade changed WKB serialization and uploads would stop
deduplicating against previously stored geometries.
"""

import hashlib

from shapely.geometry import Polygon

EMPTY_POLYGON_MD5 = "75b6f320f5eb33d79cbcd9cf62be5a83"
AWKWARD_POLYGON_MD5 = "f6b505d3b022c9826c36a5c1527d63b0"


def test_wkb_hash_canary():
    assert hashlib.md5(Polygon().wkb).hexdigest() == EMPTY_POLYGON_MD5

    awkward = Polygon(
        [(-71.1234567, 42.7654321), (-70.9876543, 42.7654329), (-70.9876543, 43.0123456)]
    )
    assert hashlib.md5(awkward.wkb).hexdigest() == AWKWARD_POLYGON_MD5
