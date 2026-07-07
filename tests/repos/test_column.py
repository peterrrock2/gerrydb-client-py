"""Integration/VCR tests for columns."""

import asyncio

import httpx
import pytest
from httpx import HTTPError
from shapely import box

from gerrydb.client import GerryDB
from gerrydb.schemas import ColumnKind, ColumnType


@pytest.fixture
def column(pop_column_meta):
    """Column metadata."""
    return pop_column_meta


@pytest.mark.vcr
def test_column_repo_create_get(client_ns, column):
    with client_ns.context(notes="adding a column") as ctx:
        col = ctx.columns.create(**column)

    assert col.kind == ColumnKind.COUNT
    assert col.type == ColumnType.INT
    assert client_ns.columns["total_pop"] == col
    assert client_ns.columns["totpop"] == col
    assert client_ns.columns[(f"{client_ns.namespace}", "total_pop")] == col


@pytest.mark.vcr
def test_column_repo_create_all(client_ns, column):
    with client_ns.context(notes="adding a column") as ctx:
        ctx.columns.create(**column)

    assert "total_pop" in [col.path for col in client_ns.columns.all()]


@pytest.mark.vcr
def test_column_repo_create_update_get(client_ns, column):
    with client_ns.context(notes="adding and then updating a column") as ctx:
        ctx.columns.create(**column)
        updated_col = ctx.columns.update("total_pop", aliases=["population"])

    assert client_ns.columns["population"] == updated_col


def test_column_repo_set_values(client_ns, column):
    n = 10000
    with client_ns.context(notes="adding a column, geographies, and values") as ctx:
        col = ctx.columns.create(**column)
        with ctx.geo.bulk() as geo_ctx:
            geo_ctx.create({f"{idx:010d}": box(0, 0, 1, 1) for idx in range(n)})
        ctx.columns.set_values(path=col.path, values={f"{idx:010d}": idx for idx in range(n)})


def test_column_repo_set_values_invalid_path_or_col(client_ns):
    with pytest.raises(ValueError, match="Either `path` or `col` must be provided."):
        client_ns.context().columns.set_values(
            path=None,
            namespace="test",
            col=None,
            values={},
        )


def test_column_repo_async_set_values_invalid_values(client_ns):
    with pytest.raises(
        ValueError,
        match="Either `path` or `col` must be provided.",
    ):
        asyncio.run(
            client_ns.context().columns.async_set_values(
                path=None,
                namespace="test",
                col=None,
                values={},
            )
        )


# turn off both “unused mock” and “unexpected request” errors
pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    assert_all_requests_were_expected=False,
)


def test_async_set_values_ephemeral_client(httpx_mock):
    # 1) stub the POST /meta/ for WriteContext.__enter__
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "irrelevant",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )

    # 2) stub the PUT that async_set_values will fire
    #    URL is: {base_url}/{namespace}/{path}
    httpx_mock.add_response(
        method="PUT",
        url="http://localhost:8000/api/v1/columns/test_ns/foo",
        status_code=204,
    )

    # 3) spin up a real client + context
    db = GerryDB(
        host="localhost:8000",
        key="dummy-key",
        namespace="test_ns",
        cache_max_size_gb=0.001,
    )

    with db.context(notes="whatever") as ctx:
        asyncio.run(
            ctx.columns.async_set_values(
                path="foo",
                namespace="test_ns",
                values={"aa": 1, "bb": 2},
                client=None,  # explicit, but same as omitting
            )
        )


def test_async_set_values_ephemeral_client_is_closed(httpx_mock, monkeypatch):
    # 1) stub WriteContext metadata call
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "irrelevant",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )

    # 2) stub the PUT
    httpx_mock.add_response(
        method="PUT",
        url="http://localhost:8000/api/v1/columns/test_ns/foo",
        status_code=204,
    )

    # 3) intercept AsyncClient.aclose
    closed = False

    async def fake_aclose(self):
        nonlocal closed
        closed = True

    monkeypatch.setattr(httpx.AsyncClient, "aclose", fake_aclose)

    # 4) run the code
    db = GerryDB(
        host="localhost:8000",
        key="dummy-key",
        namespace="test_ns",
        cache_max_size_gb=0.001,
    )

    with db.context(notes="whatever") as ctx:
        asyncio.run(
            ctx.columns.async_set_values(
                path="foo",
                namespace="test_ns",
                values={"aa": 1, "bb": 2},
            )
        )

    # 5) confirm that our fake aclose ran
    assert closed, "Expected the ephemeral AsyncClient to be closed"


def test_async_set_values_bad_response(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "irrelevant",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )

    httpx_mock.add_response(
        method="PUT",
        url="http://localhost:8000/api/v1/columns/test_ns/foo",
        status_code=404,
    )

    db = GerryDB(
        host="localhost:8000",
        key="dummy-key",
        namespace="test_ns",
        cache_max_size_gb=0.001,
    )

    with pytest.raises(HTTPError, match="Client error '404 Not Found'"):
        with db.context(notes="whatever") as ctx:
            asyncio.run(
                ctx.columns.async_set_values(
                    path="foo",
                    namespace="test_ns",
                    values={"aa": 1, "bb": 2},
                )
            )


@pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    assert_all_requests_were_expected=False,
)
def test_column_clone_and_reference_listing(httpx_mock):
    """clone() posts a validated reference; all(include_references=True)
    queries the flagged listing."""
    import json as jsonlib

    from gerrydb.client import GerryDB

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "irrelevant",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/column-refs/tgt",
        status_code=201,
        json={"path": "pop", "target_namespace": "src", "target_path": "pop"},
    )

    db = GerryDB(
        host="localhost:8000",
        key="dummy-key",
        namespace="tgt",
        cache_max_size_gb=0.001,
    )
    with db.context(notes="cloning") as ctx:
        out = ctx.columns.clone(
            "pop", from_namespace="src", from_path="pop", validate_paths=True
        )
    assert out["target_namespace"] == "src"
    clone_req = [
        r
        for r in httpx_mock.get_requests()
        if r.url.path.endswith("/column-refs/tgt") and r.method == "POST"
    ][-1]
    sent = jsonlib.loads(clone_req.content)
    assert sent == {
        "path": "pop",
        "target_namespace": "src",
        "target_path": "pop",
        "validate_paths": True,
    }

    col_payload = {
        "canonical_path": "pop",
        "namespace": "tgt",
        "description": "d",
        "kind": "count",
        "type": "int",
        "aliases": [],
        "meta": {
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "n",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    }
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:8000/api/v1/columns/tgt?include_references=true",
        json=[col_payload],
    )
    cols = db.columns.all(include_references=True)
    assert [c.canonical_path for c in cols] == ["pop"]


@pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    assert_all_requests_were_expected=False,
)
def test_column_set_values_allow_local_updates_materializes_first(httpx_mock):
    """set_values(allow_local_updates=True) on a cross-namespace reference calls the
    materialize endpoint before the value PUT; without the flag the write
    goes straight to the server (which refuses refs)."""
    from gerrydb.client import GerryDB

    meta_json = {
        "uuid": "00000000-0000-0000-0000-000000000000",
        "notes": "n",
        "created_at": "2025-04-26T00:00:00Z",
        "created_by": "test-user@example.com",
    }
    httpx_mock.add_response(
        method="POST", url="http://localhost:8000/api/v1/meta/", json=meta_json
    )
    src_col = {
        "canonical_path": "pop",
        "namespace": "src",
        "description": "d",
        "kind": "count",
        "type": "int",
        "aliases": [],
        "meta": meta_json,
    }
    tgt_col = dict(src_col, namespace="tgt")
    httpx_mock.add_response(
        method="GET", url="http://localhost:8000/api/v1/columns/tgt/pop", json=src_col
    )
    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/columns/tgt/pop/materialize",
        json=tgt_col,
    )
    httpx_mock.add_response(
        method="PUT",
        url="http://localhost:8000/api/v1/columns/tgt/pop",
        status_code=204,
    )

    db = GerryDB(
        host="localhost:8000", key="dummy-key", namespace="tgt", cache_max_size_gb=0.001
    )
    with db.context(notes="diverging") as ctx:
        ctx.columns.set_values("pop", values={"g0": 1}, allow_local_updates=True)

    methods = [
        (r.method, r.url.path)
        for r in httpx_mock.get_requests()
        if "/columns/" in r.url.path
    ]
    mat_idx = methods.index(("POST", "/api/v1/columns/tgt/pop/materialize"))
    put_idx = methods.index(("PUT", "/api/v1/columns/tgt/pop"))
    assert mat_idx < put_idx


@pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    assert_all_requests_were_expected=False,
)
def test_column_clone_list(httpx_mock):
    """clone() accepts a list of paths, defaulting source paths to the same
    names, and returns one result per clone."""
    import json as jsonlib

    from gerrydb.client import GerryDB

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "n",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )
    for name in ("pres_16_dem", "pres_16_rep"):
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8000/api/v1/column-refs/tgt",
            status_code=201,
            json={"path": name, "target_namespace": "src", "target_path": name},
        )

    db = GerryDB(
        host="localhost:8000", key="dummy-key", namespace="tgt", cache_max_size_gb=0.001
    )
    with db.context(notes="bulk clone") as ctx:
        out = ctx.columns.clone(
            ["pres_16_dem", "pres_16_rep"], from_namespace="src"
        )
    assert [o["path"] for o in out] == ["pres_16_dem", "pres_16_rep"]

    bodies = [
        jsonlib.loads(r.content)
        for r in httpx_mock.get_requests()
        if r.url.path.endswith("/column-refs/tgt")
    ]
    assert [(b["path"], b["target_path"]) for b in bodies] == [
        ("pres_16_dem", "pres_16_dem"),
        ("pres_16_rep", "pres_16_rep"),
    ]

    httpx_mock.add_response(
        method="POST",
        url="http://localhost:8000/api/v1/meta/",
        json={
            "uuid": "00000000-0000-0000-0000-000000000000",
            "notes": "n",
            "created_at": "2025-04-26T00:00:00Z",
            "created_by": "test-user@example.com",
        },
    )
    with pytest.raises(Exception, match="clone"):
        with db.context(notes="mismatched") as ctx:
            ctx.columns.clone(["a", "b"], from_namespace="src", from_path=["a"])
