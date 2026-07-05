"""Tests for high-level load/import operations."""


def test_load_dataframe__with_geo__ia_counties(client_ns, ia_dataframe, ia_column_meta):
    with client_ns.context(notes="Importing Iowa counties shapefile") as ctx:
        columns = {name: ctx.columns.create(**meta) for name, meta in ia_column_meta.items()}
        layer = ctx.geo_layers.create(
            path="counties",
            description="2020 U.S. Census counties.",
            source_url="https://www.census.gov/",
        )
        locality = ctx.localities.create(canonical_path="load_iowa", name="State of Iowa")
        ctx.load_dataframe(
            df=ia_dataframe,
            columns=columns,
            create_geos=True,
            namespace=client_ns.namespace,
            layer=layer,
            locality=locality,
        )
        # TODO: better testing here. Namely, did everything make it in?


def test_load_dataframe__duplicate_content_becomes_reference(
    request, client, client_ns, ia_dataframe, ia_column_meta
):
    """Re-uploading identical content into a second namespace creates
    references instead of duplicate values; force_duplicate_column uploads."""
    source_ns = client_ns.namespace
    with client_ns.context(notes="source load") as ctx:
        columns = {name: ctx.columns.create(**meta) for name, meta in ia_column_meta.items()}
        layer = ctx.geo_layers.create(path="counties", description="t")
        locality = ctx.localities.create(
            canonical_path=f"{source_ns}-loc", name=f"{source_ns} loc"
        )
        ctx.load_dataframe(
            df=ia_dataframe,
            columns_to_update=columns,
            create_geos=True,
            namespace=source_ns,
            layer=layer,
            locality=locality,
        )

    user_ns = f"{source_ns}-user"
    with client.context(notes="user ns") as ctx:
        ctx.namespaces.create(path=user_ns, description="t", public=True)
    client.namespace = user_ns
    with client.context(notes="user load") as ctx:
        user_layer = ctx.geo_layers.create(
            path="counties", description="t", namespace=user_ns
        )
        report = ctx.load_dataframe(
            df=ia_dataframe,
            columns_to_update=columns,
            create_geos=True,
            namespace=user_ns,
            layer=user_layer,
            locality=locality,
        )
    # Every candidate column matched the source namespace's content.
    assert set(report["referenced"]) == {meta["path"] for meta in ia_column_meta.values()}
    assert all(v == f"{source_ns}/{k}" for k, v in report["referenced"].items())
    assert report["uploaded"] == []

    # The references resolve as columns in the user namespace.
    with client.context(notes="check refs") as ctx:
        for meta in ia_column_meta.values():
            col = ctx.columns.get(meta["path"], namespace=user_ns)
            assert col is not None

    # Forcing duplication re-uploads real values under a fresh namespace.
    forced_ns = f"{source_ns}-forced"
    with client.context(notes="forced ns") as ctx:
        ctx.namespaces.create(path=forced_ns, description="t", public=True)
    client.namespace = forced_ns
    with client.context(notes="forced load") as ctx:
        forced_cols = {
            name: ctx.columns.create(namespace=forced_ns, **meta)
            for name, meta in ia_column_meta.items()
        }
        forced_layer = ctx.geo_layers.create(
            path="counties", description="t", namespace=forced_ns
        )
        report = ctx.load_dataframe(
            df=ia_dataframe,
            columns_to_update=forced_cols,
            create_geos=True,
            force_duplicate_column=True,
            namespace=forced_ns,
            layer=forced_layer,
            locality=locality,
        )
    assert report["referenced"] == {}
    assert set(report["uploaded"]) == {meta["path"] for meta in ia_column_meta.values()}


def test_load_dataframe__modified_copy_uploads_normally(
    request, client, client_ns, ia_dataframe, ia_column_meta
):
    """A near-duplicate (a few values changed) does not match: the modified
    column uploads as a real column while untouched columns become
    references; updating through a reference is refused with guidance."""
    source_ns = client_ns.namespace
    with client_ns.context(notes="source load") as ctx:
        columns = {name: ctx.columns.create(**meta) for name, meta in ia_column_meta.items()}
        layer = ctx.geo_layers.create(path="counties", description="t")
        locality = ctx.localities.create(
            canonical_path=f"{source_ns}-loc", name=f"{source_ns} loc"
        )
        ctx.load_dataframe(
            df=ia_dataframe,
            columns_to_update=columns,
            create_geos=True,
            namespace=source_ns,
            layer=layer,
            locality=locality,
        )

    modified = ia_dataframe.copy()
    name_col = "NAME20"
    modified.loc[modified.index[:2], name_col] = ["Changed A", "Changed B"]

    user_ns = f"{source_ns}-user"
    with client.context(notes="user ns") as ctx:
        ctx.namespaces.create(path=user_ns, description="t", public=True)
    client.namespace = user_ns
    with client.context(notes="user load") as ctx:
        # Pre-create only the column we modified; the rest should match and
        # become references.
        modified_col = ctx.columns.create(namespace=user_ns, **ia_column_meta[name_col])
        user_layer = ctx.geo_layers.create(path="counties", description="t", namespace=user_ns)
        cols = dict(columns)
        cols[name_col] = modified_col
        report = ctx.load_dataframe(
            df=modified,
            columns_to_update=cols,
            create_geos=True,
            namespace=user_ns,
            layer=user_layer,
            locality=locality,
        )
    mod_path = ia_column_meta[name_col]["path"]
    assert mod_path in report["uploaded"]
    assert mod_path not in report["referenced"]
    other_paths = {m["path"] for k, m in ia_column_meta.items() if k != name_col}
    assert set(report["referenced"]) == other_paths

    # Updating values through one of the references is refused with guidance.
    other_col = next(k for k in ia_column_meta if k != name_col)
    retry = ia_dataframe.copy()
    retry_path = ia_column_meta[other_col]["path"]
    retry[retry_path] = retry[other_col]
    with client.context(notes="write through reference") as ctx:
        import pytest as _pytest

        with _pytest.raises(ValueError, match="reference"):
            ctx.load_dataframe(
                df=retry,
                columns_to_update=[retry_path],
                include_geos=False,
                namespace=user_ns,
                layer=user_layer,
                locality=locality,
            )
