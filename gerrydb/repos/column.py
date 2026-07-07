"""Repository for columns."""

from typing import Any, Optional, Union

import httpx
import numpy as np

from gerrydb.logging import log
from gerrydb.repos.base import (
    NamespacedObjectRepo,
    err,
    namespaced,
    normalize_path,
    online,
    write_context,
)
from gerrydb.schemas import (
    Column,
    ColumnCreate,
    ColumnKind,
    ColumnPatch,
    ColumnType,
    ColumnValue,
    Geography,
)


class ColumnRepo(NamespacedObjectRepo[Column]):
    """Repository for columns."""

    @err("Failed to create column")
    @namespaced
    @write_context
    @online
    def create(
        self,
        path: str,
        namespace: Optional[str] = None,
        *,
        column_kind: ColumnKind,
        column_type: ColumnType,
        description: str,
        source_url: Optional[str] = None,
        aliases: Optional[list[str]] = None,
    ) -> Column:
        """Creates a tabular data column.

        Args:
            canonical_path: A short identifier for the column (e.g. `total_pop`).
            column_kind: Meaning of the column -- is a column value a `count`, a
                `percent`, a `categorical` label, or something `other`?
            column_type: Data type of the column (`int`, `float`, `bool`, `str`,
                or `json`-serializable blob).
            description: Longform description of the column
                (e.g. `2020 U.S. Census total population`).
            source_url: Optional original source of the column
                (e.g. a link to documentation on the U.S. Census Bureau website).
             aliases: Alternate short identifiers for the column.
                For instance, a column might be referred to by its numerical
                Census identifier and a more descriptive name.

        Raises:
            RequestError: If the column cannot be created on the server side,
                if the parameters fail validation, or if no namespace is provided.

        Returns:
            Metadata for the new column.
        """
        path = normalize_path(path)
        log.debug("Creating column at path: %s in namespace: %s", path, namespace)
        log.debug("POST request to %s", f"{self.base_url}/{namespace}")
        response = self.ctx.client.post(
            f"{self.base_url}/{namespace}",
            json=ColumnCreate(
                canonical_path=path,
                namespace=namespace,
                description=description,
                kind=column_kind,
                type=column_type,
                source_url=source_url,
                aliases=aliases,
            ).model_dump(mode="json"),
        )
        response.raise_for_status()

        return self.schema(**response.json())

    @err("Failed to update column")
    @namespaced
    @write_context
    @online
    def update(self, path: str, namespace: Optional[str] = None, *, aliases: list[str]) -> Column:
        """Updates a tabular data column.

        Currently, only adding aliases is supported.

        Args:
            path: Short identifier for the column.
            aliases: Alternate short identifiers to add to the column.

        Raises:
            RequestError: If the column cannot be created on the server side,
                or if the parameters fail validation.

        Returns:
            The updated column.
        """
        clean_path = normalize_path(f"{self.base_url}/{namespace}/{path}")
        response = self.ctx.client.patch(
            clean_path,
            json=ColumnPatch(aliases=aliases).model_dump(mode="json"),
        )
        response.raise_for_status()

        return Column(**response.json())

    # all() and get() come from NamespacedObjectRepo: the base methods honor
    # the resolved namespace argument; earlier overrides here silently read
    # session.namespace instead.

    @err("Failed to run column preflight")
    @online
    def preflight_duplicates(self, candidates: list[dict], namespace: str) -> list[dict]:
        """Asks the server which candidate columns' content already exists.

        Each candidate is {name, locality, layer, hash_hi, hash_lo}; the
        response pairs each name with the namespace/path of a readable
        column holding identical content, or nulls.
        """
        response = self.session.client.post(
            f"/column-refs/{namespace}/preflight", json={"candidates": candidates}
        )
        response.raise_for_status()
        return response.json()["results"]

    @err("Failed to create column reference")
    @write_context
    @online
    def create_reference(
        self,
        path: str,
        *,
        target_namespace: str,
        target_path: str,
        namespace: str,
        validate_paths: bool = False,
    ) -> dict:
        """Creates a reference in `namespace` to an existing column.

        References may only target columns in public namespaces (or the
        caller's own); the referenced values are never copied. With
        `validate_paths`, the server refuses the reference if the target
        column carries current values on geography paths missing from
        `namespace`.
        """
        response = self.ctx.client.post(
            f"/column-refs/{namespace}",
            json={
                "path": path,
                "target_namespace": target_namespace,
                "target_path": target_path,
                "validate_paths": validate_paths,
            },
        )
        response.raise_for_status()
        return response.json()

    @err("Failed to clone column")
    @namespaced
    @write_context
    @online
    def clone(
        self,
        path: Union[str, list[str]],
        namespace: Optional[str] = None,
        *,
        from_namespace: str,
        from_path: Union[str, list[str], None] = None,
        validate_paths: bool = False,
    ) -> Union[dict, list[dict]]:
        """Clones one or more columns from another namespace as local references.

        No values are copied: reads resolve through the source columns, and
        each clone's content fingerprint equals its source's wherever the two
        namespaces' geography paths align. The first write that changes a
        clone's values materializes it into an owned column (see
        `load_dataframe`'s `allow_local_updates` flow).

        `from_path` defaults to the same name(s) as `path`; when both are
        lists they pair up positionally. A single path returns one result
        dict, a list returns a list.
        """
        paths = [path] if isinstance(path, str) else list(path)
        if from_path is None:
            from_paths = paths
        elif isinstance(from_path, str):
            from_paths = [from_path]
        else:
            from_paths = list(from_path)
        if len(paths) != len(from_paths):
            raise ValueError(
                f"Got {len(paths)} clone path(s) but {len(from_paths)} source path(s)."
            )
        results = [
            self.create_reference(
                local,
                target_namespace=from_namespace,
                target_path=source,
                namespace=namespace,
                validate_paths=validate_paths,
            )
            for local, source in zip(paths, from_paths)
        ]
        return results[0] if isinstance(path, str) else results

    @err("Failed to load columns")
    def all(
        self, namespace: Optional[str] = None, *, include_references: bool = False
    ) -> list[Column]:
        """Gets all columns in a namespace.

        Plain listings hold only columns the namespace owns; with
        `include_references`, cross-namespace references (e.g. clones) are
        appended, labeled by their local path.
        """
        namespace = self.session.namespace if namespace is None else namespace
        if not include_references:
            return super().all(namespace=namespace)
        response = self.session.client.get(
            f"{self.base_url}/{namespace}?include_references=true"
        )
        response.raise_for_status()
        return [self.schema(**obj) for obj in response.json()]

    @err("Failed to materialize column reference")
    @namespaced
    @write_context
    @online
    def materialize(self, path: str, namespace: Optional[str] = None) -> Column:
        """Materializes a cross-namespace reference into an owned column.

        The server copies the source's current values onto same-path
        geographies in `namespace` and repoints the namespace's refs;
        existing views and template versions keep the source column.
        """
        response = self.ctx.client.post(f"{self.base_url}/{namespace}/{path}/materialize")
        response.raise_for_status()
        return self.schema(**response.json())

    def _materialize_if_reference(self, path: str, namespace: str) -> None:
        """Materializes `path` first if it resolves through a cross-namespace
        reference (no-op for owned columns and unknown paths, whose errors
        surface from the write itself)."""
        try:
            col = self.get(path, namespace=namespace)
        except Exception:
            return
        if col is not None and col.namespace != namespace:
            log.warning(
                "Materializing '%s/%s' (a reference to %s/%s) before a divergent write.",
                namespace,
                path,
                col.namespace,
                col.path,
            )
            self.materialize(path, namespace=namespace)

    @err("Failed to set column values")
    @namespaced
    @write_context
    @online
    def set_values(
        self,
        path: Optional[str] = None,
        namespace: Optional[str] = None,
        *,
        col: Optional[Column] = None,
        values: dict[Union[str, Geography], Any],
        allow_local_updates: bool = False,
    ) -> None:
        """Sets the values of a column on a collection of geographies.

        Args:
            path: Short identifier for the column. Only this or `col` should be provided.
                If both are provided, the path attribute of `col` will be used in place
                of the passed `path` argument.
            col: `Column` metadata object. If the `path` is not provided, the column's
                path will be used.
            namespace: Namespace of the column (used when `path_or_col` is a raw path).
            values:
                A mapping from geography paths or `Geography` metadata objects
                to column values.

        Raises:
            RequestError: If the values cannot be set on the server side.
        """
        assert path is None or isinstance(path, str)
        assert col is None or isinstance(col, Column)

        if path is None and col is None:
            raise ValueError("Either `path` or `col` must be provided.")

        path = col.path if col is not None else path
        if allow_local_updates:
            self._materialize_if_reference(path, namespace)
        clean_path = normalize_path(f"{self.base_url}/{namespace}/{path}")

        response = self.ctx.client.put(
            clean_path,
            json=[
                ColumnValue(
                    path=(f"/{geo.namespace}/{geo.path}" if isinstance(geo, Geography) else geo),
                    value=value,
                ).model_dump(mode="json")
                for geo, value in values.items()
            ],
        )
        response.raise_for_status()

        # TODO: what's the proper caching behavior here?

    @err("Failed to set column values")
    @namespaced
    @write_context
    @online
    async def async_set_values(
        self,
        path: Optional[str] = None,
        namespace: Optional[str] = None,
        *,
        col: Optional[Column] = None,
        values: dict[Union[str, Geography], Any],
        client: Optional[httpx.AsyncClient] = None,
        allow_local_updates: bool = False,
    ) -> None:
        """Asynchronously sets the values of a column on a collection of geographies.

        Args:
            path: Short identifier for the column. Only this or `col` should be provided.
                If both are provided, the path attribute of `col` will be used in place
                of the passed `path` argument.
            col: `Column` metadata object. If the `path` is not provided, the column's
                path will be used.
            namespace: Namespace of the column (used when `path_or_col` is a raw path).
            values:
                A mapping from geography paths or `Geography` metadata objects
                to column values.
            client: Asynchronous API client to use (for efficient connection pooling
                across batched requests).

        Raises:
            RequestError: If the values cannot be set on the server side.
        """
        assert path is None or isinstance(path, str)
        assert col is None or isinstance(col, Column)

        if path is None and col is None:
            raise ValueError("Either `path` or `col` must be provided.")

        path = col.path if col is not None else path
        if allow_local_updates:
            # Shared with the sync path: materialize-once, then no-op.
            self._materialize_if_reference(path, namespace)
        clean_path = normalize_path(f"{self.base_url}/{namespace}/{path}")

        ephemeral_client = client is None
        if ephemeral_client:
            params = self.ctx.client_params.copy()
            params["transport"] = httpx.AsyncHTTPTransport(retries=1)
            client = httpx.AsyncClient(**params)

        # Peter Note: the geos are generally strings here
        json = [
            ColumnValue(
                path=(f"/{geo.namespace}/{geo.path}" if isinstance(geo, Geography) else geo),
                value=_coerce(value),
            ).model_dump(mode="json")
            for geo, value in values.items()
        ]
        log.debug("PUT request to %s", clean_path)
        response = await client.put(
            clean_path,
            json=json,
        )

        if response.status_code != 204:
            log.debug(f"For path = {path} and col = {col} returned {response}")

        response.raise_for_status()

        if ephemeral_client:
            await client.aclose()


def _coerce(val: Any) -> Any:  # pragma: no cover
    """Coerces values for JSON serialization."""
    if isinstance(val, np.int64):
        return int(val)
    if isinstance(val, np.float64):
        return float(val)
    return val
