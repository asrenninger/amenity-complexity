import duckdb
import pandas as pd
from typing import Optional, Tuple, List

BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)

# Optional: convert DuckDB GEOMETRY to shapely (if shapely is installed and duckdb returns WKB-ish)
try:
    from shapely import wkb  # type: ignore
except Exception:  # pragma: no cover
    wkb = None


# -----------------------------
# DuckDB setup
# -----------------------------

def get_duckdb_connection(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=database)
    con.execute("INSTALL httpfs;")
    con.execute("INSTALL spatial;")
    con.execute("LOAD httpfs;")
    con.execute("LOAD spatial;")
    con.execute("SET enable_progress_bar = false;")
    return con


_SCHEMA_CACHE: dict[str, set[str]] = {}


def _parquet_columns(con: duckdb.DuckDBPyConnection, parquet_url: str) -> set[str]:
    """
    Return top-level column names for a parquet dataset without scanning rows.
    Cached per parquet_url.
    """
    if parquet_url in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[parquet_url]

    df0 = con.execute(
        f"SELECT * FROM read_parquet('{parquet_url}', hive_partitioning=1) LIMIT 0"
    ).df()
    cols = set(df0.columns)
    _SCHEMA_CACHE[parquet_url] = cols
    return cols


def _nullify_empty(expr: str) -> str:
    """SQL helper: treat empty strings as NULL so concat_ws doesn't include blank parts."""
    return f"NULLIF({expr}, '')"


def _safe_first_item(x):
    """[a,b] -> a; <NA>/None/[] -> None; scalar -> scalar"""
    if x is None:
        return None
    try:
        if x is pd.NA:
            return None
    except Exception:
        pass
    if isinstance(x, str):
        return x
    try:
        return x[0] if len(x) > 0 else None
    except Exception:
        return None


def _convert_geometry_series(geom: pd.Series) -> pd.Series:
    """
    Convert DuckDB GEOMETRY column to something usable.
    - If duckdb returns bytes-like WKB and shapely is available -> shapely geometry
    - Else: leave as-is (DuckDB might already give an object you can keep)
    """
    if wkb is None:
        return geom
    # Try converting bytes/memoryview entries
    def conv(g):
        if g is None or (isinstance(g, float) and pd.isna(g)):
            return None
        if isinstance(g, (bytes, bytearray, memoryview)):
            try:
                return wkb.loads(bytes(g))
            except Exception:
                return g
        return g
    return geom.apply(conv)


# -----------------------------
# POI extraction: Foursquare
# -----------------------------

def pois_from_foursquare(
    *,
    release: str = "2025-02-06",
    bbox: Optional[BBox] = None,
    limit: Optional[int] = None,
    trim_cols: bool = True,
    include_closed: bool = False,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Foursquare OSS Places -> POIs.

    Core columns (always):
      id, name, top_category, sub_category, longitude, latitude

    Extras (trim_cols=False):
      categories, address, website, telephone, geometry

    Note: operating_status is not provided for FSQ (date_closed exists but often null).
    """
    close_con = False
    if con is None:
        con = get_duckdb_connection()
        close_con = True

    con.execute("SET s3_region='us-east-1'")
    parquet_url = f"s3://fsq-os-places-us-east-1/release/dt={release}/places/parquet/*"
    available = _parquet_columns(con, parquet_url)

    required = {"fsq_place_id", "name", "latitude", "longitude", "fsq_category_labels"}
    missing = sorted(list(required - available))
    if missing:
        raise ValueError(f"FSQ release {release} missing required columns: {missing}")

    # Primary category label for top/sub parsing (DuckDB list indexing is 1-based)
    cat0 = "fsq_category_labels[1]"
    top_expr = f"split_part({cat0}, ' > ', 1) AS top_category"
    sub_expr = f"NULLIF(split_part({cat0}, ' > ', 2), '') AS sub_category"

    select_exprs = [
        "fsq_place_id AS id",
        "name",
        top_expr,
        sub_expr,
        "longitude",
        "latitude",
    ]

    if not trim_cols:
        select_exprs.append("fsq_category_labels AS categories")

        # One comparable address string
        # Use whichever components exist; missing ones become NULL.
        parts = []
        for col in ["address", "locality", "region", "postcode", "country"]:
            parts.append(_nullify_empty(col) if col in available else "NULL")
        select_exprs.append(f"concat_ws(', ', {', '.join(parts)}) AS address")

        select_exprs.append(("website AS website") if "website" in available else "NULL AS website")
        select_exprs.append(("tel AS telephone") if "tel" in available else "NULL AS telephone")

        # FSQ geometry column might be `geom` (WKB) in some releases; keep if present
        if "geom" in available:
            select_exprs.append("ST_AsText(geom) AS geometry")
        else:
            select_exprs.append("NULL AS geometry")

    query = (
        "SELECT\n  " + ",\n  ".join(select_exprs) +
        f"\nFROM read_parquet('{parquet_url}', hive_partitioning=1)"
    )

    conditions: List[str] = []
    if not include_closed and "date_closed" in available:
        conditions.append("date_closed IS NULL")

    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        conditions.append(f"latitude BETWEEN {miny} AND {maxy}")
        conditions.append(f"longitude BETWEEN {minx} AND {maxx}")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    df = con.execute(query).df()

    core_cols = ["id", "name", "top_category", "sub_category", "longitude", "latitude"]
    if df.empty:
        if close_con:
            con.close()
        if trim_cols:
            return pd.DataFrame(columns=core_cols)
        return pd.DataFrame(columns=core_cols + ["categories", "address", "website", "telephone", "geometry", "operating_status"])

    if trim_cols:
        out = df[core_cols].copy()
        out["operating_status"] = None  # align columns with Overture output
    else:
        df["geometry"] = _convert_geometry_series(df["geometry"])
        out = df[core_cols + ["categories", "address", "website", "telephone", "geometry"]].copy()
        out["operating_status"] = None  # align

    if close_con:
        con.close()

    # Ensure consistent column order with Overture
    final_cols = core_cols + ["categories", "address", "website", "telephone", "geometry", "operating_status"]
    for c in final_cols:
        if c not in out.columns:
            out[c] = None
    return out[final_cols] if not trim_cols else out[core_cols + ["operating_status"]]


# -----------------------------
# POI extraction: Overture
# -----------------------------

def pois_from_overture(
    *,
    release: str = "2025-12-17.0",
    bbox: Optional[BBox] = None,
    limit: Optional[int] = None,
    trim_cols: bool = True,
    include_closed: bool = False,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Overture Places -> POIs.

    Core columns (always):
      id, name, top_category, sub_category, longitude, latitude

    - top_category := basic_category
    - sub_category := first element of categories.alternate (or NULL)
    - longitude/latitude := ST_X(geometry), ST_Y(geometry) (geometry is GEOMETRY in your setup)

    Extras (trim_cols=False):
      categories, address, website, telephone, geometry, operating_status

    Efficiency:
      - If bbox exists, filter with bbox.* pushdown inside read_parquet (fast)
      - Avoid ST_CENTROID and avoid ST_GeomFromWKB
    """
    close_con = False
    if con is None:
        con = get_duckdb_connection()
        close_con = True

    con.execute("SET s3_region='us-west-2'")
    parquet_url = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    available = _parquet_columns(con, parquet_url)

    required = {"id", "names", "basic_category", "categories", "geometry"}
    missing = sorted(list(required - available))
    if missing:
        raise ValueError(f"Overture release {release} missing required columns: {missing}")

    # Build WHERE for pushdown
    where_clauses: List[str] = []

    if not include_closed and "operating_status" in available:
        where_clauses.append("(operating_status IS NULL OR operating_status != 'closed')")

    bbox_pushdown = False
    if bbox is not None and "bbox" in available:
        bbox_pushdown = True
        minx, miny, maxx, maxy = bbox
        where_clauses.append(f"bbox.xmin >= {minx} AND bbox.xmax <= {maxx}")
        where_clauses.append(f"bbox.ymin >= {miny} AND bbox.ymax <= {maxy}")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # CTE does filtering first (fast) then computes lon/lat
    cte_select = [
        "id",
        "names.primary AS name",
        "basic_category AS top_category",
        "categories.alternate[1] AS sub_category",
        "ST_X(geometry) AS longitude",
        "ST_Y(geometry) AS latitude",
    ]

    if not trim_cols:
        cte_select.append("categories AS categories")
        if "addresses" in available:
            # DuckDB lists are 1-indexed; addresses[1] is the first address struct
            cte_select.append(
                "concat_ws(', ', "
                "NULLIF(addresses[1].freeform, ''), "
                "NULLIF(addresses[1].locality, ''), "
                "NULLIF(addresses[1].region, ''), "
                "NULLIF(addresses[1].postcode, ''), "
                "NULLIF(addresses[1].country, '')"
                ") AS address"
            )
        else:
            cte_select.append("NULL AS address")
        cte_select.append("websites[1] AS website" if "websites" in available else "NULL AS website")
        cte_select.append("phones[1] AS telephone" if "phones" in available else "NULL AS telephone")
        cte_select.append("ST_AsText(geometry) AS geometry")
        cte_select.append("operating_status AS operating_status" if "operating_status" in available else "NULL AS operating_status")
    else:
        cte_select.append("operating_status AS operating_status" if "operating_status" in available else "NULL AS operating_status")

    query = (
        "SELECT\n  " + ",\n  ".join(cte_select) +
        f"\nFROM read_parquet('{parquet_url}', hive_partitioning=1)\n"
        f"{where_sql}"
    )

    # If bbox is requested but bbox column absent, filter on lon/lat (slower)
    if bbox is not None and not bbox_pushdown:
        minx, miny, maxx, maxy = bbox
        # add WHERE / AND correctly
        if "WHERE" in query:
            query += f"\nAND latitude BETWEEN {miny} AND {maxy} AND longitude BETWEEN {minx} AND {maxx}"
        else:
            query += f"\nWHERE latitude BETWEEN {miny} AND {maxy} AND longitude BETWEEN {minx} AND {maxx}"

    if limit is not None:
        query += f"\nLIMIT {int(limit)}"

    df = con.execute(query).df()

    core_cols = ["id", "name", "top_category", "sub_category", "longitude", "latitude"]

    if df.empty:
        if close_con:
            con.close()
        if trim_cols:
            return pd.DataFrame(columns=core_cols + ["operating_status"])
        return pd.DataFrame(columns=core_cols + ["categories", "address", "website", "telephone", "geometry", "operating_status"])

    # Normalize sub_category
    df["sub_category"] = df["sub_category"].apply(_safe_first_item)

    if not trim_cols and "geometry" in df.columns:
        df["geometry"] = _convert_geometry_series(df["geometry"])

    if close_con:
        con.close()

    if trim_cols:
        return df[core_cols + ["operating_status"]].copy()

    # Ensure columns align with FSQ output
    out = df[core_cols + ["categories", "address", "website", "telephone", "geometry", "operating_status"]].copy()
    return out
