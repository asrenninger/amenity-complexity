from __future__ import annotations

from typing import Any, Optional

import geopandas as gpd
import pandas as pd


def _is_na(x: object) -> bool:
    """Return True for None / pandas NA / NaN."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _get_field(obj: object, key: str) -> Any:
    """Safely pull a field off dict-like / struct-like objects."""
    if _is_na(obj):
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return getattr(obj, key)
    except Exception:
        return None


def _first(x: object) -> Any:
    """Return the first item of a list/tuple, else the value itself."""
    if _is_na(x):
        return None
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else None
    return x


def parse_geometry(
    df: pd.DataFrame,
    crs: str = "EPSG:3035",
    geometry_col: str = "geometry",
) -> gpd.GeoDataFrame:
    from shapely import wkt, wkb

    def to_geom(x):
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        if isinstance(x, str):
            return wkt.loads(x)

        if isinstance(x, (bytes, bytearray, memoryview)):
            return wkb.loads(bytes(x))

        if isinstance(x, list) and all(isinstance(i, int) for i in x):
            return wkb.loads(bytes(x))

        try:
            return wkt.loads(str(x))
        except Exception:
            return None

    gs = gpd.GeoSeries(df[geometry_col].apply(to_geom), index=df.index)
    gf = gpd.GeoDataFrame(df.copy(), geometry=gs, crs="EPSG:4326")
    return gf.to_crs(crs)


def parse_foursquare_categories(
    df: pd.DataFrame,
    data_source: str = "fsq",
    category_col: str = "category",
    sep: str = " > ",
    levels: int = 2,
    *,
    already_split: bool = False,
) -> pd.DataFrame:
    """
    Add hierarchical category columns:
      - top_category (level 0 name)
      - sub_category (level 1 name, optional)
    """
    out = df.copy()

    # (The data_source branch is kept for backward compatibility, but the
    # parsing logic is identical.)
    if already_split:
        parts = out[category_col]
    else:
        parts = out[category_col].astype("string").str.split(sep)

    out["top_category"] = parts.str[0]
    if levels >= 2:
        out["sub_category"] = parts.str[1]

    return out


def parse_overture_categories(
    df: pd.DataFrame,
    *,
    levels: int = 2,
    overwrite: bool = False,
    overwrite_cols: Optional[list[str]] = None,
    basic_category_col: str = "basic_category",
    categories_col: str = "categories",
    taxonomy_col: str = "taxonomy",
) -> pd.DataFrame:
    """
    Parse Overture Places categories into standardized hierarchical columns.

    Adds:
      - top_category
      - sub_category  (only when levels >= 2)

    Strategy:
      - top_category: prefer `basic_category` if present, else `categories.primary`,
        else fall back to `taxonomy.hierarchy[0]` / `taxonomy.primary`.
      - sub_category: prefer `taxonomy.primary` (most specific), else `categories.primary`.

    overwrite semantics:
      - overwrite=True overwrites both columns
      - overwrite_cols=["sub_category"] overwrites only those listed
    """
    out = df.copy()

    # Allow granular overwrite (e.g., only recompute sub_category).
    cols = {"top_category", "sub_category"} if overwrite else set()
    if overwrite_cols:
        cols |= set(overwrite_cols)
    overwrite_top = "top_category" in cols
    overwrite_sub = "sub_category" in cols

    # ------------------
    # top_category
    # ------------------
    if overwrite_top or "top_category" not in out.columns:
        if basic_category_col in out.columns:
            top = out[basic_category_col]
        elif categories_col in out.columns:
            top = out[categories_col].apply(lambda o: _first(_get_field(o, "primary")))
        elif taxonomy_col in out.columns:
            # taxonomy.hierarchy is ordered from general -> specific
            top = out[taxonomy_col].apply(
                lambda o: _first(_get_field(o, "hierarchy")) or _first(_get_field(o, "primary"))
            )
        else:
            raise KeyError(
                "Could not determine Overture category columns. Expected one of: "
                f"'{basic_category_col}', '{categories_col}', '{taxonomy_col}'."
            )
        out["top_category"] = pd.Series(top, index=out.index).astype("string")

    # ------------------
    # sub_category
    # ------------------
    if levels >= 2 and (overwrite_sub or "sub_category" not in out.columns):
        if taxonomy_col in out.columns:
            sub = out[taxonomy_col].apply(lambda o: _first(_get_field(o, "primary")))
        elif categories_col in out.columns:
            sub = out[categories_col].apply(lambda o: _first(_get_field(o, "primary")))
        else:
            sub = None
        out["sub_category"] = pd.Series(sub, index=out.index).astype("string")

    return out


def parse_categories(
    df: pd.DataFrame,
    *,
    data_source: str,
    levels: int = 2,
    # Foursquare options
    category_col: str = "category",
    sep: str = " > ",
    already_split: bool = False,
    # Overture options
    overwrite: bool = False,
    overwrite_cols: Optional[list[str]] = None,
    basic_category_col: str = "basic_category",
    categories_col: str = "categories",
    taxonomy_col: str = "taxonomy",
) -> pd.DataFrame:
    """Dispatch to the correct parser by data source."""
    ds = str(data_source).strip().lower()

    if ds in {"fsq", "foursquare", "foursquare_os", "foursquare_open_source_places"}:
        return parse_foursquare_categories(
            df,
            data_source="fsq",
            category_col=category_col,
            sep=sep,
            levels=levels,
            already_split=already_split,
        )

    if ds in {"overture", "omf", "overturemaps"}:
        return parse_overture_categories(
            df,
            levels=levels,
            overwrite=overwrite,
            overwrite_cols=overwrite_cols,
            basic_category_col=basic_category_col,
            categories_col=categories_col,
            taxonomy_col=taxonomy_col,
        )

    raise ValueError(f"Unknown data_source='{data_source}'. Expected 'fsq' or 'overture'.")

def get_categories():
    categories = pd.read_csv("https://raw.githubusercontent.com/OvertureMaps/schema/refs/heads/main/docs/schema/concepts/by-theme/places/overture_categories.csv")
    return categories

def relevel_categories():
    categories = get_categories()
    return categories