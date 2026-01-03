from __future__ import annotations

import pandas as pd
import geopandas as gpd
from typing import Optional, Union

import pandas as pd
import geopandas as gpd

def parse_geometry(
    df: pd.DataFrame, 
    crs: str = "EPSG:3035", 
    geometry_col: str = "geometry"
) -> gpd.GeoDataFrame:
    from shapely import wkt, wkb

    def to_geom(x):
        if x is None:
            return None
        # pandas NA
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        # Case 1: WKT string
        if isinstance(x, str):
            return wkt.loads(x)

        # Case 2: bytes / memoryview WKB
        if isinstance(x, (bytes, bytearray, memoryview)):
            return wkb.loads(bytes(x))

        # Case 3: list-of-ints WKB (common when DuckDB returns GEOMETRY-ish blobs)
        if isinstance(x, list) and all(isinstance(i, int) for i in x):
            return wkb.loads(bytes(x))

        # Fallback: try string as WKT
        try:
            return wkt.loads(str(x))
        except Exception:
            return None

    gs = gpd.GeoSeries(df[geometry_col].apply(to_geom), index=df.index)

    gf = gpd.GeoDataFrame(df.copy(), geometry=gs, crs="EPSG:4326")
    gf = gf.to_crs(crs)
    return gf


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

    Parameters
    ----------
    already_split:
      If True, df[category_col] is assumed to be a list-like of levels.
      If False, it's assumed to be a string with `sep` separators.
    """
    out = df.copy()

    if data_source == "fsq":
        if already_split:
            parts = out[category_col]
        else:
            parts = out[category_col].astype("string").str.split(sep)

        out["top_category"] = parts.str[0]

        if levels >= 2:
            out["sub_category"] = parts.str[1]
    else:
        if already_split:
            parts = out[category_col]
        else:
            parts = out[category_col].astype("string").str.split(sep)

        out["top_category"] = parts.str[0]

        if levels >= 2:
            out["sub_category"] = parts.str[1]

    return out

def parse_overture_categories():

    return x

def parse_categories():

    return x

def get_categories():
    categories = pd.read_csv("https://raw.githubusercontent.com/OvertureMaps/schema/refs/heads/main/docs/schema/concepts/by-theme/places/overture_categories.csv")
    return categories

def relevel_categories():
    categories = get_categories()
    return categories