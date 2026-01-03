import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon, Point, box
import numpy as np
from typing import Tuple, Optional

BBox = Tuple[float, float, float, float]

def load_fuas(path: str) -> gpd.GeoDataFrame:
    """
    Load Functional Urban Areas (FUAs) from a file (GPKG, SHP, etc.).
    Ensures the CRS is EPSG:4326.
    """
    gdf = gpd.read_file(path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_countries(path: str) -> gpd.GeoDataFrame:
    """
    Load countries from a file (GPKG, SHP, etc.).
    Ensures the CRS is EPSG:4326.
    """
    gdf = gpd.read_file(path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _ensure_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")
    return gdf


def _bbox_from_id(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    id_value: str,
    *,
    fuzzy_match: bool = False,
    buffer_m: float = 5000.0,   # ~5 km
) -> BBox:
    gdf = _ensure_4326(gdf)

    s = gdf[id_col].astype("string")
    if fuzzy_match:
        mask = s.str.contains(str(id_value), case=False, regex=False, na=False)
    else:
        mask = s == str(id_value)

    geoms = gdf.loc[mask, "geometry"]
    if geoms.empty:
        raise ValueError(f"ID '{id_value}' not found in column '{id_col}'")

    # Union -> hull (nice and tight) in 4326
    geom = geoms.unary_union.convex_hull

    # Buffer in meters by projecting to a local UTM CRS
    gs = gpd.GeoSeries([geom], crs="EPSG:4326")
    utm = gs.estimate_utm_crs()  # picks UTM zone from centroid
    if utm is None:
        # Rare (e.g., poles); fallback to Web Mercator meters-ish
        utm = "EPSG:3857"

    geom_buf = gs.to_crs(utm).iloc[0].buffer(buffer_m)

    # Back to 4326 and return bounds
    geom_ll = gpd.GeoSeries([geom_buf], crs=utm).to_crs("EPSG:4326").iloc[0]
    minx, miny, maxx, maxy = geom_ll.bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def bbox_from_fua(
    fua_gdf: gpd.GeoDataFrame,
    fua_id_col: str,
    fua_id: str,
    *,
    fuzzy_match: bool = False,
    buffer_m: float = 5000.0,
) -> BBox:
    return _bbox_from_id(fua_gdf, fua_id_col, fua_id, fuzzy_match=fuzzy_match, buffer_m=buffer_m)


def bbox_from_country(
    country_gdf: gpd.GeoDataFrame,
    country_id_col: str,
    country_id: str,
    *,
    fuzzy_match: bool = False,
    buffer_m: float = 5000.0,
) -> BBox:
    return _bbox_from_id(country_gdf, country_id_col, country_id, fuzzy_match=fuzzy_match, buffer_m=buffer_m)


def point_to_h3(lat: float, lon: float, resolution: int) -> str:
    """Convert lat/lon to H3 index."""
    return h3.latlng_to_cell(lat, lon, resolution)


def pois_to_h3(df: pd.DataFrame, lat_col: str, lon_col: str, resolution: int, geometry: bool = False) -> pd.DataFrame:
    """
    Add an 'h3' column to the POI DataFrame with the resolution specified.
    """
    df[f'h3_lvl{resolution}'] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], resolution), 
        axis=1
    )
    if geometry:
        df['geometry'] = df[f'h3_lvl{resolution}'].apply(lambda x: Polygon(h3.cell_to_boundary(x)))
    return df


def h3_to_polygon(h3_index: str) -> Polygon:
    """
    Convert an H3 index to a Shapely Polygon.
    Handles the coordinate swap from (lat, lon) [H3] to (lon, lat) [Shapely].
    """
    # h3 returns (lat, lng); shapely wants (lng, lat)
    boundary = h3.cell_to_boundary(h3_index)
    return Polygon([(lng, lat) for (lat, lng) in boundary])


def aggregate_by_h3(df: pd.DataFrame, h3_col: str, category_col: str) -> pd.DataFrame:
    """
    Create the matrix of H3 cell x Category count.
    Returns a DataFrame where index is H3 cell and columns are categories.
    """
    # Group by H3 cell and category, then count
    counts = df.groupby([h3_col, category_col]).size().reset_index(name='count')
    
    # Pivot to wide format
    matrix = counts.pivot(index=h3_col, columns=category_col, values='count').fillna(0)
    
    return matrix


def aggregate_by_area(df: pd.DataFrame, area_col: str, category_col: str) -> pd.DataFrame:
    """
    Create the matrix of area x Category count.
    Returns a DataFrame where index is area and columns are categories.
    """
    # Group by area and category, then count
    counts = df.groupby([area_col, category_col]).size().reset_index(name='count')
    
    # Pivot to wide format
    matrix = counts.pivot(index=area_col, columns=category_col, values='count').fillna(0)
    
    return matrix