# Amenity Complexity

Tools for extracting points of interest (POIs) from **Foursquare Open Source Places** and **Overture Places**, aggregating them to spatial units (e.g., **H3**), and computing **amenity complexity** for any region.

The core workflow is:

**region → POIs → unit × category matrix → RCA → binary specialization matrix → complexity scores**

## Quickstart

This example uses a **Functional Urban Area (FUA)** to define the region, fetches POIs from **Foursquare**, aggregates to **H3**, and computes complexity.

```python

from amenity_complexity.geo import load_fuas, bbox_from_fua, pois_to_h3
from amenity_complexity.io import pois_from_foursquare
from amenity_complexity.core import compute_complexity

# 1) Load FUAs and pick a city
fuas = load_fuas("data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
city = "Vienna"
row = fuas.loc[fuas["eFUA_name"] == city].iloc[0]

# 2) Convert the FUA polygon to a buffered bbox (lon/lat)
bbox = bbox_from_fua(
    fuas,
    fua_id_col="eFUA_ID",
    fua_id=str(row["eFUA_ID"]),
    buffer_m=5000,
)

# 3) Fetch POIs (remote Parquet on public S3 via DuckDB)
pois = pois_from_foursquare(bbox=bbox, trim_cols=True)

# 4) Assign POIs to H3 cells
res = 8
pois = pois_to_h3(
    pois,
    lat_col="latitude",
    lon_col="longitude",
    resolution=res,
    geometry=False,
)

# 5) Compute complexity (units = H3 cells; categories = POI categories)
profile = compute_complexity(
    pois,
    unit_col=f"h3_lvl{res}",
    category_col="sub_category",  # or "top_category" for a coarser taxonomy
    methods=("juhasz", "hidalgo"),
)

```

## Installation

```bash

conda env create -f environment.yaml
conda activate amenity-complexity
pip install -e .

```

## Data sources

This package reads public Parquet releases directly with DuckDB:

- **Foursquare Open Source Places** (`pois_from_foursquare`)
- **Overture Places** (`pois_from_overture`)

Both functions accept a `bbox=(minx, miny, maxx, maxy)` in EPSG:4326 and a `release=...` string so you can pin a dataset version for reproducibility.

## What “amenity complexity” means here

We adapt the “economic complexity” toolkit to intra-urban amenities:

- Build a unit × category count matrix (e.g., H3 cells × amenity categories)
- Compute **RCA** (revealed comparative advantage)
- Binarize to a specialization matrix **M** (`RCA >= 1` by default)
- Compute:
  - **diversity** (row sums of M) and **ubiquity** (column sums of M)
  - **complexity scores** for units and categories

## Methods implemented

`compute_complexity(..., methods=("juhasz", "hidalgo"))` runs two closely related variants:

- **`juhasz`**: “amenity complexity 
  This matches the amenity / neighborhood complexity construction used by **Juhász et al. (2023)**, where complexity is read from the structure of the binary RCA matrix via the second eigenvector of similarity.  
  Paper: *Amenity complexity and urban locations of socio-economic mixing* (EPJ Data Science, 2023).  
  https://doi.org/10.1140/epjds/s13688-023-00413-6

- **`hidalgo`**: economic complexity
  This follows the economic complexity tradition introduced by **Hidalgo & Hausmann (2009)** (RCA → binary **M** → reflections / eigenvectors), using a normalized operator so degree (diversity / ubiquity) is accounted for more explicitly.  
  Paper: *The building blocks of economic complexity* (PNAS, 2009).  
  https://doi.org/10.1073/pnas.0900943106

Both methods return unit-side and category-side scores; in `profile.units` these appear as `complexity_juhasz` and `complexity_hidalgo` (and similarly in `profile.categories`). Scores are z-scored and sign-oriented for stable interpretation.

## API overview

- `amenity_complexity.io`
  - `pois_from_foursquare(...)`
  - `pois_from_overture(...)`

- `amenity_complexity.geo`
  - `load_fuas(...)`, `bbox_from_fua(...)`
  - `pois_to_h3(...)`, `h3_to_polygon(...)`

- `amenity_complexity.core`
  - `compute_complexity(...)` (end-to-end)
  - lower-level building blocks: `count_matrix`, `rca`, `specialization`, `complexity`

## Notes

- Category harmonization (FSQ ↔ Overture) is a work in progress; today you choose `top_category` / `sub_category` within each source.

## Notebook

- `notebooks/01_exploratory.ipynb` runs the full pipeline end-to-end.

## References

- Juhász, S. et al. (2023). *Amenity complexity and urban locations of socio-economic mixing.* EPJ Data Science, 12:34. https://doi.org/10.1140/epjds/s13688-023-00413-6  
- Hidalgo, C. A., & Hausmann, R. (2009). *The building blocks of economic complexity.* PNAS, 106(26), 10570–10575. https://doi.org/10.1073/pnas.0900943106
