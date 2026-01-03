import click
import pandas as pd
from .geo import load_fuas, bbox_from_fua, pois_to_h3
from .io import pois_from_foursquare, pois_from_overture
from .core import compute_complexity

@click.group()
def main():
    """Amenity Complexity CLI"""
    pass

@main.command()
@click.option('--fua-path', required=True, help='Path to FUA file (GPKG/SHP)')
@click.option('--fua-id', required=True, help='FUA ID to analyze (matched against eFUA_ID or similar)')
@click.option('--source', default='fsq', type=click.Choice(['fsq', 'overture']), help='POI Source')
@click.option('--resolution', default=9, help='H3 Resolution')
@click.option('--output', default='output.csv', help='Output file path for unit complexity')
@click.option('--limit', type=int, default=None, help='Limit number of POIs (for testing)')
def compute(fua_path, fua_id, source, resolution, output, limit):
    """Compute amenity complexity for a given FUA."""
    click.echo(f"Loading FUA {fua_id} from {fua_path}...")
    try:
        fuas = load_fuas(fua_path)
    except Exception as e:
        click.echo(f"Error loading FUAs: {e}")
        return

    # Attempt to find the ID column
    id_col = 'eFUA_ID'
    if id_col not in fuas.columns:
        possible = [c for c in fuas.columns if 'ID' in c and 'FUA' in c]
        if possible:
            id_col = possible[0]
            click.echo(f"Using inferred ID column: {id_col}")
        else:
            # Fallback to checking any ID-like column or erroring
            click.echo(f"Warning: 'eFUA_ID' not found. Available columns: {list(fuas.columns)}")
            # We'll try to proceed differently or user might need to specify column. 
            # For now, let's assume if eFUA_ID isn't there, we might fail or need a generic id arg.
            # But bbox_from_fua requires a col.
            click.echo("Please ensure FUA file has 'eFUA_ID' or similar.")
            return

    try:
        click.echo(f"Calculating BBox for {fua_id}...")
        bbox = bbox_from_fua(fuas, id_col, fua_id)
        click.echo(f"BBox: {bbox}")
    except ValueError as e:
        click.echo(f"Error finding FUA bbox: {e}")
        return

    click.echo(f"Fetching POIs from {source}...")
    try:
        if source == 'fsq':
            pois = pois_from_foursquare(bbox=bbox, limit=limit)
        elif source == 'overture':
            pois = pois_from_overture(bbox=bbox, limit=limit)
        else:
            click.echo(f"Unknown source: {source}")
            return
    except Exception as e:
        click.echo(f"Error fetching POIs: {e}")
        return

    click.echo(f"Loaded {len(pois)} POIs.")
    if pois.empty:
        click.echo("No POIs found. Exiting.")
        return

    click.echo(f"Assigning H3 resolution {resolution}...")
    pois = pois_to_h3(pois, 'latitude', 'longitude', resolution)
    h3_col = f"h3_lvl{resolution}"

    click.echo("Computing complexity profile...")
    # compute_complexity takes the long dataframe
    try:
        profile = compute_complexity(
            pois,
            unit_col=h3_col,
            category_col='top_category',
            min_unit_total=5,  # sane defaults for CLI
            min_category_total=5
        )
    except Exception as e:
        click.echo(f"Error computing complexity: {e}")
        return

    # User asked for a CSV output. We usually save the units table.
    # profile.units contains n_amenities, diversity, and complexity scores.
    if profile.units is not None:
        click.echo(f"Saving results to {output}...")
        profile.units.to_csv(output)
        click.echo("Done.")
    else:
        click.echo("No complexity results generated (matrix might be empty after pruning).")

if __name__ == '__main__':
    main()
