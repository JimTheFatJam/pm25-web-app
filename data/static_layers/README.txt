Road density cache (REQUIRED before running any real prediction)
====================================================================

backend/prediction/static_predictors.py requires this exact file:

    road_density_by_cell.csv

with columns:

    grid_id, road_density_local, road_density_neighborhood_mean

Why this can't be computed live per request
--------------------------------------------
Road density comes from parsing philippines-latest.osm.pbf
(data/local_datasets/osm/). That parse is expensive regardless of how few
cells a prediction request covers -- and reimplementing that parsing
logic fresh, rather than reusing your own already-validated
extraction script, would risk it silently drifting from what the model
was actually trained on. So this backend does NOT parse the .pbf file
itself. Instead, you generate this cache ONCE using your own existing,
validated road-density extraction script (extract_road_density_local.py /
extract_road_density_streaming.py), run against the FULL prediction grid
(official_grid_ph_full.gpkg, 313,845 cells) rather than just the 915
training cells it was originally run against.

How to generate it
--------------------
In your research environment, run your existing road-density extraction
script against data/official_grid_ph_full.gpkg (copy or point the script
at this file instead of the 915-cell training grid), producing local and
neighborhood_mean road density for every cell. Save the result here as
road_density_by_cell.csv with the exact column names above.

If you only need a handful of cells for a first small test, you do NOT
need to run this across the full 313,845-cell grid immediately -- a
partial cache covering just the cells you intend to test is enough to get
started. The backend will fail clearly, naming exactly which requested
cell(s) are missing from the cache, rather than silently substituting a
default value, so you'll know immediately if a cell you selected isn't
covered yet.

Nothing else needs to be precomputed for this diagnostic-testing stage.
Elevation, land cover, NDVI, and nighttime lights are fetched live from
GEE per request (see static_predictors.py) -- fine for a small test, not
optimized for scale (see the deployment plan, Section 4, for the
eventual production design: precomputed once for the full grid, refreshed
on a schedule).
