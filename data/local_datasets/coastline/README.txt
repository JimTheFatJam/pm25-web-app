Natural Earth coastline (for distance-to-coast)
=================================================

Place the coastline file here with this exact filename, matching the
research pipeline's own extract_distance_to_coast.py:

    ne_10m_coastline.gpkg

Source: Natural Earth, 1:10m scale (recommended by the research
pipeline's own script docstring -- do not substitute a coarser scale
without updating the extraction logic accordingly).

Notes:
- This is a single global file, not per-country -- no year, no per-country
  naming needed.
- Static feature: this file never needs refreshing on any schedule (coastlines
  do not meaningfully change over the timescales this project operates on).
- If you have this file under a different name or format (e.g. a shapefile
  instead of GeoPackage), either rename/convert it to match the expected
  filename above, or the backend's loading path will need updating to
  match whatever you actually place here -- flagging this rather than
  assuming a silent rename will always be safe.
