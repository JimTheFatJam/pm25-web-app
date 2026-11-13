WorldPop population rasters
============================

Place one GeoTIFF per year here, covering the full 2015-2030 range you
have stated is available from WorldPop. Expected filename pattern,
matching the naming convention already used by the research pipeline's
own extract_population.py:

    phl_pop_<YEAR>_CN_1km_R2025A_UA_v1.tif

Example filenames:
    phl_pop_2015_CN_1km_R2025A_UA_v1.tif
    phl_pop_2016_CN_1km_R2025A_UA_v1.tif
    ...
    phl_pop_2030_CN_1km_R2025A_UA_v1.tif

Notes:
- Each file's 4-digit year is what the backend uses to select the correct
  raster for a given prediction year -- the exact filename prefix/suffix
  around that year does not need to match this pattern exactly, as long
  as a 4-digit year (2015-2030) appears somewhere in the filename.
- Only supply the years you actually have -- do not fabricate missing
  years. The backend's population-loading logic must reject or clearly
  flag (never silently substitute) a requested prediction year that has
  no matching local file.
- IMPORTANT: the research pipeline's own extract_population.py has a
  hardcoded filename-year regex that only recognizes 2020-2026. The
  backend's own loading logic (not this script) must use a widened
  pattern to actually make use of any 2015-2019 or 2027-2030 files placed
  here -- see the main deployment plan, Section 4.
- You do not need to provide every year immediately -- place what you
  have now; the backend should be designed to work with however many
  years actually exist in this folder at any given time.
