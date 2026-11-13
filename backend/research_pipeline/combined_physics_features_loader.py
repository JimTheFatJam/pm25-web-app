"""
Combined Feature Loader: Non-Fire Additional Features + Physics/Transport
================================================================================

Runtime-merges THREE layers on top of the original feature set, in
memory only (no new CSV, matching lag_features.py's and
additional_features_loader.py's established philosophy in this
project):
  1. Original features (unchanged, read-only)
  2. The non-fire subset of the additional features already validated in
     Experiment 1 (consecutive_dry_days + all calendar/holiday columns).
     Fire columns are explicitly EXCLUDED here -- Experiment 1 found
     every fire feature at exactly zero SHAP importance, later confirmed
     to be a near-constant/near-zero-variance signal specifically in the
     Philippine fine-tuning population, not a capacity or leakage issue.
     Carrying them forward would just re-add 14 zero-value columns to
     every future ablation for no reason.
  3. NEW physics (5 local-only interaction features) and wind-transport
     (6 PM2.5-free upwind-weighted pollutant features) columns, computed
     via DIRECT IMPORT of add_physics_features / add_wind_transport_features
     / build_cell_adjacency from run_physics_transport_experiment.py --
     not reimplemented. That module already fixed and verified a real
     180-degree wind-direction sign bug (documented in its own
     add_wind_transport_features docstring) -- reusing it directly avoids
     re-introducing that exact class of bug.

IMPORTANT METHODOLOGICAL FLAG (also raised directly before implementation
began): run_physics_transport_experiment.py fixes ALPHA=1.0, which WAS
the established best value at the time that script was built, but has
since been superseded by alpha=0.75 (established by a later alpha-
sensitivity experiment and used in every experiment since, including the
immediately-prior Experiment 1). This new experiment uses alpha=0.75,
NOT the 1.0 originally requested, to actually satisfy the stated
"exactly one variable changes, everything else matches the current best
validated pipeline" principle -- using 1.0 here would have silently
changed a SECOND variable (alpha) in addition to the intended one
(feature representation), contradicting that principle rather than
satisfying it.
"""

from pathlib import Path

import pandas as pd

import additional_features_loader as afl
from run_physics_transport_experiment import (
    add_physics_features, add_wind_transport_features, build_cell_adjacency,
    PHYSICS_FEATURE_NAMES, TRANSPORT_FEATURE_NAMES, TRANSPORT_VARIABLES, ADJACENCY_RADIUS_KM,
)

IDENTIFIER_COLUMN = "grid_id"
DATETIME_COLUMN = "datetime_utc"

# Explicit exclusion list -- every fire-related column from Experiment 1,
# named individually rather than pattern-matched, so a future column
# rename can't silently let a fire column back in unnoticed.
FIRE_COLUMNS_TO_EXCLUDE = [
    "hotspot_count", "fire_present", "max_frp", "mean_frp", "sum_frp",
    "hotspot_count_5km", "fire_present_5km", "max_frp_5km",
    "hotspot_count_10km", "fire_present_10km", "max_frp_10km",
    "rolling_24h_hotspot_count", "rolling_72h_hotspot_count", "rolling_7day_hotspot_count",
]

ADJACENCY_CACHE_DIR = Path(__file__).resolve().parent / "physics_transport_adjacency_cache"


def load_non_fire_additional_features(domain: str) -> pd.DataFrame:
    """Loads the full additional-features file (reusing
    additional_features_loader.py's own loader, not a new read path)
    and drops the fire columns explicitly."""
    full_df = afl.load_additional_features(domain)
    present_fire_cols = [c for c in FIRE_COLUMNS_TO_EXCLUDE if c in full_df.columns]
    return full_df.drop(columns=present_fire_cols)


def get_or_build_adjacency(centroids: pd.DataFrame, domain: str) -> dict:
    """Adjacency is O(n_cells^2) and depends only on cell geometry, which
    never changes -- cached to disk per domain so repeated runs (or a
    resumed run after interruption) never recompute it."""
    ADJACENCY_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = ADJACENCY_CACHE_DIR / f"{domain}_adjacency_r{ADJACENCY_RADIUS_KM}km.pkl"
    if cache_path.exists():
        print(f"  Loading cached adjacency table from {cache_path.name}...")
        return pd.read_pickle(cache_path)

    print(f"  Building adjacency table for {domain} ({len(centroids)} cells, radius={ADJACENCY_RADIUS_KM}km)...")
    adjacency = build_cell_adjacency(centroids, ADJACENCY_RADIUS_KM)
    pd.to_pickle(adjacency, cache_path)
    print(f"  Cached to {cache_path.name}.")
    return adjacency


def build_combined_features(base_df: pd.DataFrame, metadata_df: pd.DataFrame, domain: str) -> tuple:
    """base_df: original features (no 'split', no 'datetime_utc').
    metadata_df: must contain datetime_utc, centroid_latitude,
    centroid_longitude, row-aligned with base_df.
    Returns (combined_df, new_feature_column_names)."""
    if len(base_df) != len(metadata_df):
        raise ValueError(f"base_df/metadata_df row count mismatch ({len(base_df)} vs {len(metadata_df)}).")

    keyed_df = base_df.copy()
    keyed_df[DATETIME_COLUMN] = metadata_df[DATETIME_COLUMN].values
    keyed_df["centroid_latitude"] = metadata_df["centroid_latitude"].values
    keyed_df["centroid_longitude"] = metadata_df["centroid_longitude"].values

    # Layer 1: non-fire additional features, via the existing loader's
    # own integrity-checked merge.
    additional_df = load_non_fire_additional_features(domain)
    merged = keyed_df.merge(additional_df, on=[IDENTIFIER_COLUMN, DATETIME_COLUMN], how="left")
    afl.verify_merge_integrity(keyed_df, additional_df, merged)
    non_fire_cols = [c for c in additional_df.columns if c not in (IDENTIFIER_COLUMN, DATETIME_COLUMN)]

    # Layer 2: physics features (pure row-wise derivation, no merge risk).
    merged = add_physics_features(merged)

    # Layer 3: wind-transport features (needs the adjacency table).
    centroids = merged[[IDENTIFIER_COLUMN, "centroid_latitude", "centroid_longitude"]].drop_duplicates(subset=[IDENTIFIER_COLUMN])
    adjacency = get_or_build_adjacency(centroids, domain)
    merged = add_wind_transport_features(merged, adjacency)

    new_feature_cols = non_fire_cols + PHYSICS_FEATURE_NAMES + TRANSPORT_FEATURE_NAMES
    merged = merged.drop(columns=["centroid_latitude", "centroid_longitude"])
    return merged, new_feature_cols
