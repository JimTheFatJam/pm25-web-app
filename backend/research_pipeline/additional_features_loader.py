"""
Additional Features Runtime Loader
========================================

Merges source_additional_features.csv / philippines_additional_features.csv
into the original feature set AT RUNTIME, in memory -- no new combined
CSV is ever written to disk, matching lag_features.py's own philosophy
in this project (compute/merge in memory, never persist a combined
file). The original source_features.csv / philippines_features.csv are
only ever read here, never modified.

Controlled by a single flag (INCLUDE_ADDITIONAL_FEATURES) so any
experiment can toggle the merge on/off without any other code change --
existing experiments that never import this module are entirely
unaffected.

Required integrity checks (run automatically before returning the merged
frame, per explicit instruction: correctness checks, not experimental
diagnostics -- failures STOP immediately with a clear error, they do not
warn and continue):
  - row count unchanged after the merge
  - no duplicated (grid_id, datetime_utc) keys in the additional-features file
  - no missing keys (every original row finds a match)
  - exactly one row per (grid_id, datetime_utc) in the result
"""

from pathlib import Path

import pandas as pd

IDENTIFIER_COLUMN = "grid_id"
DATETIME_COLUMN = "datetime_utc"

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ADDITIONAL_FEATURES_PATH = SCRIPT_DIR / "source_additional_features.csv"
PH_ADDITIONAL_FEATURES_PATH = SCRIPT_DIR / "philippines_additional_features.csv"


def load_additional_features(domain: str) -> pd.DataFrame:
    """domain must be 'source' or 'philippines'."""
    path = SOURCE_ADDITIONAL_FEATURES_PATH if domain == "source" else PH_ADDITIONAL_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run build_additional_features.py for all countries first.")
    return pd.read_csv(path)


def verify_merge_integrity(original_df: pd.DataFrame, additional_df: pd.DataFrame, merged_df: pd.DataFrame):
    """Correctness checks, not experimental diagnostics -- any failure
    STOPS immediately with a clear, specific error rather than a warning,
    per explicit instruction. Each check is verified independently so a
    failure message names exactly which guarantee broke, not just "merge
    failed"."""
    keys = [IDENTIFIER_COLUMN, DATETIME_COLUMN]

    dup_keys = additional_df.duplicated(subset=keys, keep=False)
    if dup_keys.any():
        n_dup = dup_keys.sum()
        example = additional_df.loc[dup_keys, keys].iloc[0].to_dict()
        raise AssertionError(
            f"INTEGRITY CHECK FAILED: {n_dup} duplicated (grid_id, datetime_utc) key(s) found in the additional "
            f"features file (example: {example}). This most likely means a country's rows were appended more "
            f"than once during build_additional_features.py's runs -- check "
            f"source_additional_features_assembled_countries.txt for a country listed twice, or a country's "
            f"rows appended before its assembly was properly tracked.")

    if len(merged_df) != len(original_df):
        raise AssertionError(
            f"INTEGRITY CHECK FAILED: row count changed after merging additional features "
            f"({len(original_df)} original rows -> {len(merged_df)} merged rows). A correct merge must produce "
            f"exactly one output row per original row -- this delta means either missing keys (rows in "
            f"original_df with no match, if merged_df is smaller) or an unexpected key duplication that slipped "
            f"past the check above (if merged_df is larger).")

    # Missing-key check: after a LEFT merge (original_df is authoritative,
    # per the row-count-unchanged requirement above), every original row
    # should have found a match -- a new-feature column that is entirely
    # NaN for a row means that row's key was never found in additional_df.
    new_cols = [c for c in additional_df.columns if c not in keys]
    if new_cols:
        missing_mask = merged_df[new_cols].isna().all(axis=1)
        if missing_mask.any():
            n_missing = missing_mask.sum()
            example_key = merged_df.loc[missing_mask, keys].iloc[0].to_dict()
            raise AssertionError(
                f"INTEGRITY CHECK FAILED: {n_missing} row(s) in the original features have NO matching key in "
                f"the additional features file (example: {example_key}). This means additional-feature "
                f"extraction is incomplete for at least one country/cell -- check "
                f"source_additional_features_assembled_countries.txt against the full country list before "
                f"proceeding, rather than training on a partially-covered feature set.")

    dup_merged = merged_df.duplicated(subset=keys, keep=False)
    if dup_merged.any():
        raise AssertionError(
            f"INTEGRITY CHECK FAILED: {dup_merged.sum()} duplicated (grid_id, datetime_utc) key(s) in the final "
            f"merged output -- expected exactly one row per key.")

    print(f"  Merge integrity checks PASSED: {len(merged_df)} rows, one row per (grid_id, datetime_utc), "
          f"no duplicates, no missing keys.")


def load_and_merge_features(original_df: pd.DataFrame, metadata_df: pd.DataFrame, domain: str) -> tuple:
    """original_df must NOT already contain 'datetime_utc' (attach it
    from metadata_df first if needed) -- this function attaches it
    internally to build merge keys, then returns a DataFrame with the
    SAME row order/index as original_df, plus the new feature columns.
    Returns (merged_df, new_feature_column_names)."""
    if len(original_df) != len(metadata_df):
        raise ValueError(f"original_df/metadata_df row count mismatch ({len(original_df)} vs {len(metadata_df)}) "
                          f"-- cannot safely attach datetime_utc positionally.")

    keyed_original = original_df.copy()
    keyed_original[DATETIME_COLUMN] = metadata_df[DATETIME_COLUMN].values

    additional_df = load_additional_features(domain)
    new_feature_cols = [c for c in additional_df.columns if c not in (IDENTIFIER_COLUMN, DATETIME_COLUMN)]

    merged_df = keyed_original.merge(additional_df, on=[IDENTIFIER_COLUMN, DATETIME_COLUMN], how="left")
    verify_merge_integrity(keyed_original, additional_df, merged_df)

    return merged_df, new_feature_cols
