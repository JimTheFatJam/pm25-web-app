"""
Validation Strategy Split Generators
==========================================

Two genuinely new split methodologies, precisely defined per the
critical review preceding this implementation:

random_holdout_split(): ROW-LEVEL random assignment, deliberately
    ignoring cell boundaries -- the same cell's rows can land in both
    fine-tune-train and the evaluation set. This is intentional: it
    exists specifically as a literature-comparison artifact, illustrating
    how much of the PM2.5 literature's typically higher reported R2
    reflects an easier task (interpolating at partially-seen locations)
    rather than genuinely better modeling. NEVER cite this number as "the
    model's performance" without that caveat.

kfold_cell_splits(): CELL-LEVEL k-fold -- each fold's held-out cells are
    entirely unseen (matching the spatial holdout's own rigor), with the
    remaining (k-1)/k cells further divided into fine-tune-train/
    fine-tune-val using the EXISTING, already-validated stratified
    split_cells() logic from run_transfer_learning_experiment.py (same
    3:1-ish ratio, same land-cover stratification, same singleton-class
    rule) -- not a new allocation rule invented for this experiment.

Both split types still carry the original grid_id on every row, so any
statistical test performed on top of either split (including random
holdout) can still aggregate to per-cell mean error before testing --
required specifically for random holdout, where the split itself ignores
cell boundaries but the STATISTICAL TEST must not, to avoid
reintroducing the row-level autocorrelation problem this project has
been careful to avoid everywhere else.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_transfer_learning_experiment import build_cell_level_table, split_cells

IDENTIFIER_COLUMN = "grid_id"


def random_holdout_split(ph_df: pd.DataFrame, train_frac: float = 0.45, val_frac: float = 0.15,
                          seed: int = 42) -> pd.DataFrame:
    """Row-level random split -- returns a DataFrame with one row per
    original ph_df row, plus a 'split' column. Deliberately ignores cell
    boundaries (see module docstring)."""
    rng = np.random.RandomState(seed)
    n = len(ph_df)
    permuted_idx = rng.permutation(n)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    split_labels = np.empty(n, dtype=object)
    split_labels[permuted_idx[:n_train]] = "finetune_train"
    split_labels[permuted_idx[n_train:n_train + n_val]] = "finetune_val"
    split_labels[permuted_idx[n_train + n_val:]] = "holdout"

    result = ph_df[[IDENTIFIER_COLUMN]].copy()
    result["split"] = split_labels
    return result


def kfold_cell_splits(ph_df: pd.DataFrame, k: int, seed: int = 42) -> list:
    """Cell-level k-fold. Returns a list of k DataFrames, each with one
    row per grid_id and a 'split' column (finetune_train/finetune_val/
    holdout) for that fold. Fold i's held-out cells are drawn from a
    KFold partition of all cells; the REMAINING cells for that fold are
    divided into finetune_train/finetune_val using the existing,
    already-validated split_cells() stratified logic (roughly 3:1,
    matching the original 45:15 spatial-holdout ratio)."""
    lc_cols = sorted(c for c in ph_df.columns if c.startswith("landcover_local_"))
    cell_table = build_cell_level_table(ph_df, lc_cols)
    all_cell_ids = cell_table[IDENTIFIER_COLUMN].values

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_splits = []

    for fold_idx, (remaining_idx, holdout_idx) in enumerate(kf.split(all_cell_ids)):
        holdout_cells = set(all_cell_ids[holdout_idx])
        remaining_cell_table = cell_table[cell_table[IDENTIFIER_COLUMN].isin(all_cell_ids[remaining_idx])]

        # Same stratified logic already validated for the spatial holdout
        # split -- train_frac/val_frac given as fractions OF THE REMAINING
        # cells (roughly 3:1), not of the full cell pool.
        remaining_assignment = split_cells(remaining_cell_table, train_frac=0.75, val_frac=0.25, seed=seed + fold_idx)
        remaining_assignment = remaining_assignment.rename(columns={"split": "split"})[[IDENTIFIER_COLUMN, "split"]]

        holdout_assignment = pd.DataFrame({IDENTIFIER_COLUMN: sorted(holdout_cells), "split": "holdout"})
        fold_assignment = pd.concat([remaining_assignment, holdout_assignment], ignore_index=True)

        assert set(fold_assignment[IDENTIFIER_COLUMN]) == set(all_cell_ids), \
            f"Fold {fold_idx}: cell coverage mismatch -- some cells missing from the fold assignment."
        assert fold_assignment[IDENTIFIER_COLUMN].duplicated().sum() == 0, \
            f"Fold {fold_idx}: a cell was assigned to more than one split."

        fold_splits.append(fold_assignment)

    return fold_splits
