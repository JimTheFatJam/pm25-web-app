"""
selection/request_validator.py -- Prediction-request validation.

NOT enforcing any real limit yet, per explicit instruction -- this
module exists so the eventual MAX_PREDICTION_ROWS enforcement has a
single, clearly-separated place to live without requiring any change to
the map-selection or UI code that calls this function.
"""

from dataclasses import dataclass

from config.settings import MAX_PREDICTION_ROWS


@dataclass
class ValidationResult:
    is_valid: bool
    message: str


def validate_prediction_request(selected_cells, start_datetime, end_datetime, temporal_resolution) -> ValidationResult:
    """Signature matches the one specified for this prototype. Currently
    always returns VALID (with an informational message) -- the actual
    row-limit check is intentionally not implemented here yet, since
    MAX_PREDICTION_ROWS has no real value defined until the backend
    specifies it (see config/settings.py)."""
    if selected_cells is None or len(selected_cells) == 0:
        return ValidationResult(False, "No prediction cells were selected.")
    if start_datetime is None:
        return ValidationResult(False, "Please select a prediction date before continuing.")

    if MAX_PREDICTION_ROWS is None:
        return ValidationResult(
            True,
            "Prediction request limits will be applied when the backend service is connected."
        )

    # Reachable once MAX_PREDICTION_ROWS is actually set by a future
    # backend contract -- included now so the enforcement path exists
    # and is exercised by nothing else needing to change.
    n_timestamps = _estimate_timestamp_count(start_datetime, end_datetime, temporal_resolution)
    estimated_rows = len(selected_cells) * n_timestamps
    if estimated_rows > MAX_PREDICTION_ROWS:
        return ValidationResult(
            False,
            f"This request would produce approximately {estimated_rows:,} prediction rows, "
            f"exceeding the maximum of {MAX_PREDICTION_ROWS:,}. Please reduce the selected area or time range."
        )
    return ValidationResult(True, f"Request evaluated against the {MAX_PREDICTION_ROWS:,}-row limit: within bounds.")


def _estimate_timestamp_count(start_datetime, end_datetime, temporal_resolution) -> int:
    end = end_datetime or start_datetime
    delta_hours = max(0, (end - start_datetime).total_seconds() / 3600)
    if temporal_resolution == "hourly":
        return int(delta_hours) + 1
    elif temporal_resolution == "daily":
        return int(delta_hours / 24) + 1
    return 1
