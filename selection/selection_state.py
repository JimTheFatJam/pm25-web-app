"""
selection/selection_state.py -- Frontend application state.

Deliberately a plain dataclass with no coupling to model inference,
research-pipeline modules, or any backend call. These fields are exactly
what would eventually form a backend prediction request (see
selection/request_validator.py and the README's API schema section).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import geopandas as gpd


@dataclass
class SelectionState:
    selected_polygon: Optional[object] = None  # shapely Polygon, in DISPLAY_CRS (EPSG:4326)
    selected_cells: Optional[gpd.GeoDataFrame] = None  # subset of the grid, in the grid's analysis CRS
    selected_cell_ids: list = field(default_factory=list)
    selected_cell_count: int = 0
    selected_area_km2: float = 0.0

    area_outside_domain_excluded: bool = False  # True if clip_polygon_to_boundary trimmed anything

    prediction_mode: str = "single_date"  # "single_date" | "date_range"
    start_datetime: Optional[datetime] = None  # Philippine local time (see config.settings.LOCAL_TIMEZONE)
    end_datetime: Optional[datetime] = None
    temporal_resolution: str = "hourly"  # "hourly" | "daily" -- kept as a string, not hardcoded per-control

    @property
    def estimated_prediction_rows(self) -> int:
        """cells x timestamps, per the documented future row-limit
        formula -- a pure frontend estimate, never sent to any model."""
        if self.selected_cell_count == 0 or self.start_datetime is None:
            return 0
        n_timestamps = self._count_timestamps()
        return self.selected_cell_count * n_timestamps

    def _count_timestamps(self) -> int:
        if self.start_datetime is None:
            return 0
        end = self.end_datetime or self.start_datetime
        delta_hours = max(0, (end - self.start_datetime).total_seconds() / 3600)
        if self.temporal_resolution == "hourly":
            return int(delta_hours) + 1
        elif self.temporal_resolution == "daily":
            return int(delta_hours / 24) + 1
        return 1

    def clear_selection(self):
        self.selected_polygon = None
        self.selected_cells = None
        self.selected_cell_ids = []
        self.selected_cell_count = 0
        self.selected_area_km2 = 0.0
        self.area_outside_domain_excluded = False

    def to_backend_request_preview(self) -> dict:
        """Exactly the shape documented in the README's future API
        schema -- built here once so the UI and any future real request
        builder share a single source of truth for the field names."""
        return {
            "cell_ids": self.selected_cell_ids,
            "start_datetime": self.start_datetime.isoformat() if self.start_datetime else None,
            "end_datetime": (self.end_datetime or self.start_datetime).isoformat() if self.start_datetime else None,
            "temporal_resolution": self.temporal_resolution,
        }
