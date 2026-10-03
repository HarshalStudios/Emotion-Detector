"""Models module for facial emotion recognition."""

from src.models.spatial_baseline import (
    SpatialBaseline,
    create_spatial_baseline,
    SUPPORTED_BACKBONES,
)
from src.models.spatial_frequency import (
    SpatialFrequencyModel,
    create_spatial_frequency_model,
    LearnableSpectralFilter,
    FrequencyBranch,
)
from src.models.spatial_frequency_geometry import (
    SpatialFrequencyGeometryModel,
    create_spatial_frequency_geometry_model,
    GeometryBranch,
)
from src.models.spatial_frequency_geometry_gated import (
    SpatialFrequencyGeometryGatedModel,
    create_spatial_frequency_geometry_gated_model,
    LearnedGatedFusion,
)

__all__ = [
    "SpatialBaseline",
    "create_spatial_baseline",
    "SUPPORTED_BACKBONES",
    "SpatialFrequencyModel",
    "create_spatial_frequency_model",
    "LearnableSpectralFilter",
    "FrequencyBranch",
    "SpatialFrequencyGeometryModel",
    "create_spatial_frequency_geometry_model",
    "GeometryBranch",
    "SpatialFrequencyGeometryGatedModel",
    "create_spatial_frequency_geometry_gated_model",
    "LearnedGatedFusion",
]
