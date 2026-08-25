"""Planar trajectory representation and geometry."""

from .bspline import (
    ARC_LENGTH_OVERSAMPLE_FACTOR,
    BSPLINE_BENDING_REGULARIZATION_M4,
    PlanarBSplineCodec,
)
from .metrics import path_scale_summary
from .normalization import PlanarScaleNormalizer
from .resampling import path_arc_length, resample_path_by_arc_length

__all__ = [
    "PlanarBSplineCodec",
    "ARC_LENGTH_OVERSAMPLE_FACTOR",
    "BSPLINE_BENDING_REGULARIZATION_M4",
    "PlanarScaleNormalizer",
    "path_arc_length",
    "path_scale_summary",
    "resample_path_by_arc_length",
]
