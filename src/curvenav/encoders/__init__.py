"""Calibrated depth and metric configuration-space encoders."""

from .depth import DepthObservationEncoder
from .geometry import MetricDepthProjector

__all__ = [
    "DepthObservationEncoder",
    "MetricDepthProjector",
]
