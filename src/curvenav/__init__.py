"""CurveNav: Curvature-aware Local Navigation.

Model symbols are imported lazily so the CPU-only expert manifest builder can
run without importing PyTorch or initializing a CUDA runtime.
"""

__all__ = [
    "CurveNavConfig",
    "PolicyCondition",
    "TrajectoryPrediction",
    "TrajectoryTarget",
    "build_policy",
]


def __getattr__(name: str):
    if name == "CurveNavConfig":
        from .config import CurveNavConfig

        return CurveNavConfig
    if name == "build_policy":
        from .factory import build_policy

        return build_policy
    if name in {"PolicyCondition", "TrajectoryPrediction", "TrajectoryTarget"}:
        from .types import PolicyCondition, TrajectoryPrediction, TrajectoryTarget

        return {
            "PolicyCondition": PolicyCondition,
            "TrajectoryPrediction": TrajectoryPrediction,
            "TrajectoryTarget": TrajectoryTarget,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
