"""Analysis package: pose comparison and metrics.

Exposes high-level swing analysis utilities.
"""

from .swing_compare import (
    compare_swings,
    analyze_differences,
    GolfSwingAnalyzer,
    draw_skeleton,
)

__all__ = [
    "compare_swings",
    "analyze_differences",
    "GolfSwingAnalyzer",
    "draw_skeleton",
]

