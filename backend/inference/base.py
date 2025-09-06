"""Pose extraction base interfaces and shared helpers (SOLID-friendly).

- Single Responsibility: contains only abstractions and tiny, pure helpers.
- Open/Closed: new extractors implement `PoseExtractor` without client changes.
- Liskov: concrete extractors interchangeable via the interface.
- Interface Segregation: minimal, focused interface surface.
- Dependency Inversion: clients depend on `PoseExtractor` abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


# Shared defaults/utilities
TARGET_SIZE: Tuple[int, int] = (1280, 720)
BASE_RESOLUTION: int = 720


def normalize_coords(keypoints, width: int, height: int):
    return [(kp[0] / width, kp[1] / height, kp[2]) for kp in keypoints]


def scale_score(score: float, height: int, base: int = BASE_RESOLUTION) -> float:
    return score * (base / max(height, 1))


class PoseExtractor(ABC):
    @abstractmethod
    def extract(self, video_path, *, target_size: tuple[int, int] | None = TARGET_SIZE):
        """Return a sequence of normalized keypoints per frame."""
        raise NotImplementedError


__all__ = [
    "PoseExtractor",
    "TARGET_SIZE",
    "BASE_RESOLUTION",
    "normalize_coords",
    "scale_score",
]

