from dataclasses import dataclass, field
from typing import List

from detector import FaceResult


@dataclass
class BBoxResult:
    """Detection result for a single face bounding box in normalized coords."""
    x: float
    y: float
    w: float
    h: float
    label: str = "BOX"


@dataclass
class DetectionResult:
    """Detector output normalized for the rest of the app."""
    faces: List[FaceResult] = field(default_factory=list)
    bboxes: List[BBoxResult] = field(default_factory=list)
    source: str = "LOST"

    @property
    def found(self) -> bool:
        return bool(self.faces or self.bboxes)
