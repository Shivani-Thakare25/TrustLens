"""Evidence signal schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

DIRECTIONS = ("increases_risk", "decreases_risk", "neutral", "unavailable")


@dataclass
class Signal:
    signal_name: str
    category: str
    signal_value: Optional[float]
    normalized_score: Optional[float]  # 0 (benign) .. 1 (strong risk evidence); None if unavailable
    direction: str
    explanation: str
    evidence_strength: str  # none | low | medium | high
    availability: bool

    def to_dict(self) -> dict:
        return asdict(self)
