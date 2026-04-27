"""
schemas.py — Pydantic models for MicroPlastiNet M5 GenAI Report Generator
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PolymerData(BaseModel):
    """Polymer classification output from M2b spectral classifier."""
    station_id: str
    polymers: dict[str, float]         # polymer → fraction (sums to ~1)
    confidence: dict[str, float]       # polymer → classifier confidence
    total_particles: int

    @property
    def top_polymer(self) -> tuple[str, float]:
        best = max(self.polymers, key=self.polymers.get)
        return best, self.polymers[best]

    @property
    def dominant_polymers(self) -> list[tuple[str, float]]:
        """Return polymers sorted by fraction, descending."""
        return sorted(self.polymers.items(), key=lambda x: x[1], reverse=True)


class SourceEntry(BaseModel):
    """A single candidate source with attribution probability."""
    rank: int
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float  = Field(ge=0.0, le=1.0)
    distance_km: float
    lat: float
    lon: float


class AttributionData(BaseModel):
    """GNN-derived source attribution for a contamination event."""
    station_id: str
    event_id: str
    event_date: str
    sources: list[SourceEntry]

    @property
    def top_source(self) -> SourceEntry:
        return self.sources[0]

    @property
    def summary_line(self) -> str:
        top = self.top_source
        return (
            f"{top.name} (probability {top.probability*100:.0f}%, "
            f"confidence {top.confidence*100:.0f}%, distance {top.distance_km:.1f} km)"
        )


class ReportInput(BaseModel):
    """Full input bundle for the report generator."""
    station_id: str
    event_data: PolymerData
    attribution_data: AttributionData
    generated_at: datetime = Field(default_factory=datetime.now)
    report_mode: str = "template"   # "template" | "openai"


class ReportSection(BaseModel):
    """A single section in the generated report."""
    title: str
    content: str


class ReportOutput(BaseModel):
    """Generated regulator report."""
    station_id: str
    event_id: str
    generated_at: datetime
    mode: str
    sections: list[ReportSection]
    full_text: str

    def to_markdown(self) -> str:
        lines = [f"# MicroPlastiNet Environmental Compliance Report\n"]
        lines.append(f"**Station:** {self.station_id}  |  "
                     f"**Event:** {self.event_id}  |  "
                     f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}\n")
        lines.append("---\n")
        for sec in self.sections:
            lines.append(f"## {sec.title}\n")
            lines.append(f"{sec.content}\n")
        return "\n".join(lines)
