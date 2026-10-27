from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ObjectTag(BaseModel):
    label: str
    confidence: Optional[float] = None
    source: Literal["model", "detector", "fusion"] = "model"


class DescribeRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded RGB image.")
    model: Optional[str] = Field(None, description="Specific model slug to run.")
    composition: Optional[List[str]] = Field(None, description="Ordered list of model slugs to ensemble.")
    include_detections: bool = Field(True, description="Run detector to generate context.")
    use_cache: bool = Field(True, description="Return cached result when available.")
    timeout_s: Optional[float] = Field(None, description="Override per-request timeout.")


class DescribeResult(BaseModel):
    model: str
    caption: str
    objects: List[ObjectTag]
    latency_ms: float
    cached: bool = False
    device: Optional[str] = None
    diagnostics: Dict[str, float] = Field(default_factory=dict)


class DescribeResponse(BaseModel):
    results: List[DescribeResult]
    aggregate_caption: str
    aggregate_objects: List[ObjectTag]


class MatchCriteriaRequest(BaseModel):
    image_b64: str
    criteria: List[str]
    model: Optional[str] = None
    include_detections: bool = True
    use_cache: bool = True
    timeout_s: Optional[float] = None


class CriteriaVerdict(BaseModel):
    verdict: Dict[str, bool]
    reasoning: str
    model: str
    latency_ms: float
    cached: bool = False
    device: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    models_loaded: Dict[str, bool]

