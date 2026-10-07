"""Request and response schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Severity = Literal["low", "medium", "high"]
TestCategory = Literal["functional", "positive", "negative", "boundary", "edge", "security"]
Priority = Literal["critical", "high", "medium", "low"]


class RequirementRequest(BaseModel):
    requirement: str = Field(..., min_length=1)

    @field_validator("requirement")
    @classmethod
    def requirement_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Requirement cannot be empty.")
        from app.core.config import get_settings

        max_len = get_settings().application.max_requirement_length
        if len(cleaned) > max_len:
            raise ValueError(f"Requirement exceeds the maximum length of {max_len} characters.")
        return value


class Ambiguity(BaseModel):
    phrase: str
    reason: str
    severity: Severity
    suggested_clarification: str | None = None


class ExtractedComponents(BaseModel):
    actors: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    quantitative_values: list[str] = Field(default_factory=list)
    time_limits: list[str] = Field(default_factory=list)
    security_related: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    id: str
    title: str
    description: str = ""
    category: TestCategory
    priority: Priority
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(min_length=1)
    test_data: Any = None
    expected_result: str = Field(min_length=1)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        mapping = {
            "edge case": "edge",
            "edge_case": "edge",
            "edge-case": "edge",
        }
        normalized = str(value).strip().lower()
        return mapping.get(normalized, normalized)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: str) -> str:
        return str(value).strip().lower()

    @field_validator("steps")
    @classmethod
    def steps_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("Test steps cannot be empty.")
        return cleaned

    @field_validator("expected_result")
    @classmethod
    def expected_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Expected result cannot be empty.")
        return cleaned


class AnalysisResponse(BaseModel):
    status: str = "success"
    original_requirement: str
    processed_text: str
    quality_score: int = Field(ge=0, le=100)
    ambiguity_score: int = Field(ge=0, le=100)
    completeness_score: int = Field(ge=0, le=100)
    extracted: ExtractedComponents
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImproveResponse(BaseModel):
    status: str = "success"
    original_requirement: str
    improved_requirement: str
    assumptions: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    status: str = "success"
    original_requirement: str
    test_cases: list[TestCase]
    disclaimer: str = (
        "AI-generated test cases should be reviewed by a qualified software tester "
        "before being used in production."
    )


class HealthResponse(BaseModel):
    status: str = "healthy"
    application: str
    version: str


class ErrorResponse(BaseModel):
    error: str


class LLMAnalysisPayload(BaseModel):
    requirement_summary: str | None = None
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    completeness_score: int | None = None
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    extracted: ExtractedComponents | None = None


class LLMImprovePayload(BaseModel):
    improved_requirement: str
    assumptions: list[str] = Field(default_factory=list)


class LLMGeneratePayload(BaseModel):
    requirement_summary: str | None = None
    test_cases: list[TestCase]
