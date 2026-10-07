"""REST API routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import (
    AnalysisResponse,
    GenerateResponse,
    HealthResponse,
    ImproveResponse,
    RequirementRequest,
)
from app.services.requirement_service import analyze, improve
from app.services.testcase_service import generate_test_cases

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        application=settings.application.name,
        version=settings.application.version,
    )


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_endpoint(payload: RequirementRequest) -> AnalysisResponse:
    return analyze(payload.requirement)


@router.post("/improve", response_model=ImproveResponse)
def improve_endpoint(payload: RequirementRequest) -> ImproveResponse:
    return improve(payload.requirement)


@router.post("/generate", response_model=GenerateResponse)
def generate_endpoint(payload: RequirementRequest) -> GenerateResponse:
    return generate_test_cases(payload.requirement)
