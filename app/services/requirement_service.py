"""Input validation and requirement analysis orchestration."""

from __future__ import annotations

from pydantic import ValidationError

from app.core.config import get_settings
from app.core.exceptions import InvalidRequirementError, LLMOutputError
from app.core.logging import logger
from app.models.schemas import (
    AnalysisResponse,
    Ambiguity,
    ExtractedComponents,
    ImproveResponse,
    LLMAnalysisPayload,
    LLMImprovePayload,
)
from app.services.llm_service import generate_json
from app.services.nlp_service import NLPResult, analyze_requirement, nlp_context_blob
from app.services.prompt_loader import render_prompt


def sanitize_requirement(text: str | None) -> str:
    settings = get_settings()
    if text is None or not str(text).strip():
        raise InvalidRequirementError("Requirement cannot be empty.")
    cleaned = str(text).replace("\x00", " ").strip()
    if not cleaned:
        raise InvalidRequirementError("Requirement cannot be empty.")
    max_len = settings.application.max_requirement_length
    if len(cleaned) > max_len:
        raise InvalidRequirementError(
            f"Requirement exceeds the maximum length of {max_len} characters."
        )
    return cleaned


def _merge_lists(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item.strip())
    return merged


def _merge_extracted(base: ExtractedComponents, extra: ExtractedComponents | None) -> ExtractedComponents:
    if extra is None:
        return base
    data = {}
    for field in ExtractedComponents.model_fields:
        data[field] = _merge_lists(getattr(base, field), getattr(extra, field))
    return ExtractedComponents(**data)


def _merge_ambiguities(base: list[Ambiguity], extra: list[Ambiguity]) -> list[Ambiguity]:
    seen: set[str] = set()
    merged: list[Ambiguity] = []
    for item in base + extra:
        key = item.phrase.strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def analyze(requirement: str, use_llm: bool = True) -> AnalysisResponse:
    original = sanitize_requirement(requirement)
    logger.info("Requirement received")
    nlp = analyze_requirement(original)
    assumptions: list[str] = []
    extracted = nlp.extracted
    ambiguities = nlp.ambiguities
    missing = nlp.missing_information
    completeness = nlp.completeness_score

    settings = get_settings()
    should_enrich = (
        use_llm
        and settings.llm.enrich_analysis
        and bool(settings.llm_api_key)
        and (nlp.ambiguity_score >= 16 or nlp.completeness_score < 75)
    )
    if should_enrich:
        try:
            llm_data = generate_json(
                render_prompt(
                    "analyze_requirement.txt",
                    requirement=original,
                    nlp_context=nlp_context_blob(nlp),
                ),
                LLMAnalysisPayload,
            )
        except ValidationError as exc:
            raise LLMOutputError("LLM analysis payload failed schema validation.") from exc
        extracted = _merge_extracted(extracted, llm_data.extracted)
        ambiguities = _merge_ambiguities(ambiguities, llm_data.ambiguities)
        missing = _merge_lists(missing, llm_data.missing_information)
        assumptions = llm_data.assumptions
        if llm_data.completeness_score is not None:
            completeness = round((completeness + llm_data.completeness_score) / 2)

    return AnalysisResponse(
        original_requirement=nlp.original_text,
        processed_text=nlp.processed_text,
        quality_score=nlp.quality_score,
        ambiguity_score=nlp.ambiguity_score,
        completeness_score=completeness,
        extracted=extracted,
        ambiguities=ambiguities,
        missing_information=missing,
        assumptions=assumptions,
    )


def improve(requirement: str) -> ImproveResponse:
    original = sanitize_requirement(requirement)
    nlp = analyze_requirement(original)
    try:
        data = generate_json(
            render_prompt(
                "improve_requirement.txt",
                requirement=original,
                nlp_context=nlp_context_blob(nlp),
            ),
            LLMImprovePayload,
        )
    except ValidationError as exc:
        raise LLMOutputError("LLM improvement payload failed schema validation.") from exc
    return ImproveResponse(
        original_requirement=original,
        improved_requirement=data.improved_requirement.strip(),
        assumptions=data.assumptions,
    )


def nlp_only(requirement: str) -> NLPResult:
    return analyze_requirement(sanitize_requirement(requirement))
