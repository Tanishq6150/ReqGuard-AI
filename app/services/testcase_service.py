"""Test-case generation, validation, and deduplication."""

from __future__ import annotations

from pydantic import ValidationError

from app.core.config import get_settings
from app.core.exceptions import LLMOutputError
from app.core.logging import logger
from app.models.schemas import GenerateResponse, LLMGeneratePayload, TestCase
from app.services.llm_service import generate_json
from app.services.nlp_service import nlp_context_blob, semantic_similarity
from app.services.prompt_loader import render_prompt
from app.services.requirement_service import nlp_only, sanitize_requirement


def _case_text(case: TestCase) -> str:
    return " ".join(
        [
            case.title,
            case.description,
            " ".join(case.steps),
            case.expected_result,
            case.category,
        ]
    )


def deduplicate_test_cases(cases: list[TestCase], threshold: float | None = None) -> list[TestCase]:
    settings = get_settings()
    cutoff = settings.nlp.similarity_threshold if threshold is None else threshold
    unique: list[TestCase] = []
    for case in cases:
        duplicate = False
        candidate = _case_text(case)
        for existing in unique:
            if semantic_similarity(candidate, _case_text(existing)) >= cutoff:
                duplicate = True
                break
        if not duplicate:
            unique.append(case)
    return unique


def _renumber(cases: list[TestCase]) -> list[TestCase]:
    updated: list[TestCase] = []
    for index, case in enumerate(cases, start=1):
        updated.append(case.model_copy(update={"id": f"TC-{index:03d}"}))
    return updated


def generate_test_cases(requirement: str) -> GenerateResponse:
    original = sanitize_requirement(requirement)
    nlp = nlp_only(original)
    settings = get_settings()
    try:
        data = generate_json(
            render_prompt(
                "generate_testcases.txt",
                requirement=original,
                nlp_context=nlp_context_blob(nlp),
                max_test_cases=settings.generation.max_test_cases,
                include_positive=settings.generation.include_positive,
                include_negative=settings.generation.include_negative,
                include_boundary=settings.generation.include_boundary,
                include_edge_cases=settings.generation.include_edge_cases,
                include_security=settings.generation.include_security,
            ),
            LLMGeneratePayload,
        )
    except ValidationError as exc:
        raise LLMOutputError("LLM test-case payload failed schema validation.") from exc

    if not data.test_cases:
        raise LLMOutputError("LLM did not return any test cases.")

    limited = data.test_cases[: settings.generation.max_test_cases]
    unique = deduplicate_test_cases(limited)
    numbered = _renumber(unique)
    logger.info("Test case generation completed")
    return GenerateResponse(original_requirement=original, test_cases=numbered)
