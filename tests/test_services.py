"""Service-level tests with mocked embeddings and LLM output."""

from app.models.schemas import TestCase
from app.services.requirement_service import sanitize_requirement
from app.services.testcase_service import deduplicate_test_cases
from app.core.exceptions import InvalidRequirementError
import pytest


def _case(title: str, expected: str) -> TestCase:
    return TestCase(
        id="TC-001",
        title=title,
        category="positive",
        priority="high",
        steps=["Step 1"],
        expected_result=expected,
    )


def test_sanitize_rejects_whitespace() -> None:
    with pytest.raises(InvalidRequirementError):
        sanitize_requirement("   ")


def test_sanitize_rejects_overlong_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", "")
    settings = get_settings()
    too_long = "a" * (settings.application.max_requirement_length + 1)
    with pytest.raises(InvalidRequirementError):
        sanitize_requirement(too_long)


def test_deduplication_removes_similar_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_similarity(left: str, right: str) -> float:
        if left == right:
            return 1.0
        combined = f"{left} {right}".lower()
        if "valid pdf" in combined and "oversized" not in combined:
            return 0.95
        return 0.2

    monkeypatch.setattr("app.services.testcase_service.semantic_similarity", fake_similarity)
    cases = [
        _case("Upload valid PDF", "File is uploaded."),
        _case("Upload a valid PDF file", "The file is uploaded."),
        _case("Reject oversized PDF", "Upload is rejected."),
    ]
    unique = deduplicate_test_cases(cases, threshold=0.85)
    assert len(unique) == 2
