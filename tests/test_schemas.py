"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from app.models.schemas import TestCase


def test_valid_test_case() -> None:
    case = TestCase(
        id="TC-001",
        title="Upload valid PDF",
        description="Happy path",
        category="Positive",
        priority="High",
        preconditions=["User is authenticated"],
        steps=["Open upload page", "Select a PDF", "Click Upload"],
        expected_result="The PDF is uploaded.",
    )
    assert case.category == "positive"
    assert case.priority == "high"


def test_edge_case_category_alias() -> None:
    case = TestCase(
        id="TC-002",
        title="Empty file",
        category="edge case",
        priority="medium",
        steps=["Upload an empty PDF"],
        expected_result="The upload is rejected.",
    )
    assert case.category == "edge"


def test_rejects_empty_steps() -> None:
    with pytest.raises(ValidationError):
        TestCase(
            id="TC-003",
            title="Invalid",
            category="functional",
            priority="low",
            steps=[" "],
            expected_result="Something happens.",
        )


def test_rejects_empty_expected_result() -> None:
    with pytest.raises(ValidationError):
        TestCase(
            id="TC-004",
            title="Invalid",
            category="functional",
            priority="low",
            steps=["Do a thing"],
            expected_result="   ",
        )
