"""Automated tests for NLP helpers."""

from app.services.nlp_service import analyze_requirement, detect_ambiguities, semantic_similarity


def test_detects_quickly_as_ambiguous() -> None:
    ambiguities = detect_ambiguities("The system should upload files quickly.")
    phrases = {item.phrase for item in ambiguities}
    assert "quickly" in phrases
    assert any(item.severity == "high" for item in ambiguities)


def test_preprocessing_preserves_original_and_cleans_whitespace() -> None:
    original = "  Authenticated users can upload PDF files up to 50 MB.  "
    result = analyze_requirement(original)
    assert result.original_text == original
    assert result.processed_text == "Authenticated users can upload PDF files up to 50 MB."
    assert result.extracted.actions
    assert any("50" in value for value in result.extracted.quantitative_values)


def test_completeness_flags_missing_file_size_and_error_behavior() -> None:
    result = analyze_requirement("The website should allow registered users to upload PDF files quickly.")
    assert result.completeness_score < 90
    joined = " ".join(result.missing_information).lower()
    assert "file size" in joined or "maximum" in joined
    assert result.quality_score <= 100


def test_similarity_detects_near_duplicates() -> None:
    score = semantic_similarity(
        "Upload a valid PDF below 50 MB",
        "Upload a valid PDF below 50 MB",
    )
    assert score > 0.99
