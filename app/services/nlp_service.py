"""Deterministic NLP analysis for software requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import NLPProcessingError
from app.core.logging import logger
from app.models.schemas import Ambiguity, ExtractedComponents

AMBIGUOUS_TERMS: dict[str, tuple[str, str, str]] = {
    "quickly": (
        "high",
        "No measurable response time has been defined.",
        "Specify a maximum response time in seconds.",
    ),
    "efficiently": (
        "medium",
        "Efficiency is subjective and not measurable.",
        "Define throughput, latency, or resource limits.",
    ),
    "securely": (
        "high",
        "Security is not specified in measurable controls.",
        "State authentication, encryption, or access-control rules.",
    ),
    "user-friendly": (
        "medium",
        "Usability is subjective.",
        "Define concrete UX criteria or acceptance checks.",
    ),
    "user friendly": (
        "medium",
        "Usability is subjective.",
        "Define concrete UX criteria or acceptance checks.",
    ),
    "appropriate": (
        "medium",
        "The expected appropriate behavior is unspecified.",
        "Describe the exact accepted behavior or values.",
    ),
    "large": (
        "medium",
        "Size is not quantified.",
        "Provide an explicit size, count, or limit.",
    ),
    "small": (
        "medium",
        "Size is not quantified.",
        "Provide an explicit size, count, or limit.",
    ),
    "soon": (
        "high",
        "No measurable time bound is defined.",
        "Specify an exact deadline or duration.",
    ),
    "easy": (
        "medium",
        "Ease of use is subjective.",
        "Define measurable usability or operation steps.",
    ),
    "fast": (
        "high",
        "No measurable performance target is defined.",
        "Specify latency, throughput, or SLA values.",
    ),
    "regularly": (
        "medium",
        "Frequency is unspecified.",
        "State how often the action must occur.",
    ),
    "several": (
        "medium",
        "Quantity is vague.",
        "Provide an exact number or range.",
    ),
    "some": (
        "low",
        "Quantity or selection is vague.",
        "Identify which items or how many.",
    ),
    "as needed": (
        "medium",
        "Trigger conditions are unspecified.",
        "Define when the behavior must occur.",
    ),
    "if necessary": (
        "medium",
        "Necessity criteria are unspecified.",
        "Define the conditions that trigger the behavior.",
    ),
    "etc": (
        "low",
        "The list of items is open-ended.",
        "Enumerate all supported values.",
    ),
    "should": (
        "low",
        "Obligation is weaker than a shall/must requirement.",
        "Use shall/must if the behavior is mandatory.",
    ),
    "might": (
        "medium",
        "The behavior is optional or uncertain.",
        "State whether the behavior is required.",
    ),
    "may": (
        "low",
        "The behavior may be interpreted as optional.",
        "Clarify if the capability is mandatory.",
    ),
}

ACTOR_HINTS = (
    "user",
    "admin",
    "administrator",
    "customer",
    "system",
    "operator",
    "guest",
    "client",
    "developer",
    "manager",
    "student",
    "teacher",
    "employee",
)

SECURITY_HINTS = (
    "auth",
    "login",
    "password",
    "token",
    "encrypt",
    "permission",
    "role",
    "access",
    "session",
    "credential",
    "malware",
    "inject",
    "xss",
    "csrf",
)

ERROR_HINTS = (
    "error",
    "fail",
    "invalid",
    "reject",
    "denied",
    "exception",
    "timeout",
    "not be accepted",
    "must not",
    "shall not",
)

OUTPUT_HINTS = (
    "return",
    "display",
    "show",
    "notify",
    "message",
    "response",
    "download",
    "send",
    "email",
)

INPUT_HINTS = (
    "enter",
    "input",
    "upload",
    "provide",
    "submit",
    "select",
    "file",
    "email",
    "password",
)

QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds|minute|minutes|hour|hours|mb|gb|kb|%|percent)?\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|minute|minutes|hour|hours|day|days)\b",
    re.IGNORECASE,
)
CONDITION_RE = re.compile(
    r"\b(if|when|unless|until|provided that|in case)\b.{0,80}",
    re.IGNORECASE,
)


@dataclass
class NLPResult:
    original_text: str
    processed_text: str
    sentences: list[str]
    tokens: list[str]
    extracted: ExtractedComponents
    ambiguities: list[Ambiguity]
    missing_information: list[str]
    quality_score: int
    completeness_score: int
    ambiguity_score: int


def _clean_text(text: str) -> str:
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


@lru_cache(maxsize=1)
def _spacy_nlp() -> Any | None:
    settings = get_settings()
    try:
        import spacy

        return spacy.load(settings.nlp.spacy_model)
    except Exception as exc:  # pragma: no cover - optional model
        logger.warning("spaCy model unavailable; using rule-based NLP fallback: %s", exc)
        return None


@lru_cache(maxsize=1)
def _embedder() -> Any | None:
    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.nlp.embedding_model)
    except Exception as exc:  # pragma: no cover - optional model
        logger.warning("Sentence transformer unavailable; using TF-IDF similarity: %s", exc)
        return None


def _extract_with_spacy(text: str) -> ExtractedComponents:
    nlp = _spacy_nlp()
    if nlp is None:
        return ExtractedComponents()

    doc = nlp(text)
    actors: list[str] = []
    actions: list[str] = []
    objects: list[str] = []
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ in {"NOUN", "PROPN"}:
            actors.append(token.text)
        if token.pos_ == "VERB" and not token.is_stop:
            actions.append(token.lemma_)
        if token.dep_ in {"dobj", "pobj", "attr"} and token.pos_ in {"NOUN", "PROPN"}:
            objects.append(token.text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG"}:
            actors.append(ent.text)
    return ExtractedComponents(
        actors=_unique(actors),
        actions=_unique(actions),
        objects=_unique(objects),
    )


def _rule_extract(text: str) -> ExtractedComponents:
    lowered = text.lower()
    actors = [hint for hint in ACTOR_HINTS if re.search(rf"\b{re.escape(hint)}s?\b", lowered)]
    if "authenticated" in lowered or "registered" in lowered:
        actors.append("authenticated/registered user")
    verbs = re.findall(
        r"\b(allow|upload|reset|login|log in|create|delete|update|send|receive|download|"
        r"validate|reject|accept|generate|display|notify|authenticate|authorize)\b",
        lowered,
    )
    objects = re.findall(
        r"\b(password|email|file|pdf|document|account|user|token|session|report|form)\b",
        lowered,
    )
    inputs = [hint for hint in INPUT_HINTS if hint in lowered]
    outputs = [hint for hint in OUTPUT_HINTS if hint in lowered]
    security = [hint for hint in SECURITY_HINTS if hint in lowered]
    constraints = []
    for match in QUANTITY_RE.finditer(text):
        constraints.append(match.group(0))
    conditions = [match.group(0).strip() for match in CONDITION_RE.finditer(text)]
    time_limits = [match.group(0) for match in TIME_RE.finditer(text)]
    quantitative = [match.group(0) for match in QUANTITY_RE.finditer(text)]
    preconditions = []
    if "authenticated" in lowered:
        preconditions.append("User must be authenticated")
    if "registered" in lowered:
        preconditions.append("User must be registered")
    postconditions = []
    if any(word in lowered for word in ("success", "uploaded", "reset", "created")):
        postconditions.append("Operation completes with a success or failure outcome")
    return ExtractedComponents(
        actors=_unique(actors),
        actions=_unique(verbs),
        objects=_unique(objects),
        inputs=_unique(inputs),
        outputs=_unique(outputs),
        preconditions=_unique(preconditions),
        postconditions=_unique(postconditions),
        constraints=_unique(constraints),
        conditions=_unique(conditions),
        quantitative_values=_unique(quantitative),
        time_limits=_unique(time_limits),
        security_related=_unique(security),
    )


def detect_ambiguities(text: str) -> list[Ambiguity]:
    lowered = text.lower()
    found: list[Ambiguity] = []
    for phrase, (severity, reason, suggestion) in AMBIGUOUS_TERMS.items():
        if phrase in lowered:
            found.append(
                Ambiguity(
                    phrase=phrase,
                    reason=reason,
                    severity=severity,  # type: ignore[arg-type]
                    suggested_clarification=suggestion,
                )
            )
    return found


def _missing_information(text: str, extracted: ExtractedComponents) -> list[str]:
    lowered = text.lower()
    missing: list[str] = []
    if not extracted.actors:
        missing.append("Actor / user role")
    if not extracted.actions:
        missing.append("Primary action")
    if "upload" in lowered or "file" in lowered:
        if not re.search(r"\b(pdf|docx|jpg|png|csv|txt|json)\b", lowered):
            missing.append("Supported file formats")
        if not re.search(r"\b\d+\s*(kb|mb|gb)\b", lowered):
            missing.append("Maximum file size")
    if not extracted.time_limits and any(word in lowered for word in ("fast", "quickly", "soon", "performance")):
        missing.append("Expected response time")
    if not any(hint in lowered for hint in ERROR_HINTS):
        missing.append("Failure / error behavior")
    if extracted.inputs and not extracted.outputs:
        missing.append("Output behavior")
    if any(word in lowered for word in ("user", "account", "upload", "reset")) and not extracted.security_related:
        missing.append("Authentication / authorization requirements")
    if extracted.quantitative_values == [] and any(word in lowered for word in ("limit", "maximum", "at least")):
        missing.append("Boundary / quantitative limits")
    return _unique(missing)


def _scores(
    extracted: ExtractedComponents,
    ambiguities: list[Ambiguity],
    missing: list[str],
    text: str,
) -> tuple[int, int, int]:
    completeness = 100
    completeness -= min(60, len(missing) * 12)
    if extracted.actors:
        completeness += 0
    else:
        completeness -= 8
    completeness = max(0, min(100, completeness))

    severity_weights = {"low": 8, "medium": 16, "high": 24}
    ambiguity_score = min(100, sum(severity_weights[item.severity] for item in ambiguities))
    if not ambiguities:
        ambiguity_score = 5 if "should" in text.lower() else 0

    specificity = 15 if extracted.quantitative_values else 0
    measurability = 15 if extracted.time_limits or extracted.constraints else 0
    clarity = max(0, 40 - ambiguity_score // 3)
    completeness_component = round(completeness * 0.3)
    quality = max(0, min(100, clarity + completeness_component + specificity + measurability + 20))
    if ambiguities and quality > 90:
        quality = 90
    return quality, completeness, ambiguity_score


def analyze_requirement(text: str) -> NLPResult:
    settings = get_settings()
    logger.info("NLP processing started")
    try:
        processed = _clean_text(text)
        sentences = _split_sentences(processed)
        tokens = re.findall(r"\w+|[^\w\s]", processed)
        rule_components = _rule_extract(processed)
        spacy_components = _extract_with_spacy(processed)
        extracted = ExtractedComponents(
            actors=_unique(rule_components.actors + spacy_components.actors),
            actions=_unique(rule_components.actions + spacy_components.actions),
            objects=_unique(rule_components.objects + spacy_components.objects),
            inputs=rule_components.inputs,
            outputs=rule_components.outputs,
            preconditions=rule_components.preconditions,
            postconditions=rule_components.postconditions,
            constraints=rule_components.constraints,
            conditions=rule_components.conditions,
            quantitative_values=rule_components.quantitative_values,
            time_limits=rule_components.time_limits,
            security_related=rule_components.security_related,
        )
        ambiguities = detect_ambiguities(processed) if settings.nlp.ambiguity_detection else []
        missing = (
            _missing_information(processed, extracted)
            if settings.nlp.completeness_analysis
            else []
        )
        quality, completeness, ambiguity = _scores(extracted, ambiguities, missing, processed)
        logger.info("NLP processing completed")
        return NLPResult(
            original_text=text,
            processed_text=processed,
            sentences=sentences,
            tokens=tokens,
            extracted=extracted,
            ambiguities=ambiguities,
            missing_information=missing,
            quality_score=quality,
            completeness_score=completeness,
            ambiguity_score=ambiguity,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("NLP processing failure")
        raise NLPProcessingError() from exc


def semantic_similarity(left: str, right: str) -> float:
    embedder = _embedder()
    if embedder is not None:
        vectors = embedder.encode([left, right])
        from numpy import dot
        from numpy.linalg import norm

        denom = float(norm(vectors[0]) * norm(vectors[1]))
        if denom == 0:
            return 0.0
        return float(dot(vectors[0], vectors[1]) / denom)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    matrix = TfidfVectorizer().fit_transform([left, right])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def nlp_context_blob(result: NLPResult) -> str:
    extracted = result.extracted.model_dump()
    lines = [
        f"Quality={result.quality_score}",
        f"Completeness={result.completeness_score}",
        f"Ambiguity={result.ambiguity_score}",
        f"Extracted={extracted}",
        f"Ambiguities={[item.model_dump() for item in result.ambiguities]}",
        f"Missing={result.missing_information}",
    ]
    return "\n".join(lines)
