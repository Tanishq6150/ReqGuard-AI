"""ReqGuard AI Streamlit interface."""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

st.set_page_config(page_title="ReqGuard AI", layout="wide")

DISCLAIMER = (
    "AI-generated test cases should be reviewed by a qualified software tester "
    "before being used in production."
)


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{API_PREFIX}{path}"


def post_json(path: str, requirement: str) -> dict[str, Any]:
    try:
        response = httpx.post(
            api_url(path),
            json={"requirement": requirement},
            timeout=300.0,
            trust_env=False,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not reach the ReqGuard API. Start the FastAPI backend and check API_BASE_URL."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The API returned a non-JSON response.") from exc

    if response.is_error:
        message = payload.get("error") if isinstance(payload, dict) else None
        raise RuntimeError(message or f"API error ({response.status_code}).")
    return payload


def score_color(value: int, invert: bool = False) -> str:
    score = 100 - value if invert else value
    if score >= 75:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


def test_cases_to_csv(cases: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id",
            "title",
            "description",
            "category",
            "priority",
            "preconditions",
            "steps",
            "test_data",
            "expected_result",
        ],
    )
    writer.writeheader()
    for case in cases:
        writer.writerow(
            {
                "id": case.get("id", ""),
                "title": case.get("title", ""),
                "description": case.get("description", ""),
                "category": case.get("category", ""),
                "priority": case.get("priority", ""),
                "preconditions": " | ".join(case.get("preconditions") or []),
                "steps": " | ".join(case.get("steps") or []),
                "test_data": json.dumps(case.get("test_data")),
                "expected_result": case.get("expected_result", ""),
            }
        )
    return output.getvalue()


if "requirement" not in st.session_state:
    st.session_state.requirement = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "improved" not in st.session_state:
    st.session_state.improved = None
if "test_cases" not in st.session_state:
    st.session_state.test_cases = None

st.title("ReqGuard AI")
st.caption("Requirement Analysis & Intelligent Test Generation")
st.info(DISCLAIMER)

requirement = st.text_area(
    "Enter your software requirement...",
    height=180,
    key="requirement",
    placeholder="The system shall allow registered users to reset their password using their email address.",
)

analyze_clicked = st.button("Analyze Requirement", type="primary")
if analyze_clicked:
    try:
        with st.spinner("Analyzing requirement..."):
            st.session_state.analysis = post_json("/analyze", requirement)
            st.session_state.improved = None
            st.session_state.test_cases = None
        st.success("Analysis complete.")
    except RuntimeError as exc:
        st.error(str(exc))

analysis = st.session_state.analysis
if analysis:
    st.subheader("Requirement")
    st.write(analysis.get("original_requirement", requirement))

    st.subheader("Requirement Analysis")
    col1, col2, col3 = st.columns(3)
    quality = int(analysis.get("quality_score", 0))
    completeness = int(analysis.get("completeness_score", 0))
    ambiguity = int(analysis.get("ambiguity_score", 0))
    col1.metric("Quality Score", f"{quality}/100")
    col2.metric("Completeness Score", f"{completeness}/100")
    col3.metric("Ambiguity Score", f"{ambiguity}/100")
    st.caption(
        f":{score_color(quality)}[Quality]  "
        f":{score_color(completeness)}[Completeness]  "
        f":{score_color(ambiguity, invert=True)}[Ambiguity (lower is better)]"
    )

    extracted = analysis.get("extracted") or {}
    st.subheader("Extracted Information")
    extract_cols = st.columns(3)
    groups = [
        ("Actors", extracted.get("actors")),
        ("Actions", extracted.get("actions")),
        ("Objects", extracted.get("objects")),
        ("Inputs", extracted.get("inputs")),
        ("Outputs", extracted.get("outputs")),
        ("Constraints", extracted.get("constraints")),
        ("Preconditions", extracted.get("preconditions")),
        ("Postconditions", extracted.get("postconditions")),
    ]
    for index, (label, values) in enumerate(groups):
        with extract_cols[index % 3]:
            st.markdown(f"**{label}**")
            if values:
                for item in values:
                    st.write(f"- {item}")
            else:
                st.write("- None detected")

    st.subheader("Ambiguity Report")
    ambiguities = analysis.get("ambiguities") or []
    if ambiguities:
        st.dataframe(ambiguities, use_container_width=True)
    else:
        st.write("No ambiguous phrases were detected by the local NLP rules.")

    missing = analysis.get("missing_information") or []
    st.markdown("**Missing information**")
    if missing:
        for item in missing:
            st.write(f"- {item}")
    else:
        st.write("- None highlighted")

    assumptions = analysis.get("assumptions") or []
    if assumptions:
        st.markdown("**Model assumptions (inferred, not explicit)**")
        for item in assumptions:
            st.write(f"- {item}")

    if st.button("Improve Requirement"):
        try:
            with st.spinner("Generating an improved requirement..."):
                st.session_state.improved = post_json("/improve", requirement)
            st.success("Improved requirement generated.")
        except RuntimeError as exc:
            st.error(str(exc))

improved = st.session_state.improved
if improved:
    st.subheader("Improved Requirement")
    st.write(improved.get("improved_requirement", ""))
    extra_assumptions = improved.get("assumptions") or []
    if extra_assumptions:
        st.caption("Assumptions introduced by the model:")
        for item in extra_assumptions:
            st.write(f"- {item}")
    if st.button("Use Improved Requirement"):
        st.session_state.requirement = improved.get("improved_requirement", "")
        st.rerun()

st.subheader("Test Case Generation")
if st.button("Generate Test Cases"):
    try:
        with st.spinner("Generating structured test cases..."):
            generated = post_json("/generate", requirement)
            st.session_state.test_cases = generated
        st.success("Test cases generated.")
    except RuntimeError as exc:
        st.error(str(exc))

generated = st.session_state.test_cases
if generated:
    cases = generated.get("test_cases") or []
    st.caption(generated.get("disclaimer", DISCLAIMER))
    st.dataframe(cases, use_container_width=True)

    json_bytes = json.dumps(generated, indent=2).encode("utf-8")
    csv_text = test_cases_to_csv(cases)
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="reqguard_test_cases.json",
            mime="application/json",
        )
    with download_col2:
        st.download_button(
            "Download CSV",
            data=csv_text.encode("utf-8"),
            file_name="reqguard_test_cases.csv",
            mime="text/csv",
        )
