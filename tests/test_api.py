"""API tests using mocked LLM calls."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


def test_analyze_rejects_empty_requirement() -> None:
    response = client.post("/api/v1/analyze", json={"requirement": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


def test_analyze_rejects_overlong_requirement() -> None:
    from app.core.config import get_settings

    max_len = get_settings().application.max_requirement_length
    response = client.post("/api/v1/analyze", json={"requirement": "a" * (max_len + 1)})
    assert response.status_code == 400
    assert "maximum length" in response.json()["error"].lower()


def test_analyze_nlp_only(monkeypatch) -> None:
    monkeypatch.setattr("app.services.requirement_service.generate_json", lambda prompt: {})
    response = client.post(
        "/api/v1/analyze",
        json={"requirement": "Authenticated users can upload PDF files up to 50 MB."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "quality_score" in body
    assert body["extracted"]["actions"]


def test_improve_uses_llm(monkeypatch) -> None:
    def fake_generate(prompt, model_cls=None):
        payload = {
            "improved_requirement": "Authenticated users shall upload PDF files up to 50 MB.",
            "assumptions": [],
        }
        return model_cls.model_validate(payload) if model_cls else payload

    monkeypatch.setattr("app.services.requirement_service.generate_json", fake_generate)
    response = client.post(
        "/api/v1/improve",
        json={"requirement": "The system should allow users to upload files quickly."},
    )
    assert response.status_code == 200
    assert "shall" in response.json()["improved_requirement"].lower()


def test_generate_validates_and_returns_cases(monkeypatch) -> None:
    def fake_generate(prompt, model_cls=None):
        payload = {
            "requirement_summary": "Users upload PDFs.",
            "test_cases": [
                {
                    "id": "TC-001",
                    "title": "Upload valid PDF below 50 MB",
                    "description": "Positive upload",
                    "category": "positive",
                    "priority": "high",
                    "preconditions": ["User is authenticated"],
                    "steps": ["Open upload page", "Select a valid PDF below 50 MB", "Click Upload"],
                    "test_data": {"size": "49MB"},
                    "expected_result": "The PDF is successfully uploaded.",
                }
            ],
        }
        return model_cls.model_validate(payload) if model_cls else payload

    monkeypatch.setattr("app.services.testcase_service.generate_json", fake_generate)
    monkeypatch.setattr(
        "app.services.testcase_service.deduplicate_test_cases",
        lambda cases, threshold=None: cases,
    )
    response = client.post(
        "/api/v1/generate",
        json={"requirement": "Authenticated users can upload PDF files up to 50 MB."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["test_cases"][0]["id"] == "TC-001"
    assert "reviewed" in body["disclaimer"].lower()
