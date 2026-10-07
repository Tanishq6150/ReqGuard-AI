"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def avoid_heavy_nlp_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.nlp_service._embedder", lambda: None)
    monkeypatch.setattr("app.services.nlp_service._spacy_nlp", lambda: None)
