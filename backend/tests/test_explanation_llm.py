from types import SimpleNamespace

import pytest
from google import genai

from explanation_agent.llm import (
    DEFAULT_MODEL,
    ExplanationModelError,
    GeminiExplanationClient,
    build_gemini_client_from_env,
)
from explanation_agent.service import build_explanation_service


class FakeModels:
    def __init__(self, text="{}", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


@pytest.fixture
def fake_gemini(monkeypatch):
    models = FakeModels()
    created = []

    def make_client(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(models=models)

    monkeypatch.setattr(genai, "Client", make_client)
    return SimpleNamespace(models=models, created=created)


@pytest.mark.parametrize("key", ["", "  ", "your_gemini_api_key_here"])
def test_missing_or_placeholder_key_means_no_client(monkeypatch, key):
    monkeypatch.setenv("GEMINI_API_KEY", key)

    assert build_gemini_client_from_env() is None


def test_missing_key_makes_the_service_unavailable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("EXPLANATION_PROVIDER", raising=False)

    assert build_explanation_service().mode == "unavailable"


def test_model_defaults_and_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("EXPLANATION_MODEL", raising=False)
    assert build_gemini_client_from_env()._model == DEFAULT_MODEL

    monkeypatch.setenv("EXPLANATION_MODEL", "gemini-3.5-flash")
    assert build_gemini_client_from_env()._model == "gemini-3.5-flash"


def test_key_enables_the_llm_service(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("EXPLANATION_PROVIDER", raising=False)

    assert build_explanation_service().mode == "llm"


def test_complete_sends_system_prompt_and_requests_json(fake_gemini):
    fake_gemini.models.text = '{"ok": true}'

    raw = GeminiExplanationClient(api_key="test-key", model="gemini-3.8-flash").complete("SYSTEM", "USER")

    assert raw == '{"ok": true}'
    assert fake_gemini.created[0]["api_key"] == "test-key"
    [call] = fake_gemini.models.calls
    assert call["model"] == "gemini-3.8-flash"
    assert call["contents"] == "USER"
    assert call["config"].system_instruction == "SYSTEM"
    assert call["config"].response_mime_type == "application/json"


def test_sdk_errors_become_model_errors(fake_gemini):
    fake_gemini.models.error = RuntimeError("quota exceeded")

    with pytest.raises(ExplanationModelError):
        GeminiExplanationClient(api_key="test-key", model="gemini-3.8-flash").complete("SYSTEM", "USER")


def test_empty_response_is_a_model_error(fake_gemini):
    fake_gemini.models.text = "   "

    with pytest.raises(ExplanationModelError):
        GeminiExplanationClient(api_key="test-key", model="gemini-3.8-flash").complete("SYSTEM", "USER")
