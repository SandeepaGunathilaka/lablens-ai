"""Language-model client. Tests and the template provider do not call the network."""

import os

DEFAULT_MODEL = "gemini-3.8-flash"
_TIMEOUT_MS = 30_000
_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here"}


class ExplanationModelError(RuntimeError):
    """The explanation model could not be called. No draft was produced."""


class ExplanationClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiExplanationClient(ExplanationClient):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ExplanationModelError("The google-genai client library is not installed.") from exc

        try:
            client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=_TIMEOUT_MS),
            )
            response = client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            )
            content = response.text
        except Exception as exc:
            raise ExplanationModelError("The explanation model request failed.") from exc

        if not content or not content.strip():
            raise ExplanationModelError("The explanation model returned an empty response.")
        return content


class TemplateExplanationClient(ExplanationClient):
    """Marker client. The service builds the draft from retrieved excerpts."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ExplanationModelError("The template provider does not call a model.")


def build_gemini_client_from_env() -> GeminiExplanationClient | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key in _PLACEHOLDER_KEYS:
        return None
    model = os.getenv("EXPLANATION_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return GeminiExplanationClient(api_key=api_key, model=model)
