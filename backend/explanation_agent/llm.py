"""Language-model client. Tests and the template provider do not call the network."""

import os

from explanation_agent.prompt import SYSTEM_PROMPT


class ExplanationModelError(RuntimeError):
    """The explanation model could not be called. No draft was produced."""


class ExplanationClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIExplanationClient(ExplanationClient):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ExplanationModelError("The OpenAI client library is not installed.") from exc

        try:
            client = OpenAI(api_key=self._api_key, timeout=30.0)
            response = client.chat.completions.create(
                model=self._model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise ExplanationModelError("The explanation model request failed.") from exc

        if not content or not content.strip():
            raise ExplanationModelError("The explanation model returned an empty response.")
        return content


class TemplateExplanationClient(ExplanationClient):
    """Marker client. The service builds the draft from retrieved excerpts."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ExplanationModelError("The template provider does not call a model.")


def system_prompt() -> str:
    return SYSTEM_PROMPT


def build_openai_client_from_env() -> OpenAIExplanationClient | None:
    provider = os.getenv("EXPLANATION_PROVIDER", "openai").strip().lower()
    if provider == "template":
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        return None
    model = os.getenv("EXPLANATION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return OpenAIExplanationClient(api_key=api_key, model=model)
