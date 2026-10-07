"""LLM provider abstraction and API clients."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.exceptions import LLMOutputError, LLMUnavailableError
from app.core.logging import logger


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# Tried in order when the primary model returns capacity / not-found errors.
_GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
)


def _is_transient_gemini_error(exc: Exception) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")
    )


class GeminiProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        logger.info("========== GEMINI REQUEST START ==========")

        settings = get_settings()

        # Safe configuration diagnostics.
        # Never logs the actual API key.
        logger.info(
            "Gemini provider configuration: provider=%s, model=%s",
            settings.llm.provider,
            settings.llm.model,
        )

        logger.info(
            "Gemini API key configured: %s",
            bool(settings.llm_api_key),
        )

        logger.info(
            "Gemini request configuration: temperature=%s, timeout=%ss",
            settings.llm.temperature,
            settings.llm.timeout,
        )

        logger.info(
            "Gemini prompt length: %d characters",
            len(prompt),
        )

        try:
            from google import genai

            logger.info("Google GenAI SDK imported successfully")

        except ImportError as exc:
            logger.exception(
                "FAILED: Could not import google.genai"
            )
            raise LLMUnavailableError(
                "Google GenAI SDK is not installed."
            ) from exc

        if not settings.llm_api_key:
            logger.error(
                "FAILED: settings.llm_api_key is empty"
            )
            raise LLMUnavailableError(
                "LLM API key is not configured."
            )

        models: list[str] = []
        for candidate in (settings.llm.model, *_GEMINI_FALLBACK_MODELS):
            if candidate and candidate not in models:
                models.append(candidate)

        last_error: Exception | None = None

        try:
            logger.info("Creating Gemini client...")

            client = genai.Client(
                api_key=settings.llm_api_key
            )

            logger.info("Gemini client created successfully")
            logger.info(
                "Using JSON response mode: application/json"
            )

            for model in models:
                try:
                    logger.info(
                        "Sending request to Gemini model: %s",
                        model,
                    )

                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "temperature": settings.llm.temperature,
                            "response_mime_type": "application/json",
                            "http_options": {
                                "timeout": settings.llm.timeout * 1000
                            },
                        },
                    )

                    logger.info(
                        "Gemini API request completed successfully with model=%s",
                        model,
                    )

                    text = getattr(response, "text", None)

                    logger.info(
                        "Gemini response contains text: %s",
                        bool(text),
                    )

                    if not text:
                        logger.error(
                            "Gemini returned an empty response"
                        )

                        raise LLMUnavailableError(
                            "LLM service returned an empty response."
                        )

                    logger.info(
                        "Gemini response length: %d characters",
                        len(text),
                    )

                    logger.info("========== GEMINI REQUEST SUCCESS ==========")

                    return text

                except LLMUnavailableError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if _is_transient_gemini_error(exc) and model != models[-1]:
                        logger.warning(
                            "Gemini model %s failed (%s); trying next fallback",
                            model,
                            type(exc).__name__,
                        )
                        continue
                    raise

            if last_error is not None:
                raise last_error
            raise LLMUnavailableError("No Gemini models available.")

        except LLMUnavailableError:
            raise

        except Exception as exc:
            # IMPORTANT:
            # We intentionally re-raise the original exception here.
            # Previously every Gemini error was converted into:
            # "LLM API key is not configured."
            #
            # That hid the real root cause.
            logger.exception(
                "========== GEMINI REQUEST FAILED =========="
            )

            logger.error(
                "Gemini exception type: %s",
                type(exc).__name__,
            )

            logger.error(
                "Gemini exception message: %s",
                str(exc),
            )

            raise


class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        logger.info("========== OPENAI REQUEST START ==========")

        settings = get_settings()

        logger.info(
            "OpenAI provider configuration: provider=%s, model=%s",
            settings.llm.provider,
            settings.llm.model,
        )

        logger.info(
            "OpenAI API key configured: %s",
            bool(settings.llm_api_key),
        )

        logger.info(
            "OpenAI prompt length: %d characters",
            len(prompt),
        )

        try:
            from openai import OpenAI

            logger.info("OpenAI SDK imported successfully")

        except ImportError as exc:
            logger.exception(
                "FAILED: Could not import OpenAI SDK"
            )

            raise LLMUnavailableError(
                "OpenAI SDK is not installed."
            ) from exc

        if not settings.llm_api_key:
            logger.error(
                "FAILED: settings.llm_api_key is empty"
            )

            raise LLMUnavailableError(
                "LLM API key is not configured."
            )

        try:
            logger.info("Creating OpenAI client...")

            client = OpenAI(
                api_key=settings.llm_api_key,
                timeout=settings.llm.timeout,
            )

            logger.info(
                "OpenAI client created successfully"
            )

            logger.info(
                "Sending request to OpenAI model: %s",
                settings.llm.model,
            )

            response = client.chat.completions.create(
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You return only valid JSON. "
                            "Never include markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            logger.info(
                "OpenAI API request completed successfully"
            )

            content = response.choices[0].message.content

            if not content:
                logger.error(
                    "OpenAI returned an empty response"
                )

                raise LLMUnavailableError(
                    "LLM service returned an empty response."
                )

            logger.info(
                "OpenAI response length: %d characters",
                len(content),
            )

            logger.info(
                "========== OPENAI REQUEST SUCCESS =========="
            )

            return content

        except LLMUnavailableError:
            raise

        except Exception as exc:
            logger.exception(
                "========== OPENAI REQUEST FAILED =========="
            )

            logger.error(
                "OpenAI exception type: %s",
                type(exc).__name__,
            )

            logger.error(
                "OpenAI exception message: %s",
                str(exc),
            )

            raise

class OllamaProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        logger.info("========== OLLAMA REQUEST START ==========")

        # Ollama runs locally, so prevent broken system SSL
        # certificate/proxy environment variables from affecting it.
        import os

        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("SSL_CERT_DIR", None)

        try:
            from ollama import Client
        except ImportError as exc:
            raise LLMUnavailableError(
                "Ollama Python package is not installed."
            ) from exc

        settings = get_settings()

        try:
            client = Client(
                host="http://localhost:11434",
                trust_env=False,
            )

            logger.info(
                "Sending request to Ollama model: %s",
                settings.llm.model,
            )

            response = client.chat(
                model=settings.llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON-only API. "
                            "Return ONLY valid JSON. "
                            "Do not use markdown. "
                            "Do not add explanations before or after the JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 4096,
                },
            )

            content = response["message"]["content"]

            if not content:
                raise LLMUnavailableError(
                    "Ollama returned an empty response."
                )

            logger.info(
                "Ollama response length: %d characters",
                len(content),
            )

            logger.info(
                "========== OLLAMA REQUEST SUCCESS =========="
            )

            return content

        except LLMUnavailableError:
            raise

        except Exception as exc:
            logger.exception(
                "========== OLLAMA REQUEST FAILED =========="
            )

            logger.error(
                "Ollama exception type: %s",
                type(exc).__name__,
            )

            logger.error(
                "Ollama exception message: %s",
                str(exc),
            )

            raise LLMUnavailableError(
                f"Ollama request failed: {exc}"
            ) from exc

def get_provider() -> LLMProvider:
    logger.info("========== SELECTING LLM PROVIDER ==========")

    settings = get_settings()

    provider = settings.llm.provider.lower().strip()

    logger.info(
        "Configured LLM provider: %s",
        provider,
    )

    logger.info(
        "Configured LLM model: %s",
        settings.llm.model,
    )

    if provider in {
        "gemini",
        "google",
        "google-gemini",
    }:
        logger.info(
            "Selected provider: GeminiProvider"
        )

        return GeminiProvider()

    if provider in {
        "openai",
        "gpt",
    }:
        logger.info(
            "Selected provider: OpenAIProvider"
        )

        return OpenAIProvider()

    if provider in {
        "ollama",
        "llama",
    }:
        logger.info(
            "Selected provider: OllamaProvider"
        )

        return OllamaProvider()

    logger.error(
        "Unsupported LLM provider: %s",
        settings.llm.provider,
    )

    raise LLMUnavailableError(
        f"Unsupported LLM provider: {settings.llm.provider}"
    )

def extract_json(raw: str) -> dict[str, Any]:
    logger.info(
        "========== JSON EXTRACTION START =========="
    )

    logger.info(
        "Raw LLM response length: %d characters",
        len(raw),
    )

    text = raw.strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        re.DOTALL,
    )

    if fenced:
        logger.info(
            "Detected fenced JSON response"
        )

        text = fenced.group(1)

    else:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            logger.info(
                "Extracting JSON object from raw response"
            )

            text = text[start : end + 1]

        else:
            logger.warning(
                "No JSON object boundaries detected"
            )

    try:
        parsed = json.loads(text)

        logger.info(
            "JSON parsing successful"
        )

    except json.JSONDecodeError as exc:
        logger.exception(
            "JSON parsing failed"
        )

        raise LLMOutputError(
            "LLM response was not valid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        logger.error(
            "Parsed JSON is not a dictionary. Type=%s",
            type(parsed).__name__,
        )

        raise LLMOutputError(
            "LLM response must be a JSON object."
        )

    logger.info(
        "========== JSON EXTRACTION SUCCESS =========="
    )

    return parsed


def generate_json(
    prompt: str,
    model_cls: type[BaseModel] | None = None,
) -> Any:

    logger.info(
        "========== GENERATE JSON START =========="
    )

    settings = get_settings()

    logger.info(
        "Settings: provider=%s, model=%s, api_key_configured=%s",
        settings.llm.provider,
        settings.llm.model,
        bool(settings.llm_api_key),
    )

    logger.info(
        "Prompt length: %d characters",
        len(prompt),
    )

    logger.info(
        "Validation model: %s",
        model_cls.__name__ if model_cls else "None",
    )

    provider = get_provider()

    logger.info(
        "Provider instance: %s",
        type(provider).__name__,
    )

    last_error: Exception | None = None

    attempts = max(
        1,
        settings.llm.max_retries,
    )

    logger.info(
        "Maximum LLM attempts: %d",
        attempts,
    )

    current_prompt = prompt

    for attempt in range(
        1,
        attempts + 1,
    ):

        logger.info(
            "========== LLM ATTEMPT %d/%d ==========",
            attempt,
            attempts,
        )

        try:
            logger.info(
                "Calling provider.generate()..."
            )

            raw = provider.generate(
                current_prompt
            )

            logger.info(
                "Provider returned successfully"
            )

            logger.info(
                "LLM response length: %d characters",
                len(raw),
            )

            logger.info(
                "Extracting JSON from LLM response..."
            )

            payload = extract_json(raw)

            logger.info(
                "JSON extraction completed successfully"
            )

            if model_cls is not None:

                logger.info(
                    "Validating response against %s",
                    model_cls.__name__,
                )

                validated = model_cls.model_validate(
                    payload
                )

                logger.info(
                    "Pydantic output validation successful"
                )

                logger.info(
                    "========== GENERATE JSON SUCCESS =========="
                )

                return validated

            logger.info(
                "No Pydantic model provided; "
                "returning parsed JSON"
            )

            logger.info(
                "========== GENERATE JSON SUCCESS =========="
            )

            return payload

        except LLMUnavailableError as exc:

            last_error = exc

            logger.exception(
                "LLMUnavailableError on attempt %d",
                attempt,
            )

            logger.error(
                "Error type: %s",
                type(exc).__name__,
            )

            logger.error(
                "Error message: %s",
                str(exc),
            )

            if attempt < attempts:
                logger.info(
                    "Retrying after LLM unavailable error..."
                )

            time.sleep(
                min(2 * attempt, 6)
            )

        except (
            LLMOutputError,
            ValidationError,
        ) as exc:

            last_error = LLMOutputError(
                "LLM response could not be validated."
            )

            last_error.__cause__ = exc

            logger.exception(
                "LLM output/validation error on attempt %d",
                attempt,
            )

            logger.error(
                "Error type: %s",
                type(exc).__name__,
            )

            logger.error(
                "Error message: %s",
                str(exc),
            )

            current_prompt = (
                "The previous response was not valid JSON "
                "or failed schema validation. "
                "Return ONLY valid JSON matching the "
                "requested schema. "
                "Original task follows:\n\n"
                + prompt
            )

            if attempt < attempts:
                logger.info(
                    "Retrying with corrected JSON instruction..."
                )

            time.sleep(
                min(attempt, 3)
            )

    logger.error(
        "========== GENERATE JSON FAILED =========="
    )

    if isinstance(
        last_error,
        LLMUnavailableError,
    ):
        raise last_error

    raise LLMOutputError() from last_error