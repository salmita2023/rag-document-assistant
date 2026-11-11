"""Shared Gemini client (created once, with automatic retries on rate limits)."""
from functools import lru_cache

from google import genai
from google.genai import types

from app import config


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing: add it to your .env file.")
    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            # Retry automatically on 429 (rate limit) and temporary server errors
            retry_options=types.HttpRetryOptions(attempts=5, http_status_codes=[429, 500, 503]),
        ),
    )
