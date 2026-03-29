# utils/config.py
import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(
            "config.json not found. Add your API keys there.")
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


def get_gemini_api_key() -> str:
    return _load_config().get("GEMINI_API_KEY", "")


def get_openai_api_key() -> str:
    """Return the OpenAI API key used for Whisper transcription."""
    return _load_config().get("OPENAI_API_KEY", "")