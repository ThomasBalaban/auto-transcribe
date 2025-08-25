import json
import os

def get_gemini_api_key():
    """Safely load the Gemini API key from the config file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json with Gemini API key not found.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get("GEMINI_API_KEY")