import google.generativeai as genai # type: ignore
from utils.config import get_gemini_api_key

# Configure with your API key
api_key = get_gemini_api_key()
genai.configure(api_key=api_key)

# List all available models
print("Available models:")
print("=" * 60)
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Name: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print()