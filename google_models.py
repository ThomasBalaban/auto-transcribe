"""Debug: list available Gemini models via the new SDK."""

from utils.models import get_gemini_client


def main():
    client = get_gemini_client()

    print("Available models:")
    print("=" * 60)
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"Name: {model.name}")
            display = getattr(model, "display_name", "")
            if display:
                print(f"  Display Name: {display}")
            print(f"  Supported actions: {actions}")
            print()


if __name__ == "__main__":
    main()