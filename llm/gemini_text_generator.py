# gemini_text_generator.py
"""
Cloud fallback for onomatopoeia word generation.

In the current pipeline this is shadowed by Ollama — the local model handles
onomatopoeia and animation selection — but this module stays around for
cases where Ollama isn't running.
"""

from typing import Set

from google.genai import types

from utils.models import (
    MODEL_FLASH,
    THINKING_VISION,
    get_gemini_client,
    get_safety_settings,
)


class GeminiTextGenerator:
    """Onomatopoeia generation via Gemini 3 Flash (cloud fallback)."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.client = get_gemini_client()
        self.safety_settings = get_safety_settings()
        self.log_func(
            f"✍️ Gemini Text generator initialized: {MODEL_FLASH}")

    def generate_onomatopoeia(
        self,
        video_caption: str,
        audio_context: str,
        scene_context: Set[str],
    ) -> str:
        """Ask Gemini for a single onomatopoeia word based on the event."""
        full_context = (
            f"**Gameplay Action:** {video_caption}\n"
            f"**Audio Characteristics:** {audio_context}\n"
            f"**Scene Context:** "
            f"{', '.join(scene_context) if scene_context else 'N/A'}"
        )

        prompt = (
            "You are a master comic book artist. Create the perfect "
            "onomatopoeia based on the following context and strict rules.\n\n"
            f"**CONTEXT:**\n{full_context}\n\n"
            "**RULES & INSTRUCTIONS:**\n"
            "1. BE LITERAL AND AVOID CREATIVE WORDS. The word must literally "
            "represent the sound. For a water splash use 'SPLASH' or "
            "'SPLOOSH', not 'KERCHOW'.\n"
            "2. Analyze the combined context.\n"
            "3. Create a single, impactful onomatopoeia in ALL CAPS suitable "
            "for a comic book.\n"
            "4. Output ONLY the word.\n\n"
            "**Onomatopoeia:**"
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_FLASH,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_VISION,
                    ),
                ),
            )
            if not response or not getattr(response, "text", None):
                self.log_func("💥 Gemini returned empty response")
                return ""
            onomatopoeia = response.text.strip().upper()
            self.log_func(
                f"✨ Gemini generated: '{onomatopoeia}' "
                f"from context: {full_context}"
            )
            return onomatopoeia
        except Exception as e:
            self.log_func(f"💥 Gemini generation error: {e}")
            return ""