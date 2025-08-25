# gemini_text_generator.py

import config.google.generativeai as genai
from utils.config import get_gemini_api_key
from typing import Set

class GeminiTextGenerator:
    """Onomatopoeia generation using Gemini Pro API."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.api_key = get_gemini_api_key()
        genai.configure(api_key=self.api_key)
        # === CHANGE: Use the recommended 'gemini-1.5-flash-latest' model ===
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        self.log_func("✍️ Gemini Text generator initialized with model: gemini-1.5-flash-latest")

    def generate_onomatopoeia(self, video_caption: str, audio_context: str, scene_context: Set[str]) -> str:
        """Generate onomatopoeia using Gemini Pro."""
        full_context = (
            f"**Gameplay Action:** {video_caption}\n"
            f"**Audio Characteristics:** {audio_context}\n"
            f"**Scene Context:** {', '.join(scene_context) if scene_context else 'N/A'}"
        )

        prompt = (
            "You are a master comic book artist. Create the perfect onomatopoeia based on the following context and strict rules.\n\n"
            "**CONTEXT:**\n"
            f"{full_context}\n\n"
            "**RULES & INSTRUCTIONS:**\n"
            "1. **BE LITERAL AND AVOID CREATIVE WORDS.** The word must literally represent the sound. For example, for a water splash, use 'SPLASH' or 'SPLOOSH', not 'KERCHOW'.\n"
            "2. **Analyze the combined context.**\n"
            "3. **Create a single, impactful onomatopoeia in ALL CAPS suitable for a comic book.**\n"
            "4. **Output ONLY the word.**\n\n"
            "**Onomatopoeia:**"
        )

        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            onomatopoeia = response.text.strip().upper()
            self.log_func(f"✨ Gemini generated: '{onomatopoeia}' from context: {full_context}")
            return onomatopoeia
        except Exception as e:
            self.log_func(f"💥 Gemini generation error: {e}")
            return ""