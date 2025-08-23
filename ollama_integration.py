# ollama_integration.py

"""
Ollama LLM Integration for Onomatopoeia Generation and Animation Selection.
Handles connection to Ollama API and text generation for sound effects.
"""

import requests
import json
from typing import Optional, Set, List
import re
from animations.animation_types import AnimationType


class OllamaLLM:
    """
    Ollama integration for onomatopoeia generation and AI-driven animation selection.
    """

    def __init__(self, model_name="gemma3:27b", base_url="http://localhost:11434", log_func=None):
        self.model_name = model_name
        self.base_url = base_url
        self.log_func = log_func or print
        self.available = self._check_availability()

    def _check_availability(self):
        """Check if Ollama is running and the specified model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.log_func(f"❌ Ollama not accessible at {self.base_url}")
                return False

            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            if self.model_name not in model_names:
                self.log_func(f"⚠️  Model '{self.model_name}' not found in Ollama. Available models: {model_names}")
                self.log_func(f"💡 To install, run: ollama pull {self.model_name}")
                return False

            self.log_func(f"✅ Ollama integration ready with model: {self.model_name}")
            return True

        except requests.exceptions.RequestException:
            self.log_func(f"❌ Ollama connection failed. Make sure Ollama is running.")
            return False

    def generate_onomatopoeia(self, video_caption: str, audio_context: str, scene_context: Set[str]) -> Optional[str]:
        # ... (This method remains unchanged)
        """Generate onomatopoeia using the local Ollama model with enhanced context."""
        if not self.available:
            return None

        try:
            full_context = (
                f"**Visual Context:** {video_caption}\n"
                f"**Audio Characteristics:** {audio_context}\n"
                f"**Scene Environment:** {', '.join(scene_context) if scene_context else 'N/A'}"
            )
            prompt = (
                "You are an expert comic book artist. Your task is to create the perfect onomatopoeia.\n\n"
                "Here are examples of how to map context to a sound word:\n"
                "- Context: Visual of a gunshot, audio is a sharp, high-frequency sound. -> Onomatopoeia: BLAM!\n"
                "- Context: Visual of a character falling into water, audio is a low-frequency sound. -> Onomatopoeia: SPLOOSH!\n"
                "Now, based on the following context, create the perfect onomatopoeia.\n\n"
                f"**CONTEXT:**\n{full_context}\n\n"
                "**INSTRUCTIONS:**\n"
                "1. Analyze all context provided.\n"
                "2. Create a single, impactful word in ALL CAPS.\n"
                "3. Output ONLY the onomatopoeia word.\n\n"
                "**Onomatopoeia:**"
            )
            payload = { "model": self.model_name, "prompt": prompt, "stream": False }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                raw_output = result.get('response', '').strip()
                cleaned = self._clean_output(raw_output)
                if cleaned:
                    self.log_func(f"✨ Ollama generated: '{cleaned}'")
                    return cleaned
            return None
        except Exception as e:
            self.log_func(f"💥 Ollama generation error: {e}")
            return None


    def choose_animation(self, onomatopoeia: str, audio_context: str, video_caption: str) -> Optional[str]:
        """
        Uses the LLM to intelligently choose the best animation type.
        """
        if not self.available:
            return None

        # Get the list of valid animation types
        valid_animations = AnimationType.get_all_types()

        # Create descriptions for the prompt
        animation_descriptions = "\n".join(
            f"- {name}: {desc}"
            for name, desc in AnimationType.get_display_names().items()
        )

        prompt = (
            "You are an expert motion graphics director. Your task is to choose the best animation for an on-screen onomatopoeia based on the event's context.\n\n"
            "Here are the available animations and their uses:\n"
            f"{animation_descriptions}\n\n"
            "CONTEXT:\n"
            f"- Onomatopoeia Word: '{onomatopoeia}'\n"
            f"- Audio Details: {audio_context}\n"
            f"- Video Details: '{video_caption}'\n\n"
            "Based on the context, which animation from the list is the most appropriate? "
            "Respond with ONLY the name of the animation (e.g., 'shake', 'pop_shrink').\n\n"
            "Animation:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": { "temperature": 0.2 }
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=20)
            if response.status_code == 200:
                result = response.json()
                choice = result.get('response', '').strip().lower().replace(" ", "_")

                # Validate the choice
                if choice in valid_animations:
                    self.log_func(f"🧠 AI chose animation: '{choice}'")
                    return choice
                else:
                    self.log_func(f"⚠️ AI returned an invalid animation choice: '{choice}'. Falling back.")
                    return None
            return None
        except Exception as e:
            self.log_func(f"💥 AI animation choice error: {e}")
            return None


    def _clean_output(self, text: str) -> Optional[str]:
        """Cleans the output from the LLM to get a single, valid onomatopoeia."""
        if not text:
            return None
        text = re.sub(r'\(.*\)', '', text)
        text = re.sub(r'[^A-Z-!]', '', text.upper())
        match = re.search(r'^[A-Z-!]+', text)
        if match:
            return match.group(0)
        return None