# ai_director/specialists.py
"""
Specialist AIs dispatched by the Master Director.

Changes vs prior:
  • decide_editorial_priority moved from Ollama to Gemini 3 Flash.
    This is the call that resolves "show the game moment vs show the
    player reaction" conflicts, and it's where smarter editing shows
    up most visibly. Flash with medium thinking is a large step up
    from local Gemma for this kind of nuanced judgment.
  • Ollama is still used for per-segment text content classification
    (wild / awkward / normal) — high-volume sliding-window task where
    local inference is fine.
"""

from typing import Dict, Optional

from google.genai import types

from ai_director.data_models import DirectorTask, SpecialistResponse
from llm.ollama_integration import OllamaLLM
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from utils.models import (
    MODEL_FLASH,
    THINKING_DIRECTOR,
    get_gemini_client,
    get_safety_settings,
)


class SpecialistManager:
    """Manages and executes tasks for all specialist AIs."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print

        # Local model — high-volume text classification over sliding windows
        self.text_analyzer_llm = OllamaLLM(log_func=self.log_func)

        # Gemini — reserved for lower-volume, high-judgment calls
        self._gemini_client = None
        self._gemini_safety = None

        self.log_func("🤖 AI Director Specialist Manager initialized.")

    # ── Lazy Gemini client — avoid paying init cost on videos with no
    #    editorial conflicts. ─────────────────────────────────────────────
    def _gemini(self):
        if self._gemini_client is None:
            self._gemini_client = get_gemini_client()
            self._gemini_safety = get_safety_settings()
        return self._gemini_client

    # ── Dispatch ─────────────────────────────────────────────────────────
    def dispatch_task(
        self,
        task: DirectorTask,
        full_transcript: str,
        video_path: str,
        video_analysis_map: Dict[float, Dict],
    ) -> SpecialistResponse:
        if task.task_type == "analyze_text_content":
            return self._analyze_text_content(task, full_transcript)
        elif task.task_type == "analyze_audio_event":
            return self._analyze_audio_event(task)
        elif task.task_type == "analyze_visual_context":
            return self._analyze_visual_context(task, video_analysis_map)
        return SpecialistResponse(
            task_id=task.task_id,
            result="no_significant_event",
            confidence=1.0,
            details={"error": f"Unknown task type: {task.task_type}"},
        )

    # ── Text content (sliding window) — Ollama ───────────────────────────
    def _analyze_text_content(
        self,
        task: DirectorTask,
        full_transcript: str,
    ) -> SpecialistResponse:
        if not self.text_analyzer_llm.available:
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.0,
                details={"error": "Ollama text analyzer not available."},
            )
        start_char = int(len(full_transcript) * (
            task.time_range[0] / task.context.get("video_duration", 1)))
        end_char = int(len(full_transcript) * (
            task.time_range[1] / task.context.get("video_duration", 1)))
        text_segment = full_transcript[start_char:end_char]
        if not text_segment.strip():
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=1.0,
            )
        prompt = (
            "You are a content moderator for gaming videos. Classify the "
            "following text from a player's speech into 'wild', 'awkward', "
            "or 'normal'.\n"
            "'wild': Outrageousness, standout profanity, sexual humor, "
            "over-dramatic reactions.\n"
            "'awkward': Social cringe, conversational mismatches, "
            "non-sequiturs.\n"
            "'normal': Standard gameplay commentary.\n\n"
            f"Text Segment: \"{text_segment}\"\n\n"
            "Respond with ONLY the classification word.\nClassification:"
        )
        try:
            classification = self.text_analyzer_llm.generate(prompt)
            if classification and "wild" in classification.lower():
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="wild_content_detected",
                    confidence=0.85,
                    recommended_action="zoom_to_cam_reaction",
                    details={"text_segment": text_segment},
                )
            elif classification and "awkward" in classification.lower():
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="awkward_content_detected",
                    confidence=0.80,
                    recommended_action="zoom_out",
                )
            else:
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.9,
                )
        except Exception as e:
            self.log_func(f"ERROR during text analysis: {e}")
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.0,
                details={"error": str(e)},
            )

    # ── Audio event — rule-based ─────────────────────────────────────────
    def _analyze_audio_event(
        self,
        task: DirectorTask,
    ) -> SpecialistResponse:
        event_context = task.context
        if (
            event_context.get("tier") == "major"
            or event_context.get("energy", 0) > 0.7
        ):
            self.log_func(
                f"Dramatic moment detected at {task.time_range[0]:.2f}s "
                f"(Tier: {event_context.get('tier')}, "
                f"Energy: {event_context.get('energy', 0):.2f})"
            )
            return SpecialistResponse(
                task_id=task.task_id,
                result="dramatic_moment_detected",
                confidence=0.9,
                recommended_action="zoom_to_game",
            )
        return SpecialistResponse(
            task_id=task.task_id,
            result="no_significant_event",
            confidence=1.0,
        )

    # ── Visual context — cached Gemini vision data ───────────────────────
    def _analyze_visual_context(
        self,
        task: DirectorTask,
        video_analysis_map: Dict[float, Dict],
    ) -> SpecialistResponse:
        self.log_func(
            f"Re-analyzing visual context at "
            f"{task.time_range[0]:.2f}s using cached data..."
        )
        try:
            event_time = task.time_range[0]
            if not video_analysis_map:
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.0,
                    details={"error": "No cached vision analysis."},
                )
            closest_time = min(
                video_analysis_map.keys(),
                key=lambda t: abs(t - event_time),
            )
            analysis = video_analysis_map.get(closest_time)
            if not analysis:
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.0,
                    details={"error": "No cached vision for timestamp."},
                )
            caption = analysis.get("video_caption", "")
            if not caption:
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.0,
                    details={"error": "Cached vision has no caption."},
                )
            dramatic_keywords = [
                "shoot", "shot", "explode", "crash",
                "attack", "fight", "hit", "punch", "kick",
            ]
            if any(k in caption.lower() for k in dramatic_keywords):
                self.log_func(
                    f"  - Cached visual confirms dramatic moment: "
                    f"'{caption}'"
                )
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="dramatic_moment_visual",
                    confidence=0.95,
                    recommended_action="zoom_to_game",
                    details={"caption": caption},
                )
            self.log_func(
                f"  - Cached visual shows normal gameplay: '{caption}'")
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.9,
                details={"caption": caption},
            )
        except Exception as e:
            self.log_func(f"ERROR during cached visual analysis: {e}")
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.0,
                details={"error": str(e)},
            )

    # ── Editorial priority — Gemini 3 Flash ──────────────────────────────
    def decide_editorial_priority(
        self,
        game_event_details: Dict,
        player_event_details: Dict,
    ) -> Optional[str]:
        """
        Given a conflict between a 'game' moment and a 'player' moment,
        return which one is editorially more important to show. Returns
        'game', 'player', or None on failure (caller falls back to the
        static priority map).

        This moved from Ollama/Gemma to Gemini 3 Flash because the
        nuance of "which moment makes a better clip" is exactly the
        judgment where a frontier model beats a small local one.
        """
        prompt = (
            "You are an expert gaming video editor. Decide which of two "
            "simultaneous moments is more important to show on screen.\n\n"
            "GAME EVENT (what the game is showing):\n"
            f"  {game_event_details.get('caption', 'No description')}\n\n"
            "PLAYER REACTION (what the player is saying):\n"
            f"  {player_event_details.get('text_segment', 'No transcript')}\n\n"
            "Pick 'player' when the player's reaction is genuinely "
            "quotable, ironic, or emotionally bigger than the game event "
            "(e.g., a scream, a confident line right before a fail, a "
            "funny remark). Pick 'game' when the visual moment is a "
            "significant gameplay event (boss, kill, reveal, major damage) "
            "and the player's words are generic commentary.\n\n"
            "Respond with exactly one word: 'game' or 'player'."
        )

        try:
            client = self._gemini()
            response = client.models.generate_content(
                model=MODEL_FLASH,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self._gemini_safety,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_DIRECTOR,
                    ),
                ),
            )
            if not response or not getattr(response, "text", None):
                return None
            decision = response.text.strip().lower()
            # Handle quoting, punctuation, or extra text
            if "player" in decision and "game" not in decision:
                result = "player"
            elif "game" in decision and "player" not in decision:
                result = "game"
            elif decision.startswith("player"):
                result = "player"
            elif decision.startswith("game"):
                result = "game"
            else:
                return None

            self.log_func(
                f"  - Editorial Specialist decided: '{result}' "
                f"is more important."
            )
            return result
        except Exception as e:
            self.log_func(f"ERROR during editorial decision: {e}")
            return None