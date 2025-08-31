# ai_director/specialists.py
from typing import Dict, Optional
from ai_director.data_models import DirectorTask, SpecialistResponse
from llm.ollama_integration import OllamaLLM
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer

class SpecialistManager:
    """Manages and executes tasks for all specialist AIs."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.text_analyzer_llm = OllamaLLM(log_func=self.log_func)
        self.log_func("🤖 AI Director Specialist Manager initialized.")

    def dispatch_task(self, task: DirectorTask, full_transcript: str, video_path: str, video_analysis_map: Dict[float, Dict]) -> SpecialistResponse:
        """Routes a task to the appropriate specialist handler."""
        if task.task_type == "analyze_text_content":
            return self._analyze_text_content(task, full_transcript)
        elif task.task_type == "analyze_audio_event":
            return self._analyze_audio_event(task)
        elif task.task_type == "analyze_visual_context":
            return self._analyze_visual_context(task, video_analysis_map)
        else:
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=1.0, details={"error": f"Unknown task type: {task.task_type}"})

    def _analyze_text_content(self, task: DirectorTask, full_transcript: str) -> SpecialistResponse:
        if not self.text_analyzer_llm.available:
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.0, details={"error": "Ollama text analyzer not available."})
        start_char = int(len(full_transcript) * (task.time_range[0] / task.context.get("video_duration", 1)))
        end_char = int(len(full_transcript) * (task.time_range[1] / task.context.get("video_duration", 1)))
        text_segment = full_transcript[start_char:end_char]
        if not text_segment.strip():
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=1.0)
        prompt = (
            "You are a content moderator for gaming videos. Classify the following text from a player's speech into 'wild', 'awkward', or 'normal'.\n"
            "'wild': Outrageousness, standout profanity, sexual humor, over-dramatic reactions.\n"
            "'awkward': Social cringe, conversational mismatches, non-sequiturs.\n"
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
                    details={"text_segment": text_segment}
                )
            elif classification and "awkward" in classification.lower():
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="awkward_content_detected",
                    confidence=0.80,
                    recommended_action="zoom_out"
                )
            else:
                return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.9)
        except Exception as e:
            self.log_func(f"ERROR during text analysis: {e}")
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.0, details={"error": str(e)})

    def _analyze_audio_event(self, task: DirectorTask) -> SpecialistResponse:
        event_context = task.context
        if event_context.get('tier') == 'major' or event_context.get('energy', 0) > 0.7:
            self.log_func(f"Dramatic moment detected at {task.time_range[0]:.2f}s (Tier: {event_context.get('tier')}, Energy: {event_context.get('energy', 0):.2f})")
            return SpecialistResponse(
                task_id=task.task_id,
                result="dramatic_moment_detected",
                confidence=0.9,
                recommended_action="zoom_to_game"
            )
        else:
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=1.0)

    def _analyze_visual_context(self, task: DirectorTask, video_analysis_map: Dict[float, Dict]) -> SpecialistResponse:
        self.log_func(f"Re-analyzing visual context at {task.time_range[0]:.2f}s using cached data...")
        try:
            event_time = task.time_range[0]
            closest_time = min(video_analysis_map.keys(), key=lambda t: abs(t - event_time))
            analysis = video_analysis_map.get(closest_time)
            if not analysis:
                return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.0, details={"error": "No cached vision analysis found for this timestamp."})
            caption = analysis.get('video_caption', '')
            if not caption:
                return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.0, details={"error": "Cached vision analysis has no caption."})
            dramatic_keywords = ["shoot", "shot", "explode", "crash", "attack", "fight", "hit", "punch", "kick"]
            if any(keyword in caption.lower() for keyword in dramatic_keywords):
                self.log_func(f"  - Cached visual analysis confirms dramatic moment: '{caption}'")
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="dramatic_moment_visual",
                    confidence=0.95,
                    recommended_action="zoom_to_game",
                    details={"caption": caption}
                )
            else:
                 self.log_func(f"  - Cached visual analysis shows normal gameplay: '{caption}'")
                 return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.9,
                    details={"caption": caption}
                )
        except Exception as e:
            self.log_func(f"ERROR during cached visual analysis: {e}")
            return SpecialistResponse(task_id=task.task_id, result="no_significant_event", confidence=0.0, details={"error": str(e)})

    def decide_editorial_priority(self, game_event_details: Dict, player_event_details: Dict) -> Optional[str]:
        """Uses an LLM to decide which of two conflicting events is more important."""
        if not self.text_analyzer_llm.available:
            return None # Fallback to default priority if LLM is down

        prompt = (
            "You are an expert video editor for gaming content. You must decide which of two moments is more important to show on screen.\n\n"
            "**CONTEXT:**\n"
            f"- **Game Event:** {game_event_details.get('caption', 'No description')}\n"
            f"- **Player Reaction:** {player_event_details.get('text_segment', 'No transcription')}\n\n"
            "**ANALYSIS:**\n"
            "The game event seems to be a significant, dramatic moment (e.g., a major plot point, a difficult combat encounter). The player's reaction is a typical, potentially generic, exclamation.\n"
            "For a high-quality gaming video, which is more editorially important to show?\n\n"
            "**DECISION:**\n"
            "Respond with ONLY 'game' or 'player'.\n"
            "Decision:"
        )

        try:
            decision = self.text_analyzer_llm.generate(prompt)
            if decision and decision.lower() in ['game', 'player']:
                self.log_func(f"  - Editorial Specialist decided: '{decision.lower()}' is more important.")
                return decision.lower()
            return None
        except Exception as e:
            self.log_func(f"ERROR during editorial decision: {e}")
            return None