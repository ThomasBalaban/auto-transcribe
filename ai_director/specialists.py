# ai_director/specialists.py - UPDATED
from typing import Dict
from ai_director.data_models import DirectorTask, SpecialistResponse
from llm.ollama_integration import OllamaLLM

class SpecialistManager:
    """Manages and executes tasks for all specialist AIs."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.text_analyzer_llm = OllamaLLM(log_func=self.log_func)
        self.log_func("🤖 AI Director Specialist Manager initialized.")

    def dispatch_task(self, task: DirectorTask, full_transcript: str) -> SpecialistResponse:
        """Routes a task to the appropriate specialist handler."""
        if task.task_type == "analyze_text_content":
            return self._analyze_text_content(task, full_transcript)
        elif task.task_type == "analyze_audio_event":
            return self._analyze_audio_event(task)
        elif task.task_type == "analyze_visual_context":
            return self._analyze_visual_context(task)
        else:
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=1.0,
                details={"error": f"Unknown task type: {task.task_type}"}
            )

    def _analyze_text_content(self, task: DirectorTask, full_transcript: str) -> SpecialistResponse:
        # This method remains unchanged
        if not self.text_analyzer_llm.available:
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.0,
                details={"error": "Ollama text analyzer not available."}
            )

        start_char = int(len(full_transcript) * (task.time_range[0] / task.context.get("video_duration", 1)))
        end_char = int(len(full_transcript) * (task.time_range[1] / task.context.get("video_duration", 1)))
        text_segment = full_transcript[start_char:end_char]

        if not text_segment.strip():
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=1.0
            )

        prompt = (
            "You are a content moderator for gaming videos. Your task is to classify a short text segment from a player's speech. "
            "The player is talking while playing a video game.\n\n"
            "Classify the following text into one of three categories:\n"
            "1. 'wild': Contains inappropriate jokes, sexual humor, standout profanity, over-dramatic reactions, or general outrageousness.\n"
            "2. 'awkward': Contains social cringe, conversational mismatches, non-sequiturs, or logically inconsistent replies.\n"
            "3. 'normal': Standard gameplay commentary, conversation, or reactions.\n\n"
            f"Text Segment to Analyze: \"{text_segment}\"\n\n"
            "Respond with ONLY the classification word ('wild', 'awkward', or 'normal').\n\n"
            "Classification:"
        )
        try:
            classification = self.text_analyzer_llm.generate(prompt)

            if classification and "wild" in classification.lower():
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="wild_content_detected",
                    confidence=0.85,
                    recommended_action="zoom_to_cam"
                )
            elif classification and "awkward" in classification.lower():
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="awkward_content_detected",
                    confidence=0.80,
                    recommended_action="zoom_out"
                )
            else:
                return SpecialistResponse(
                    task_id=task.task_id,
                    result="no_significant_event",
                    confidence=0.9
                )
        except Exception as e:
            self.log_func(f"ERROR during text analysis: {e}")
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=0.0,
                details={"error": str(e)}
            )

    def _analyze_audio_event(self, task: DirectorTask) -> SpecialistResponse:
        """
        Analyzes an audio event to determine if it's a dramatic moment.
        """
        event_context = task.context
        
        # --- IMPLEMENTED LOGIC ---
        # A "dramatic moment" is defined as a high-energy audio event.
        # We use the 'tier' and 'energy' from the onomatopoeia detection phase.
        is_major_tier = event_context.get('tier') == 'major'
        is_high_energy = event_context.get('energy', 0) > 0.7

        if is_major_tier or is_high_energy:
            self.log_func(f"Dramatic moment detected at {task.time_range[0]:.2f}s (Tier: {event_context.get('tier')}, Energy: {event_context.get('energy', 0):.2f})")
            return SpecialistResponse(
                task_id=task.task_id,
                result="dramatic_moment_detected",
                confidence=0.9, # High confidence for these events
                recommended_action="zoom_to_game"
            )
        else:
            return SpecialistResponse(
                task_id=task.task_id,
                result="no_significant_event",
                confidence=1.0
            )

    def _analyze_visual_context(self, task: DirectorTask) -> SpecialistResponse:
        """
        Analyzes visual data to provide context for an event.
        (Placeholder for future implementation)
        """
        self.log_func(f"Placeholder: Analyzing visual context at {task.time_range[0]:.2f}s")
        
        return SpecialistResponse(
            task_id=task.task_id,
            result="no_significant_event",
            confidence=1.0,
            details={"status": "Not implemented"}
        )