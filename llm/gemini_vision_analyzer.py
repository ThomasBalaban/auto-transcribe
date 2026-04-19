# gemini_vision_analyzer.py
"""
Per-event video analysis used by the onomatopoeia pipeline.

Migrated to the `google-genai` SDK and Gemini 3 Flash. We keep thinking at
`minimal` and media resolution at `low` because this is a high-volume
captioning task where latency and cost matter more than depth of reasoning.
"""

from typing import List, Dict
import json

import cv2  # type: ignore
from PIL import Image

from google.genai import types

from utils.models import (
    MODEL_FLASH,
    THINKING_VISION,
    MEDIA_RES_VISION,
    get_gemini_client_alpha,
    get_safety_settings,
)


class GeminiVisionAnalyzer:
    """Video scene analysis via Gemini 3 Flash."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.client = get_gemini_client_alpha()
        self.safety_settings = get_safety_settings()
        self.num_frames_for_caption = 4
        self.log_func(f"🎬 Gemini Vision analyzer initialized: {MODEL_FLASH}")

    # ── Convenience — kept for the title-less pipeline's back-compat usage ──
    #  modules still want raw frames.)
    def extract_frames_from_video(
        self,
        video_path: str,
        start_time: float,
        duration: float,
    ) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_func(f"ERROR: Could not open video file: {video_path}")
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[Image.Image] = []
        step = duration / max(self.num_frames_for_caption, 1)

        for i in range(self.num_frames_for_caption):
            frame_time = start_time + (i * step)
            frame_pos = int(frame_time * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    # ── Core captioning call ───────────────────────────────────────────────
    def generate_analysis(self, frames: List[Image.Image]) -> Dict:
        """Caption + scene context for a small batch of frames."""
        analysis: Dict = {"caption": "", "scene_context": set()}
        if not frames:
            return analysis

        prompt = (
            "You are a video game scene analyzer. Analyze these sequential "
            "frames from a gameplay clip. Ignore any UI elements or facecam "
            "overlays.\n\n"
            "Respond with a JSON object containing exactly two keys:\n"
            '1. "caption": a concise one-sentence description of the primary '
            "gameplay action.\n"
            '2. "scene_context": a list of relevant environmental keywords. '
            'If the scene is underwater, include the word "underwater". '
            "Otherwise the list may be empty.\n\n"
            'Example: {"caption": "The player shoots a rifle at an enemy.", '
            '"scene_context": []}\n'
            'Example: {"caption": "The player swims through a cave.", '
            '"scene_context": ["underwater"]}\n\n'
            "Here are the frames:"
        )

        # Build the multipart content. Each PIL image becomes a Part.
        contents: List = [prompt]
        contents.extend(frames)

        try:
            response = self.client.models.generate_content(
                model=MODEL_FLASH,
                contents=contents,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_VISION,
                    ),
                    media_resolution=MEDIA_RES_VISION,
                    response_mime_type="application/json",
                ),
            )

            if not response or not getattr(response, "text", None):
                self.log_func("💥 Gemini Vision returned empty response")
                return analysis

            raw_text = response.text.strip().replace(
                "```json", "").replace("```", "")
            parsed = json.loads(raw_text)

            analysis["caption"] = parsed.get("caption", "")
            analysis["scene_context"] = set(parsed.get("scene_context", []))

            self.log_func(
                f"👁️ Vision caption: '{analysis['caption']}'")
            if analysis["scene_context"]:
                self.log_func(
                    f"💧 Scene context: {analysis['scene_context']}")
            return analysis

        except json.JSONDecodeError as e:
            self.log_func(f"💥 Gemini Vision JSON parse error: {e}")
            if response and getattr(response, "text", None):
                self.log_func(f"   Raw response was: {response.text}")
                analysis["caption"] = response.text.strip()
            return analysis
        except Exception as e:
            self.log_func(f"💥 Gemini Vision analysis error: {e}")
            return analysis

    # ── Event-map orchestration ────────────────────────────────────────────
    def analyze_video_at_timestamps(
        self,
        video_path: str,
        events: List[Dict],
        window_duration: float = 5.0,
    ) -> Dict[float, Dict]:
        """
        Run vision analysis once per event and return a
        {event_time: analysis} map keyed by each event's original time.
        """
        analysis_map: Dict[float, Dict] = {}
        for event in events:
            event_time = event["time"]
            start_time = max(0.0, event_time - 2.0)

            frames = self.extract_frames_from_video(
                video_path, start_time, window_duration)
            if not frames:
                continue

            result = self.generate_analysis(frames)
            caption = result["caption"]
            scene_context = result["scene_context"]
            if not caption:
                continue

            analysis_map[event_time] = {
                "video_caption": caption,
                "scene_context": scene_context,
                "confidence": 0.95,
            }
        return analysis_map