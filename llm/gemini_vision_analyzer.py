# gemini_vision_analyzer.py

import google.generativeai as genai # type: ignore
from utils.config import get_gemini_api_key
from typing import List, Dict, Set
from PIL import Image
import cv2 # type: ignore
import numpy as np
import json

class GeminiVisionAnalyzer:
    """Video analysis using Gemini Pro Vision API."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.api_key = get_gemini_api_key()
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API key not found or not set in config.json")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.num_frames_for_caption = 4
        self.log_func("🎬 Gemini Vision analyzer initialized with model: gemini-2.5-flash")

    def extract_frames_from_video(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extracts evenly spaced frames from a video segment."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_func(f"ERROR: Could not open video file: {video_path}")
            return []
            
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        for i in range(self.num_frames_for_caption):
            frame_time = start_time + (i * (duration / self.num_frames_for_caption))
            frame_pos = int(frame_time * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def generate_analysis(self, frames: List[Image.Image]) -> Dict:
        """
        Generates both a caption and scene context using a single Gemini API call.
        """
        analysis = {"caption": "", "scene_context": set()}
        if not frames:
            return analysis

        prompt = [
            "You are a video game scene analyzer. Analyze these sequential frames from a gameplay clip.",
            "Ignore any UI elements or facecam overlays.",
            "Respond with a JSON object containing two keys:",
            '1. "caption": A concise, one-sentence description of the primary gameplay action.',
            '2. "scene_context": A list of relevant environmental keywords. If the scene is underwater, include the word "underwater". Otherwise, the list can be empty.',
            'Example response: {"caption": "The player character shoots a rifle at an enemy.", "scene_context": []}',
            'Another example: {"caption": "The player swims through a submerged cave.", "scene_context": ["underwater"]}',
            "\nHere are the frames:"
        ]
        prompt.extend(frames)

        response = None
        try:
            response = self.model.generate_content(
                prompt, 
                safety_settings=self.safety_settings
            )
            
            # Check if response exists and has text
            if not response or not hasattr(response, 'text') or not response.text:
                self.log_func(f"💥 Gemini Vision returned empty or invalid response")
                return analysis
            
            raw_text = response.text.strip().replace("```json", "").replace("```", "")
            parsed_json = json.loads(raw_text)
            
            analysis["caption"] = parsed_json.get("caption", "")
            analysis["scene_context"] = set(parsed_json.get("scene_context", []))

            self.log_func(f"👁️ Gemini Vision generated caption: '{analysis['caption']}'")
            if analysis["scene_context"]:
                self.log_func(f"💧 Gemini Vision detected context: {analysis['scene_context']}")

            return analysis
        except json.JSONDecodeError as e:
            self.log_func(f"💥 Gemini Vision JSON parsing error: {e}")
            if response and hasattr(response, 'text') and response.text:
                self.log_func(f"   Raw response was: {response.text}")
                analysis["caption"] = response.text.strip()
            return analysis
        except Exception as e:
            self.log_func(f"💥 Gemini Vision analysis error: {e}")
            return analysis
            
    def analyze_video_at_timestamps(self, video_path: str, events: List[Dict], window_duration: float = 5.0) -> Dict[float, Dict]:
        """
        Orchestrates the analysis for multiple events and returns a analysis map.
        The key is the event's original time, ensuring a perfect link.
        """
        analysis_map = {}
        for event in events:
            event_time = event['time']
            start_time = max(0, event_time - 2.0)
            
            frames = self.extract_frames_from_video(video_path, start_time, window_duration)
            if not frames:
                continue
            
            analysis_result = self.generate_analysis(frames)
            caption = analysis_result["caption"]
            scene_context = analysis_result["scene_context"]
            
            if not caption:
                continue
            
            # Use the original event time as the key
            analysis_map[event_time] = {
                "video_caption": caption,
                "scene_context": scene_context,
                "confidence": 0.95
            }
        return analysis_map