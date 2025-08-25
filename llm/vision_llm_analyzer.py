# vision_llm_analyzer.py

import cv2
import numpy as np
import base64
import requests
import json
from typing import List, Dict, Optional, Set
from PIL import Image
import io

class VisionLLMAnalyzer:
    """Video analysis using a local multimodal LLM (like LLaVA) via Ollama."""

    def __init__(self, model_name="llava:latest", base_url="http://localhost:11434", log_func=None):
        self.model_name = model_name
        self.base_url = base_url
        self.log_func = log_func or print
        self.target_fps = 8 # Lower FPS is fine for this approach
        self.num_frames_for_caption = 4 # We'll send a few keyframes
        self.log_func(f"🎬 Vision LLM analyzer initialized with model: {self.model_name}")

    def _pil_to_base64(self, pil_image):
        """Convert a PIL Image to a base64 encoded string."""
        with io.BytesIO() as stream:
            pil_image.save(stream, "PNG")
            return base64.b64encode(stream.getvalue()).decode('utf-8')

    def extract_frames_from_video(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * original_fps)
        
        frames = []
        # Evenly distribute frame selection across the duration
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

    def generate_caption(self, frames: List[Image.Image]) -> Optional[str]:
        if not frames:
            return None

        try:
            base64_images = [self._pil_to_base64(frame) for frame in frames]
            
            prompt = "Analyze these sequential video frames. Provide a short, direct description of the primary action occurring. Focus on the most significant event. Example: 'a character is kicking a wooden door'."

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": base64_images
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=45 # Increased timeout for multimodal models
            )

            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                self.log_func(f"👁️ Vision LLM generated caption: '{caption}'")
                return caption
            else:
                self.log_func(f"❌ Vision LLM request failed with status {response.status_code}: {response.text}")
                return None

        except Exception as e:
            self.log_func(f"💥 Vision LLM generation error: {e}")
            return None

    def analyze_scene_context(self, frames: List[Image.Image]) -> Set[str]:
        context = set()
        if not frames:
            return context

        sample_indices = np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int)
        hues, sats = [], []

        for i in sample_indices:
            frame_np = np.array(frames[i])
            frame_hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(frame_hsv)
            hues.append(np.mean(h))
            sats.append(np.mean(s))
        
        avg_hue = np.mean(hues)
        avg_sat = np.mean(sats)

        if 100 <= avg_hue <= 140 and avg_sat > 80:
            context.add("underwater")

        return context

    def analyze_video_at_timestamps(self, video_path: str, timestamps: List[float], window_duration: float = 3.0) -> List[Dict]:
        all_analyses = []
        for ts in timestamps:
            start_time = max(0, ts - (window_duration / 2))
            frames = self.extract_frames_from_video(video_path, start_time, window_duration)
            if not frames: continue
            
            caption = self.generate_caption(frames)
            scene_context = self.analyze_scene_context(frames)
            
            if not caption: continue

            confidence = 0.95 # We can be highly confident in LLaVA's analysis
            
            analysis = {
                "start_time": start_time, 
                "duration": window_duration,
                "video_caption": caption,
                "scene_context": scene_context,
                "confidence": confidence
            }
            all_analyses.append(analysis)
        return all_analyses