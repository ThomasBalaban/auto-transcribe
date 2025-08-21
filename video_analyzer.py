# video_analyzer.py

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional
from transformers import XCLIPModel, XCLIPProcessor
from PIL import Image

class VideoAnalyzer:
    """Video analysis using X-CLIP with an expanded label set for better context."""

    def __init__(self, device="mps", log_func=None):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.log_func = log_func or print
        self.target_fps = 16
        self.frame_size = (224, 224)
        self.xclip_frames = 8
        self.xclip_model = None
        self.xclip_processor = None
        self.log_func(f"🎬 Video analyzer initialized on device: {self.device}")
        self._load_models()

    def _load_models(self):
        try:
            self.log_func("Loading X-CLIP video analysis model...")
            self.xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(self.device)
            self.xclip_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
            self.log_func("✅ X-CLIP model loaded successfully.")
        except Exception as e:
            self.log_func(f"FATAL: Failed to load X-CLIP model: {e}")
            raise

    def extract_frames_from_video(self, video_path: str, start_time: float, duration: float) -> List[np.ndarray]:
        # ... (this function remains the same)
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * original_fps)
        end_frame = int((start_time + duration) * original_fps)
        frame_step = max(1, int(original_fps / self.target_fps))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame < end_frame and len(frames) < 32:
            ret, frame = cap.read()
            if not ret: break
            if (current_frame - start_frame) % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame_rgb, self.frame_size))
            current_frame += 1
        cap.release()
        return frames

    def classify_action_xclip(self, frames: List[np.ndarray]) -> Optional[Dict]:
        if not frames or not self.xclip_model: return None
        
        indices = np.linspace(0, len(frames) - 1, self.xclip_frames, dtype=int)
        sampled_frames = [Image.fromarray(frames[i]) for i in indices]

        # --- ADDED "a character fighting underwater" ---
        action_labels = [
            "a character kicking or punching", "an explosion or a large blast",
            "gunfire from a weapon", "a character being hit or taking damage",
            "an object breaking or shattering", "a character fighting underwater", # New label
            "a character falling into water", "a character emerging from water",
            "a character breathing heavily", "a calm scene with no action"
        ]

        inputs = self.xclip_processor(
            text=action_labels, videos=[sampled_frames], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.xclip_model(**inputs)
            probs = outputs.logits_per_video.softmax(dim=1)
            best_idx = probs.argmax().item()

        return {
            "primary_action": action_labels[best_idx],
            "confidence": probs[0, best_idx].item()
        }
        
    def analyze_video_at_timestamps(self, video_path: str, timestamps: List[float], window_duration: float = 2.0) -> List[Dict]:
        # ... (this function remains the same)
        all_analyses = []
        for ts in timestamps:
            start_time = max(0, ts - (window_duration / 2))
            frames = self.extract_frames_from_video(video_path, start_time, window_duration)
            if not frames: continue
            action_info = self.classify_action_xclip(frames)
            if not action_info: continue
            action_score = action_info['confidence']
            if "calm" in action_info['primary_action']: action_score *= 0.1
            analysis = {
                "start_time": start_time, "duration": window_duration,
                "action_classification": action_info, "action_score": min(1.0, action_score)
            }
            all_analyses.append(analysis)
        return all_analyses