"""
Video Analysis System using VideoMAE + X-CLIP for gaming content (Phase 2).
Provides rich video features for multimodal onomatopoeia detection.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from transformers import VideoMAEModel, VideoMAEImageProcessor, XCLIPModel, XCLIPProcessor
from PIL import Image
import os


class VideoAnalyzer:
    """Video analysis using VideoMAE for features + X-CLIP for action classification"""
    
    def __init__(self, device="mps", log_func=None):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.log_func = log_func or print
        
        # Video processing parameters - FIXED for model requirements
        self.target_fps = 16  # Sample frames at 16 FPS for analysis
        self.frame_size = (224, 224)  # Standard size for video models
        self.videomae_frames = 16  # VideoMAE expects 16 frames
        self.xclip_frames = 8     # X-CLIP expects 8 frames
        
        # Models will be loaded lazily
        self.videomae_model = None
        self.videomae_processor = None
        self.xclip_model = None
        self.xclip_processor = None
        
        self.log_func(f"🎬 Video analyzer initialized on {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load VideoMAE and X-CLIP models optimized for Mac M4"""
        try:
            self.log_func("Loading video analysis models...")
            
            # Load VideoMAE for general video features
            self.log_func("Loading VideoMAE model...")
            self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            self.videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.videomae_model.to(self.device)
            
            # Load X-CLIP for action classification
            self.log_func("Loading X-CLIP model...")
            self.xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
            self.xclip_processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
            self.xclip_model.to(self.device)
            
            self.log_func(f"✅ Video models loaded successfully on {self.device}")
            self.log_func(f"   - VideoMAE: Feature extraction ready")
            self.log_func(f"   - X-CLIP: Action classification ready")
            
        except Exception as e:
            self.log_func(f"Failed to load video models: {e}")
            raise

    def extract_frames_from_video(self, video_path: str, start_time: float, 
                             duration: float) -> List[np.ndarray]:
        """Extract frames from video segment for analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling
            start_frame = int(start_time * original_fps)
            end_frame = int((start_time + duration) * original_fps)
            
            # Sample frames at target FPS
            frame_step = max(1, int(original_fps / self.target_fps))
            
            frames = []
            current_frame = start_frame
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # LIMIT to maximum frames needed (32 max, since we'll resample later)
            max_frames = 32
            
            while current_frame < end_frame and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only keep every nth frame based on sampling rate
                if (current_frame - start_frame) % frame_step == 0:
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, self.frame_size)
                    frames.append(frame_resized)
                
                current_frame += 1
            
            cap.release()
            
            self.log_func(f"📹 Extracted {len(frames)} frames from {start_time:.1f}s-{start_time+duration:.1f}s")
            return frames
            
        except Exception as e:
            self.log_func(f"Error extracting frames: {e}")
            return []

    def extract_videomae_features(self, frames: List[np.ndarray]) -> Optional[Dict]:
        """Extract general video features using VideoMAE"""
        if not frames or not self.videomae_model:
            return None
        
        try:
            # VideoMAE expects exactly 16 frames - resample if needed
            if len(frames) != self.videomae_frames:
                # Resample to exactly 16 frames
                indices = np.linspace(0, len(frames)-1, self.videomae_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Convert frames to PIL Images and ensure correct size
            pil_frames = []
            for frame in frames:
                pil_image = Image.fromarray(frame)
                pil_image = pil_image.resize((224, 224), Image.LANCZOS)
                pil_frames.append(pil_image)
            
            # Process with VideoMAE
            inputs = self.videomae_processor(
                images=pil_frames,
                return_tensors="pt"
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.videomae_model(**inputs, output_hidden_states=True)
                video_features = outputs.last_hidden_state.mean(dim=1)
                
                # Calculate derived metrics
                motion_score = self._calculate_motion_score(frames)
                brightness_variance = self._calculate_brightness_variance(frames)
                edge_density = self._calculate_edge_density(frames)
                
                return {
                    "videomae_features": video_features.cpu().numpy(),
                    "motion_score": motion_score,
                    "brightness_variance": brightness_variance,
                    "edge_density": edge_density,
                    "num_frames": len(frames),
                    "visual_complexity": edge_density * brightness_variance
                }
                
        except Exception as e:
            self.log_func(f"Error in VideoMAE feature extraction: {e}")
            # Return manual analysis if VideoMAE fails
            try:
                motion_score = self._calculate_motion_score(frames)
                brightness_variance = self._calculate_brightness_variance(frames)
                edge_density = self._calculate_edge_density(frames)
                
                return {
                    "videomae_features": None,
                    "motion_score": motion_score,
                    "brightness_variance": brightness_variance,
                    "edge_density": edge_density,
                    "num_frames": len(frames),
                    "visual_complexity": edge_density * brightness_variance
                }
            except Exception as fallback_error:
                self.log_func(f"Fallback analysis also failed: {fallback_error}")
                return None
            
   
    def classify_action_xclip(self, frames: List[np.ndarray]) -> Optional[Dict]:
        """Classify action type using X-CLIP"""
        if not frames or not self.xclip_model:
            return None
        
        try:
            # X-CLIP expects exactly 8 frames - resample if needed
            if len(frames) != self.xclip_frames:
                # Resample to exactly 8 frames
                indices = np.linspace(0, len(frames)-1, self.xclip_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Gaming action categories optimized for horror/action games
            action_types = [
                "character attacking with weapon",
                "explosion or blast occurring", 
                "character taking damage or being hit",
                "object breaking or destruction",
                "character moving or running quickly",
                "mechanical or electronic sounds happening",
                "monster or enemy appearing",
                "environmental destruction or collapse",
                "gunfire or shooting",
                "quiet or peaceful scene with no action"
            ]
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Process with X-CLIP - ensure correct format
            inputs = self.xclip_processor(
                text=action_types,
                videos=[pil_frames],  # X-CLIP expects a list of video sequences
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            with torch.no_grad():
                outputs = self.xclip_model(**inputs)
                logits_per_video = outputs.logits_per_video
                probs = logits_per_video.softmax(dim=1)
            
            # Get best classification
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            best_action = action_types[best_idx]
            
            # Get top 3 actions for context
            top_3_indices = probs[0].argsort(descending=True)[:3]
            top_3_actions = [
                {
                    "action": action_types[idx],
                    "confidence": probs[0][idx].item()
                }
                for idx in top_3_indices
            ]
            
            return {
                "primary_action": best_action,
                "confidence": confidence,
                "top_3_actions": top_3_actions,
                "all_scores": dict(zip(action_types, probs[0].tolist()))
            }
            
        except Exception as e:
            self.log_func(f"Error in X-CLIP action classification: {e}")
            return None

    def _calculate_motion_score(self, frames: List[np.ndarray]) -> float:
        """Calculate motion intensity between consecutive frames"""
        if len(frames) < 2:
            return 0.0
        
        motion_scores = []
        for i in range(1, len(frames)):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow magnitude
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, 
                np.array([[100, 100]], dtype=np.float32),  # Simple point
                None
            )[0]
            
            if flow is not None:
                motion_magnitude = np.sqrt(flow[0][0]**2 + flow[0][1]**2)
                motion_scores.append(motion_magnitude)
        
        return float(np.mean(motion_scores)) if motion_scores else 0.0

    def _calculate_brightness_variance(self, frames: List[np.ndarray]) -> float:
        """Calculate brightness variance across frames"""
        brightness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        return float(np.var(brightness_values)) if brightness_values else 0.0

    def _calculate_edge_density(self, frames: List[np.ndarray]) -> float:
        """Calculate average edge density (visual complexity)"""
        edge_densities = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
        
        return float(np.mean(edge_densities)) if edge_densities else 0.0

    def calculate_visual_drama_score(self, video_features: Dict, action_classification: Dict) -> float:
        """Calculate overall visual drama score from features"""
        if not video_features or not action_classification:
            return 0.0
        
        # Base drama from motion and visual complexity
        motion_drama = min(video_features.get("motion_score", 0) / 10.0, 1.0)
        complexity_drama = min(video_features.get("visual_complexity", 0) / 0.1, 1.0)
        brightness_drama = min(video_features.get("brightness_variance", 0) / 1000.0, 1.0)
        
        # Action-based drama boost
        action_confidence = action_classification.get("confidence", 0)
        primary_action = action_classification.get("primary_action", "")
        
        action_drama_multiplier = 1.0
        if "explosion" in primary_action or "destruction" in primary_action:
            action_drama_multiplier = 1.8
        elif "attacking" in primary_action or "damage" in primary_action:
            action_drama_multiplier = 1.5
        elif "monster" in primary_action or "enemy" in primary_action:
            action_drama_multiplier = 1.4
        elif "quiet" in primary_action or "peaceful" in primary_action:
            action_drama_multiplier = 0.3
        
        # Combine all factors
        base_drama = (motion_drama + complexity_drama + brightness_drama) / 3.0
        final_drama = base_drama * action_drama_multiplier * action_confidence
        
        return min(final_drama, 1.0)

    def analyze_video_chunk(self, video_path: str, start_time: float, 
                           duration: float = 2.0) -> Optional[Dict]:
        """Main video analysis pipeline for a time chunk"""
        try:
            self.log_func(f"🎬 Analyzing video chunk: {start_time:.1f}s-{start_time+duration:.1f}s")
            
            # Extract frames
            frames = self.extract_frames_from_video(video_path, start_time, duration)
            if not frames:
                self.log_func(f"❌ No frames extracted from {start_time:.1f}s")
                return None
            
            # Extract VideoMAE features
            video_features = self.extract_videomae_features(frames)
            if not video_features:
                self.log_func(f"❌ VideoMAE feature extraction failed")
                return None
            
            # Classify action with X-CLIP
            action_classification = self.classify_action_xclip(frames)
            if not action_classification:
                self.log_func(f"❌ X-CLIP action classification failed")
                return None
            
            # Calculate visual drama score
            visual_drama = self.calculate_visual_drama_score(video_features, action_classification)
            
            # Combine all analysis
            result = {
                "start_time": start_time,
                "duration": duration,
                "video_features": video_features,
                "action_classification": action_classification,
                "visual_drama_score": visual_drama,
                "frame_count": len(frames)
            }
            
            self.log_func(f"✅ Video analysis complete:")
            self.log_func(f"   Action: {action_classification['primary_action']} ({action_classification['confidence']:.2f})")
            self.log_func(f"   Visual drama: {visual_drama:.2f}")
            self.log_func(f"   Motion: {video_features['motion_score']:.2f}")
            
            return result
            
        except Exception as e:
            self.log_func(f"Error in video chunk analysis: {e}")
            return None

    def analyze_video_at_timestamps(self, video_path: str, 
                                   timestamps: List[float],
                                   window_duration: float = 2.0) -> List[Dict]:
        """Analyze video at specific timestamps (for onset events)"""
        results = []
        
        for timestamp in timestamps:
            # Analyze window around timestamp
            start_time = max(0, timestamp - window_duration/2)
            
            result = self.analyze_video_chunk(video_path, start_time, window_duration)
            if result:
                result["target_timestamp"] = timestamp
                results.append(result)
        
        return results


def test_video_analyzer():
    """Test video analysis system"""
    print("Testing video analysis system...")
    
    try:
        analyzer = VideoAnalyzer(log_func=print)
        
        if analyzer.videomae_model and analyzer.xclip_model:
            print("✅ Video analysis system ready!")
            print("   - VideoMAE: Feature extraction loaded")
            print("   - X-CLIP: Action classification loaded")
            print("   - Ready for gaming content analysis")
            return True
        else:
            print("❌ Video models failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Video analyzer failed: {e}")
        return False


if __name__ == "__main__":
    test_video_analyzer()