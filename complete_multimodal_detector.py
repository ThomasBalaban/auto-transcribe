"""
Complete Multimodal Onomatopoeia Detector (Phase 4).
Integrates onset detection + video analysis + multimodal fusion for gaming content.
"""

import os
import numpy as np
import librosa # type: ignore
from typing import List, Dict, Optional, Tuple
import random
import torch

# Import all our phase components
from onset_detector import GamingOnsetDetector
from video_analyzer import VideoAnalyzer
from multimodal_fusion import MultimodalFusionEngine


class CompleteMultimodalDetector:
    """
    Complete multimodal onomatopoeia detection system.
    Combines audio onset detection + video analysis + intelligent fusion.
    """
    
    def __init__(self, 
                 sensitivity: float = 0.5,
                 device: str = "mps",
                 log_func=None):
        """Initialize the complete multimodal system."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print
        
        # Gaming optimization parameters
        self.max_effects_per_minute = 12  # Prevent overwhelming
        self.min_effect_spacing = 0.8    # Minimum time between effects
        
        self.log_func("🚀 Initializing Complete Multimodal Onomatopoeia Detector...")
        
        # Initialize all components
        self._initialize_components()
        
        self.log_func("✅ Complete multimodal system ready!")
        self.log_func(f"   - Gaming-optimized onset detection")
        self.log_func(f"   - VideoMAE + X-CLIP video analysis")
        self.log_func(f"   - Intelligent multimodal fusion")
        self.log_func(f"   - Optimized for {self.device}")

    def _initialize_components(self):
        """Initialize all detection components"""
        try:
            # Initialize audio onset detector
            self.log_func("Loading onset detection system...")
            self.onset_detector = GamingOnsetDetector(
                sensitivity=self.sensitivity,
                log_func=self.log_func
            )
            
            # Initialize video analyzer
            self.log_func("Loading video analysis system...")
            self.video_analyzer = VideoAnalyzer(
                device=self.device,
                log_func=self.log_func
            )
            
            # Initialize fusion engine
            self.log_func("Loading multimodal fusion engine...")
            self.fusion_engine = MultimodalFusionEngine(
                log_func=self.log_func
            )
            
            # Initialize CLAP and Ollama (from original system)
            self.log_func("Loading CLAP + Ollama pipeline...")
            self._initialize_clap_ollama()
            
        except Exception as e:
            self.log_func(f"Failed to initialize components: {e}")
            raise

    def _initialize_clap_ollama(self):
        """Initialize CLAP and Ollama from the original system"""
        try:
            from transformers import ClapModel, ClapProcessor
            from modern_onomatopoeia_detector import OllamaLLM
            
            # Load CLAP model
            self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.clap_model = self.clap_model.to("cpu")  # CPU for compatibility
            
            # Initialize Ollama
            self.ollama_llm = OllamaLLM(log_func=self.log_func)
            
            self.log_func("✅ CLAP + Ollama pipeline ready")
            
        except Exception as e:
            self.log_func(f"Warning: CLAP/Ollama initialization failed: {e}")
            self.clap_model = None
            self.ollama_llm = None

    def analyze_gaming_content(self, video_path: str, audio_path: str = None) -> List[Dict]:
        """
        Main analysis pipeline for gaming content.
        Analyzes both video and audio for optimal onomatopoeia placement.
        """
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"COMPLETE MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")
            
            # Check if input is audio-only
            is_audio_only = video_path.endswith(('.wav', '.mp3', '.flac', '.m4a'))
            
            # Extract audio if not provided separately
            if audio_path is None:
                if is_audio_only:
                    audio_path = video_path  # Use the audio file directly
                else:
                    audio_path = self._extract_audio_from_video(video_path)
            
            # Phase 1: Audio onset detection
            self.log_func(f"\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            if not audio_events:
                self.log_func("No audio events detected")
                return []
            
            self.log_func(f"✅ Detected {len(audio_events)} audio onset events")
            
            # Phase 2: Video analysis (skip for audio-only files)
            video_analyses = []
            if not is_audio_only:
                self.log_func(f"\n🎬 PHASE 2: Video Analysis at Onset Timestamps")
                onset_timestamps = [event['time'] for event in audio_events]
                video_analyses = self.video_analyzer.analyze_video_at_timestamps(
                    video_path, onset_timestamps, window_duration=2.0
                )
                self.log_func(f"✅ Completed video analysis for {len(video_analyses)} timestamps")
            else:
                self.log_func(f"\n🎬 PHASE 2: Skipping video analysis (audio-only file)")
            
            # Phase 3: Enhanced audio analysis with CLAP
            self.log_func(f"\n🎯 PHASE 3: Enhanced Audio Analysis")
            enhanced_audio_events = self._enhance_audio_events(audio_events, audio_path)
            
            # Phase 4: Multimodal fusion (handles both video and audio-only cases)
            self.log_func(f"\n🔄 PHASE 4: {'Multimodal Fusion' if not is_audio_only else 'Audio-Only Processing'}")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, video_analyses
            )
            
            # Phase 5: Gaming-specific optimizations
            self.log_func(f"\n🎮 PHASE 5: Gaming Content Optimization")
            optimized_effects = self._apply_gaming_optimizations(final_effects)
            
            self.log_func(f"\n🎉 {'MULTIMODAL' if not is_audio_only else 'AUDIO-ONLY'} ANALYSIS COMPLETE!")
            self.log_func(f"   - Audio events: {len(audio_events)}")
            self.log_func(f"   - Video analyses: {len(video_analyses)}")
            self.log_func(f"   - Final effects: {len(optimized_effects)}")
            
            return optimized_effects
            
        except Exception as e:
            self.log_func(f"Error in multimodal analysis: {e}")
            return []

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video for analysis"""
        import tempfile
        import subprocess
        
        # Create temporary audio file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"extracted_audio_{random.randint(1000,9999)}.wav")
        
        try:
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '48000',
                '-ac', '1',  # Mono
                audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.log_func(f"✅ Audio extracted: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.log_func(f"Failed to extract audio: {e}")
            raise

    def _enhance_audio_events(self, audio_events: List[Dict], audio_path: str) -> List[Dict]:
        """Enhance audio events with CLAP descriptions and Ollama generation"""
        if not self.clap_model or not audio_events:
            return audio_events
        
        enhanced_events = []
        audio, sr = librosa.load(audio_path, sr=48000)
        
        for event in audio_events:
            try:
                # Extract audio window around event
                onset_time = event['time']
                window_start = max(0, int((onset_time - 1.0) * sr))
                window_end = min(len(audio), int((onset_time + 1.0) * sr))
                audio_chunk = audio[window_start:window_end]
                
                # Get CLAP description
                description = self._get_clap_description(audio_chunk, event)
                
                # Generate onomatopoeia
                if description:
                    onomatopoeia = self._generate_onomatopoeia(description, event)
                    if onomatopoeia:
                        event['clap_description'] = description
                        event['generated_word'] = onomatopoeia
                        enhanced_events.append(event)
                
            except Exception as e:
                self.log_func(f"Error enhancing event at {event['time']:.1f}s: {e}")
                enhanced_events.append(event)  # Keep original event
        
        return enhanced_events

    def _get_clap_description(self, audio_chunk: np.ndarray, event: Dict) -> Optional[str]:
        """Get CLAP description for audio chunk"""
        try:
            # Process audio with CLAP
            inputs = self.clap_processor(
                audios=audio_chunk,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to("cpu")
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)
            
            # Gaming-focused descriptions based on onset type
            onset_type = event.get('onset_type', 'GENERAL')
            
            if onset_type == 'LOW_FREQ':
                descriptions = [
                    "loud explosion sound", "thunder rumbling", "bass impact",
                    "building collapse", "heavy machinery"
                ]
            elif onset_type == 'HIGH_FREQ':
                descriptions = [
                    "gunshot firing", "metal collision", "glass breaking",
                    "electronic beeping", "sharp impact"
                ]
            else:
                descriptions = [
                    "impact sound", "collision", "mechanical sound",
                    "footsteps", "object hitting"
                ]
            
            # Process descriptions
            text_inputs = self.clap_processor(
                text=descriptions,
                return_tensors="pt",
                padding=True
            )
            
            for key in text_inputs:
                if hasattr(text_inputs[key], 'to'):
                    text_inputs[key] = text_inputs[key].to("cpu")
            
            with torch.no_grad():
                text_embeds = self.clap_model.get_text_features(**text_inputs)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                audio_embeds.unsqueeze(1), 
                text_embeds.unsqueeze(0), 
                dim=2
            )
            
            best_idx = similarities.argmax().item()
            best_score = similarities.max().item()
            
            if best_score > 0.1:  # Confidence threshold
                return descriptions[best_idx]
            
            return None
            
        except Exception as e:
            self.log_func(f"CLAP description error: {e}")
            return None

    def _generate_onomatopoeia(self, description: str, event: Dict) -> Optional[str]:
        """Generate onomatopoeia using Ollama or fallback"""
        # Try Ollama first
        if self.ollama_llm and self.ollama_llm.available:
            result = self.ollama_llm.generate_onomatopoeia(description)
            if result:
                return result
        
        # Fallback generation
        return self._fallback_onomatopoeia(description, event)

    def _fallback_onomatopoeia(self, description: str, event: Dict) -> str:
        """Fallback onomatopoeia generation"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['explosion', 'blast']):
            return random.choice(['BOOM!', 'KABOOM!', 'BLAST!'])
        elif any(word in desc_lower for word in ['gun', 'shot']):
            return random.choice(['BANG!', 'SHOT!', 'FIRE!'])
        elif any(word in desc_lower for word in ['metal', 'collision']):
            return random.choice(['CLANG!', 'CRASH!', 'CLANK!'])
        elif any(word in desc_lower for word in ['glass', 'break']):
            return random.choice(['SHATTER!', 'CRASH!', 'SMASH!'])
        elif any(word in desc_lower for word in ['impact', 'hit']):
            return random.choice(['THUD!', 'WHACK!', 'SLAM!'])
        else:
            return random.choice(['BANG!', 'CRASH!', 'THUD!'])

    def _video_only_analysis(self, video_path: str) -> List[Dict]:
        """Fallback to video-only analysis when no audio events"""
        self.log_func("Performing video-only analysis...")
        
        try:
            # Get video duration
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Fix division by zero
            if fps <= 0:
                fps = 30.0  # Default FPS
            if frame_count <= 0:
                frame_count = fps * 60  # Default to 1 minute
                
            duration = frame_count / fps
            cap.release()
            
            # Analyze video in 2-second chunks
            video_events = []
            for start_time in np.arange(0, duration, 2.0):
                chunk_duration = min(2.0, duration - start_time)
                
                # For audio-only files, skip video analysis
                if video_path.endswith('.wav') or video_path.endswith('.mp3'):
                    self.log_func("Audio-only file detected, skipping video analysis")
                    return []
                
                analysis = self.video_analyzer.analyze_video_chunk(
                    video_path, start_time, chunk_duration
                )
                
                if analysis and analysis.get('visual_drama_score', 0) > 0.7:
                    # High visual drama - create effect
                    action = analysis.get('action_classification', {}).get('primary_action', 'visual event')
                    
                    if "quiet" not in action and "peaceful" not in action:
                        effect_word = self._video_to_onomatopoeia(action)
                        
                        event = {
                            'word': effect_word,
                            'start_time': start_time + 1.0,  # Middle of chunk
                            'end_time': start_time + 2.0,
                            'confidence': analysis.get('visual_drama_score', 0.7),
                            'energy': analysis.get('visual_drama_score', 0.7),
                            'context': 'video-only',
                            'source': 'visual_drama'
                        }
                        video_events.append(event)
            
            return video_events
            
        except Exception as e:
            self.log_func(f"Video-only analysis failed: {e}")
            return []
        
    def _video_to_onomatopoeia(self, action: str) -> str:
        """Convert video action to onomatopoeia"""
        action_lower = action.lower()
        
        if "explosion" in action_lower:
            return random.choice(['BOOM!', 'BLAST!', 'KABOOM!'])
        elif "attack" in action_lower:
            return random.choice(['WHACK!', 'STRIKE!', 'HIT!'])
        elif "damage" in action_lower:
            return random.choice(['SLAM!', 'WHAM!', 'CRUSH!'])
        elif "breaking" in action_lower:
            return random.choice(['CRASH!', 'SMASH!', 'SHATTER!'])
        elif "monster" in action_lower:
            return random.choice(['ROAR!', 'GROWL!', 'SNARL!'])
        elif "gunfire" in action_lower:
            return random.choice(['BANG!', 'SHOT!', 'FIRE!'])
        else:
            return random.choice(['IMPACT!', 'ACTION!', 'DRAMA!'])

    def _apply_gaming_optimizations(self, effects: List[Dict]) -> List[Dict]:
        """Apply gaming-specific optimizations to effect list"""
        if not effects:
            return effects
        
        self.log_func(f"🎮 Applying gaming optimizations to {len(effects)} effects...")
        
        # Sort by time
        effects.sort(key=lambda x: x['start_time'])
        
        # Apply density management
        optimized_effects = self._manage_effect_density(effects)
        
        # Apply minimum spacing
        spaced_effects = self._apply_minimum_spacing(optimized_effects)
        
        # Final priority filtering
        final_effects = self._priority_filtering(spaced_effects)
        
        self.log_func(f"✅ Gaming optimization complete: {len(final_effects)} final effects")
        return final_effects

    def _manage_effect_density(self, effects: List[Dict]) -> List[Dict]:
        """Manage effect density - max effects per time window"""
        if not effects:
            return effects
        
        # Group effects into 1-minute buckets
        buckets = {}
        for effect in effects:
            bucket = int(effect['start_time'] // 60)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(effect)
        
        # Keep top N effects per bucket based on confidence
        filtered_effects = []
        for bucket_effects in buckets.values():
            # Sort by confidence
            bucket_effects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            # Keep top N
            filtered_effects.extend(bucket_effects[:self.max_effects_per_minute])
        
        return sorted(filtered_effects, key=lambda x: x['start_time'])

    def _apply_minimum_spacing(self, effects: List[Dict]) -> List[Dict]:
        """Ensure minimum spacing between effects"""
        if not effects:
            return effects
        
        spaced_effects = [effects[0]]  # Keep first effect
        
        for effect in effects[1:]:
            last_effect = spaced_effects[-1]
            time_diff = effect['start_time'] - last_effect['start_time']
            
            if time_diff >= self.min_effect_spacing:
                spaced_effects.append(effect)
            else:
                # Too close - keep higher confidence effect
                if effect.get('confidence', 0) > last_effect.get('confidence', 0):
                    spaced_effects[-1] = effect
        
        return spaced_effects

    def _priority_filtering(self, effects: List[Dict]) -> List[Dict]:
        """Final priority filtering based on gaming content importance"""
        priority_effects = []
        
        for effect in effects:
            context = effect.get('context', '')
            confidence = effect.get('confidence', 0)
            
            # Boost certain contexts
            if any(key in context.lower() for key in ['explosion', 'attack', 'damage', 'monster']):
                confidence *= 1.2
            elif 'quiet' in context.lower():
                confidence *= 0.5
            
            # Apply boosted confidence
            effect['final_confidence'] = min(confidence, 1.0)
            
            # Keep effects above threshold
            if effect['final_confidence'] > 0.4:
                priority_effects.append(effect)
        
        return priority_effects

    def generate_srt_content(self, effects: List[Dict]) -> str:
        """Generate SRT content from final effects"""
        if not effects:
            return ""
        
        srt_lines = []
        for i, effect in enumerate(effects, 1):
            start_time = effect['start_time']
            end_time = effect['end_time']
            word = effect['word']
            
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            srt_lines.extend([
                str(i),
                f"{start_formatted} --> {end_formatted}",
                word,
                ""
            ])
        
        return "\n".join(srt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# Main interface function for integration
def create_multimodal_onomatopoeia_effects(video_path: str,
                                         output_path: str,
                                         sensitivity: float = 0.5,
                                         animation_setting: str = "Random",
                                         device: str = "mps",
                                         log_func=None,
                                         use_animation: bool = True) -> Tuple[bool, List[Dict]]:
    """
    Create onomatopoeia effects using complete multimodal system.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output subtitle file (.srt or .ass)
        sensitivity: AI sensitivity (0.1-0.9)
        animation_setting: Animation type for effects
        device: Processing device ("mps", "cuda", "cpu")
        log_func: Logging function
        use_animation: Whether to create animated effects
        
    Returns:
        Tuple of (success, events_list)
    """
    try:
        if log_func:
            log_func("=== COMPLETE MULTIMODAL ONOMATOPOEIA SYSTEM ===")
            log_func("Phase 1-4: Onset + Video + Fusion + Gaming Optimization")
        
        # Initialize complete detector
        detector = CompleteMultimodalDetector(
            sensitivity=sensitivity,
            device=device,
            log_func=log_func
        )
        
        # Analyze gaming content
        effects = detector.analyze_gaming_content(video_path)
        
        if not effects:
            if log_func:
                log_func("No multimodal effects generated")
            return False, []
        
        # Create output
        if use_animation:
            try:
                # Create animated ASS file
                from animations.core import OnomatopoeiaAnimator
                
                ass_path = os.path.splitext(output_path)[0] + '.ass'
                animator = OnomatopoeiaAnimator()
                animated_content = animator.generate_animated_ass_content(effects, animation_setting)
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(animated_content)
                
                if log_func:
                    log_func(f"✅ Complete multimodal animated effects: {len(effects)} events")
                    log_func(f"🎮 Gaming-optimized with context awareness")
                    log_func(f"🧠 Intelligent audio+video fusion")
                
                return True, effects
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, creating static SRT")
                use_animation = False
        
        if not use_animation:
            # Create static SRT
            srt_content = detector.generate_srt_content(effects)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            if log_func:
                log_func(f"✅ Complete multimodal static effects: {len(effects)} events")
            
            return True, effects
            
    except Exception as e:
        if log_func:
            log_func(f"Error in complete multimodal system: {e}")
        return False, []


def test_complete_system():
    """Test the complete multimodal system"""
    print("Testing complete multimodal onomatopoeia system...")
    
    try:
        detector = CompleteMultimodalDetector(log_func=print)
        
        print("✅ Complete multimodal system initialized!")
        print("✅ Phase 1: Gaming onset detection ready")
        print("✅ Phase 2: VideoMAE + X-CLIP analysis ready") 
        print("✅ Phase 3: Multimodal fusion engine ready")
        print("✅ Phase 4: Gaming optimizations ready")
        print("🎮 Ready for gaming content analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete system failed: {e}")
        return False


if __name__ == "__main__":
    test_complete_system()