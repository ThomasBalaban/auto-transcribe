"""
Main Onomatopoeia Detection System.
Orchestrates the complete multimodal detection pipeline.
"""

import os
from typing import List, Dict, Tuple

# Import all our components
from onset_detector import GamingOnsetDetector
from video_analyzer import VideoAnalyzer
from multimodal_fusion import MultimodalFusionEngine
from audio_enhancement import AudioEnhancer
from gaming_optimizer import GamingOptimizer
from subtitle_generator import SubtitleGenerator
from file_processor import FileProcessor


class OnomatopoeiaDetector:
    """
    Main onomatopoeia detection system.
    Handles both video files (multimodal) and audio files (audio-only).
    """
    
    def __init__(self, sensitivity: float = 0.5, device: str = "mps", log_func=None):
        """Initialize the detection system."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print
        
        self.log_func("🚀 Initializing Multimodal Onomatopoeia Detector...")
        self._initialize_components()
        self.log_func("✅ Multimodal system ready!")

    def _initialize_components(self):
        """Initialize all detection components"""
        try:
            # Initialize core components
            self.log_func("Loading core detection systems...")
            self.onset_detector = GamingOnsetDetector(
                sensitivity=self.sensitivity,
                log_func=self.log_func
            )
            
            self.video_analyzer = VideoAnalyzer(
                device=self.device,
                log_func=self.log_func
            )
            
            self.fusion_engine = MultimodalFusionEngine(
                log_func=self.log_func
            )
            
            # Initialize processing modules
            self.log_func("Loading processing modules...")
            self.audio_enhancer = AudioEnhancer(log_func=self.log_func)
            self.gaming_optimizer = GamingOptimizer(log_func=self.log_func)
            self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
            self.file_processor = FileProcessor(log_func=self.log_func)
            
        except Exception as e:
            self.log_func(f"Failed to initialize components: {e}")
            raise

    def analyze_file(self, input_path: str) -> List[Dict]:
        """
        Main analysis method - automatically detects file type and uses appropriate pipeline.
        
        Args:
            input_path: Path to video or audio file
            
        Returns:
            List of detected onomatopoeia events
        """
        file_type = self.file_processor.detect_file_type(input_path)
        
        if file_type == 'video':
            return self._analyze_video_file(input_path)
        elif file_type == 'audio':
            return self._analyze_audio_file(input_path)
        else:
            self.log_func(f"Unsupported file type: {os.path.splitext(input_path)[1]}")
            return []

    def _analyze_video_file(self, video_path: str) -> List[Dict]:
        """Full multimodal analysis for video files."""
        audio_path = None
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")
            
            # Extract audio from video
            audio_path = self.file_processor.extract_audio_from_video(video_path)
            
            # Phase 1: Audio onset detection
            self.log_func(f"\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            if not audio_events:
                self.log_func("No audio events detected")
                return []
            
            self.log_func(f"✅ Detected {len(audio_events)} audio onset events")
            
            # Phase 2: Video analysis at onset timestamps
            self.log_func(f"\n🎬 PHASE 2: Video Analysis at Onset Timestamps")
            onset_timestamps = [event['time'] for event in audio_events]
            video_analyses = self.video_analyzer.analyze_video_at_timestamps(
                video_path, onset_timestamps, window_duration=2.0
            )
            self.log_func(f"✅ Completed video analysis for {len(video_analyses)} timestamps")
            
            # Phase 3: Enhanced audio analysis with CLAP
            self.log_func(f"\n🎯 PHASE 3: Enhanced Audio Analysis")
            enhanced_audio_events = self.audio_enhancer.enhance_audio_events(audio_events, audio_path)
            
            # Phase 4: Multimodal fusion
            self.log_func(f"\n🔄 PHASE 4: Multimodal Fusion")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, video_analyses
            )
            
            # Phase 5: Gaming-specific optimizations
            self.log_func(f"\n🎮 PHASE 5: Gaming Content Optimization")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            
            self.log_func(f"\n🎉 MULTIMODAL ANALYSIS COMPLETE!")
            self.log_func(f"   - Audio events: {len(audio_events)}")
            self.log_func(f"   - Video analyses: {len(video_analyses)}")
            self.log_func(f"   - Final effects: {len(optimized_effects)}")
            
            return optimized_effects
            
        except Exception as e:
            self.log_func(f"Error in multimodal analysis: {e}")
            return []
        finally:
            # Cleanup temp audio file
            if audio_path:
                self.file_processor.cleanup_temp_file(audio_path)

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Audio-only analysis for audio files."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"AUDIO-ONLY ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")
            
            # Phase 1: Audio onset detection
            self.log_func(f"\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            if not audio_events:
                self.log_func("No audio events detected")
                return []
            
            # Phase 3: Enhanced audio analysis with CLAP
            self.log_func(f"\n🎯 PHASE 3: Enhanced Audio Analysis")
            enhanced_audio_events = self.audio_enhancer.enhance_audio_events(audio_events, audio_path)
            
            # Phase 4: Audio-only processing (no video data)
            self.log_func(f"\n🔄 PHASE 4: Audio-Only Processing")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, []  # Empty video analyses
            )
            
            # Phase 5: Gaming optimizations
            self.log_func(f"\n🎮 PHASE 5: Gaming Content Optimization")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            
            self.log_func(f"\n🎉 AUDIO-ONLY ANALYSIS COMPLETE!")
            self.log_func(f"   - Audio events: {len(audio_events)}")
            self.log_func(f"   - Final effects: {len(optimized_effects)}")
            
            return optimized_effects
            
        except Exception as e:
            self.log_func(f"Error in audio-only analysis: {e}")
            return []

    def create_subtitle_file(self, input_path: str, output_path: str, 
                           animation_type: str = "Random") -> Tuple[bool, List[Dict]]:
        """
        Create subtitle file with onomatopoeia effects.
        
        Args:
            input_path: Input video/audio file
            output_path: Output subtitle file (.srt or .ass)
            animation_type: Animation type for effects
            
        Returns:
            Tuple of (success, events_list)
        """
        try:
            # Analyze file
            events = self.analyze_file(input_path)
            
            if not events:
                self.log_func("No onomatopoeia events detected")
                return False, []
            
            # Create subtitle file
            success = self.subtitle_generator.create_subtitle_file(
                events, output_path, animation_type
            )
            
            return success, events
                
        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            return False, []


# Convenient function for backward compatibility in existing code
def create_onomatopoeia_srt(input_path: str, output_path: str, 
                           sensitivity: float = 0.5, animation_type: str = "Random",
                           log_func=None) -> Tuple[bool, List[Dict]]:
    """
    Convenience function to create onomatopoeia subtitle file.
    """
    detector = OnomatopoeiaDetector(sensitivity=sensitivity, log_func=log_func)
    return detector.create_subtitle_file(input_path, output_path, animation_type)


# Backward compatibility alias
CompleteMultimodalDetector = OnomatopoeiaDetector


def test_detector():
    """Test the cleaned detector system"""
    print("Testing modular onomatopoeia detector...")
    
    try:
        detector = OnomatopoeiaDetector(log_func=print)
        print("✅ Detector initialized successfully!")
        print("✅ Ready for multimodal onomatopoeia detection!")
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False


if __name__ == "__main__":
    test_detector()