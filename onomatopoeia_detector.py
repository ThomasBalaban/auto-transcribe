# onomatopoeia_detector.py

import os
from typing import List, Dict, Tuple

# Import all the necessary components from your project
from onset_detector import GamingOnsetDetector
from video_analyzer import VideoAnalyzer
from multimodal_fusion import MultimodalFusionEngine
from audio_enhancement import AudioEnhancer
from gaming_optimizer import GamingOptimizer
from subtitle_generator import SubtitleGenerator
from file_processor import FileProcessor


class OnomatopoeiaDetector:
    """
    Main onomatopoeia detection system that orchestrates the complete
    multimodal detection pipeline for superior accuracy and context-awareness.
    """

    def __init__(self, sensitivity: float = 0.5, device: str = "mps", log_func=None):
        """Initialize the full detection system."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print

        self.log_func("🚀 Initializing Multimodal Onomatopoeia Detector...")
        self._initialize_components()
        self.log_func("✅ Multimodal system ready!")

    def _initialize_components(self):
        """Initialize all detection components from your project."""
        try:
            self.log_func("Loading core detection systems...")
            self.onset_detector = GamingOnsetDetector(sensitivity=self.sensitivity, log_func=self.log_func)
            self.video_analyzer = VideoAnalyzer(device=self.device, log_func=self.log_func)
            self.fusion_engine = MultimodalFusionEngine(log_func=self.log_func)

            self.log_func("Loading processing and enhancement modules...")
            self.audio_enhancer = AudioEnhancer(log_func=self.log_func)
            self.gaming_optimizer = GamingOptimizer(log_func=self.log_func)
            self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
            self.file_processor = FileProcessor(log_func=self.log_func)

        except Exception as e:
            self.log_func(f"FATAL: Failed to initialize core components: {e}")
            raise

    def analyze_file(self, input_path: str) -> List[Dict]:
        """
        Main analysis method. It automatically detects the file type (video or audio)
        and routes it to the appropriate processing pipeline.
        """
        file_type = self.file_processor.detect_file_type(input_path)

        if file_type == 'video':
            return self._analyze_video_file(video_path=input_path)
        elif file_type == 'audio':
            return self._analyze_audio_file(audio_path=input_path)
        else:
            self.log_func(f"Unsupported file type: {os.path.splitext(input_path)[1]}")
            return []

    def _analyze_video_file(self, video_path: str) -> List[Dict]:
        """
        Executes the full multimodal analysis pipeline for video files. This is
        the core of the new system.
        """
        audio_path = None
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")

            # Step 0: Extract audio from video for analysis
            audio_path = self.file_processor.extract_audio_from_video(video_path)

            # PHASE 1: Precise Audio Onset Detection
            self.log_func(f"\n📊 PHASE 1: Detecting Precise Audio Onsets")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            if not audio_events:
                self.log_func("No significant audio events detected. Stopping analysis.")
                return []
            self.log_func(f"✅ Detected {len(audio_events)} potential audio onset events.")

            # PHASE 2: Contextual Video Analysis at Onset Timestamps
            self.log_func(f"\n🎬 PHASE 2: Analyzing Video Context at Onset Timestamps")
            onset_timestamps = [event['time'] for event in audio_events]
            video_analyses = self.video_analyzer.analyze_video_at_timestamps(
                video_path, onset_timestamps, window_duration=2.0
            )
            self.log_func(f"✅ Completed video analysis for {len(video_analyses)} timestamps.")

            # PHASE 3: Enhanced Audio Analysis with CLAP
            self.log_func(f"\n🎧 PHASE 3: Enhancing Audio Events with CLAP Model")
            enhanced_audio_events = self.audio_enhancer.enhance_audio_events(audio_events, audio_path)
            self.log_func(f"✅ Enhanced {len(enhanced_audio_events)} audio events with descriptions.")

            # PHASE 4: Multimodal Fusion
            self.log_func(f"\n🔄 PHASE 4: Fusing Audio and Video Intelligence")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, video_analyses
            )
            self.log_func(f"✅ Fusion complete. Generated {len(final_effects)} initial onomatopoeia effects.")

            # PHASE 5: Gaming-Specific Optimizations
            self.log_func(f"\n🎮 PHASE 5: Applying Gaming Content Optimizations")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(f"✅ Optimization complete. Final effect count: {len(optimized_effects)}.")

            self.log_func(f"\n🎉 MULTIMODAL ANALYSIS COMPLETE!")
            return optimized_effects

        except Exception as e:
            self.log_func(f"ERROR during multimodal analysis pipeline: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []
        finally:
            # Important: Clean up the temporary audio file
            if audio_path:
                self.file_processor.cleanup_temp_file(audio_path)

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Runs a simplified, audio-only version of the pipeline."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING AUDIO-ONLY ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")

            # Run audio-focused phases
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            if not audio_events:
                return []

            enhanced_audio_events = self.audio_enhancer.enhance_audio_events(audio_events, audio_path)
            
            # The fusion engine will intelligently handle the lack of video data
            final_effects = self.fusion_engine.process_multimodal_events(enhanced_audio_events, [])
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)

            self.log_func(f"\n🎉 AUDIO-ONLY ANALYSIS COMPLETE! Found {len(optimized_effects)} effects.")
            return optimized_effects

        except Exception as e:
            self.log_func(f"Error in audio-only analysis: {e}")
            return []

    def create_subtitle_file(self, input_path: str, output_path: str,
                           animation_type: str = "Random") -> Tuple[bool, List[Dict]]:
        """
        High-level function to analyze a file and generate the corresponding subtitle file.
        """
        try:
            events = self.analyze_file(input_path)
            if not events:
                self.log_func("No onomatopoeia events were detected to generate a subtitle file.")
                return False, []

            success = self.subtitle_generator.create_subtitle_file(
                events, output_path, animation_type
            )
            return success, events

        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            return False, []