# onomatopoeia_detector.py

import os
from typing import List, Dict, Tuple

# Import all the necessary components from your project
from onset_detector import GamingOnsetDetector
from gemini_vision_analyzer import GeminiVisionAnalyzer 
from multimodal_fusion import MultimodalFusionEngine
from gaming_optimizer import GamingOptimizer
from subtitle_generator import SubtitleGenerator
from file_processor import FileProcessor


class OnomatopoeiaDetector:
    """
    Main onomatopoeia detection system that orchestrates the complete
    multimodal detection pipeline using the Gemini API for superior accuracy.
    """

    def __init__(self, sensitivity: float = 0.5, device: str = "cpu", log_func=None):
        """Initialize the full detection system."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print

        self.log_func("🚀 Initializing Multimodal Onomatopoeia Detector with Gemini...")
        self._initialize_components()
        self.log_func("✅ Gemini-powered multimodal system ready!")

    def _initialize_components(self):
        """Initialize all detection components for the Gemini pipeline."""
        try:
            self.log_func("Loading core detection systems...")
            self.onset_detector = GamingOnsetDetector(sensitivity=self.sensitivity, log_func=self.log_func)
            self.video_analyzer = GeminiVisionAnalyzer(log_func=self.log_func) 
            self.fusion_engine = MultimodalFusionEngine(log_func=self.log_func)

            self.log_func("Loading processing and utility modules...")
            self.gaming_optimizer = GamingOptimizer(log_func=self.log_func)
            self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
            self.file_processor = FileProcessor(log_func=self.log_func)

        except Exception as e:
            self.log_func(f"FATAL: Failed to initialize core components: {e}")
            raise

    def _filter_events_with_cooldown(self, events: List[Dict], cooldown_period: float = 2.0) -> List[Dict]:
        """
        Filters events using an impact-aware cooldown. A sharper, more impactful event
        can override a less impactful one within the cooldown period.
        """
        if not events:
            return []

        self.log_func(f"⚡ Applying smart action cooldown filter with a {cooldown_period}s window...")
        
        for event in events:
            event['impact_score'] = event['energy'] + (event.get('spectral_flux', 0) * 0.5)

        events.sort(key=lambda x: x['time'])
        
        significant_events = []
        
        i = 0
        while i < len(events):
            current_event = events[i]
            
            best_event_in_window = current_event
            window_end_time = current_event['time'] + cooldown_period
            
            j = i + 1
            while j < len(events) and events[j]['time'] < window_end_time:
                if events[j]['impact_score'] > best_event_in_window['impact_score']:
                    self.log_func(f"  -> Override: Event at {events[j]['time']:.2f}s (Impact: {events[j]['impact_score']:.2f}) is more impactful than event at {best_event_in_window['time']:.2f}s (Impact: {best_event_in_window['impact_score']:.2f}).")
                    best_event_in_window = events[j]
                j += 1
            
            significant_events.append(best_event_in_window)
            self.log_func(f"  -> Keeping best event in window: {best_event_in_window['time']:.2f}s (Impact: {best_event_in_window['impact_score']:.2f})")
            
            i = j

        self.log_func(f"⚡ Smart cooldown complete. Kept {len(significant_events)} of {len(events)} events.")
        return significant_events

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
        Executes the full multimodal analysis pipeline for video files using Gemini.
        """
        audio_path = None
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING GEMINI MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")

            audio_path = self.file_processor.extract_audio_from_video(video_path, track_index="a:1")

            self.log_func(f"\n📊 PHASE 1: Detecting Precise Audio Onsets")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            significant_events = self._filter_events_with_cooldown(audio_events)

            if not significant_events:
                self.log_func("No significant audio events detected after cooldown. Stopping analysis.")
                return []
            self.log_func(f"✅ Detected {len(significant_events)} significant audio events post-cooldown.")

            self.log_func(f"\n🎬 PHASE 2: Analyzing Video Context with Gemini Pro Vision")
            
            # === UPDATED METHOD CALL: Pass the full event objects to get a linked analysis map ===
            video_analyses_map = self.video_analyzer.analyze_video_at_timestamps(
                video_path, significant_events, window_duration=5.0
            )
            self.log_func(f"✅ Completed Gemini video analysis for {len(video_analyses_map)} timestamps.")
            
            self.log_func(f"\n🔄 PHASE 3: Fusing Audio/Video Intelligence & Generating with Local LLM")
            final_effects = self.fusion_engine.process_multimodal_events(
                significant_events, video_analyses_map
            )
            self.log_func(f"✅ Fusion complete. Generated {len(final_effects)} initial onomatopoeia effects.")

            self.log_func(f"\n🎮 PHASE 4: Applying Gaming Content Optimizations")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(f"✅ Optimization complete. Final effect count: {len(optimized_effects)}.")

            self.log_func(f"\n🎉 GEMINI MULTIMODAL ANALYSIS COMPLETE!")
            return optimized_effects

        except Exception as e:
            self.log_func(f"ERROR during multimodal analysis pipeline: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []
        finally:
            if audio_path:
                self.file_processor.cleanup_temp_file(audio_path)

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Runs a simplified, audio-only version of the pipeline."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING AUDIO-ONLY ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")

            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            significant_events = self._filter_events_with_cooldown(audio_events)
            
            if not significant_events:
                return []
            
            final_effects = self.fusion_engine.process_multimodal_events(significant_events, {})
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