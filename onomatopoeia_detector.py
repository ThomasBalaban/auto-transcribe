# onomatopoeia_detector.py - TIMING FIXES

import os
from typing import List, Dict, Tuple

from onset_detector import GamingOnsetDetector
from gemini_vision_analyzer import GeminiVisionAnalyzer 
from multimodal_fusion import MultimodalFusionEngine
from gaming_optimizer import GamingOptimizer
from subtitle_generator import SubtitleGenerator
from file_processor import FileProcessor


class OnomatopoeiaDetector:
    """
    Enhanced onomatopoeia detection system with improved timing synchronization.
    """

    def __init__(self, sensitivity: float = 0.5, device: str = "cpu", log_func=None):
        """Initialize the detection system with enhanced timing."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print
        
        # Timing synchronization parameters
        self.audio_video_sync_window = 2.5  # seconds - how far apart audio/video can be
        self.impact_cooldown_base = 1.8     # base cooldown between major impacts
        self.dialogue_detection_window = 4.0 # seconds - window for speech context

        self.log_func("🚀 Initializing Enhanced Onomatopoeia Detector...")
        self._initialize_components()
        self.log_func("✅ Enhanced multimodal system ready!")

    def _initialize_components(self):
        """Initialize all detection components."""
        try:
            self.log_func("Loading enhanced detection systems...")
            self.onset_detector = GamingOnsetDetector(
                sensitivity=self.sensitivity, 
                log_func=self.log_func,
                min_energy_threshold=0.015  # Slightly more sensitive
            )
            self.video_analyzer = GeminiVisionAnalyzer(log_func=self.log_func) 
            self.fusion_engine = MultimodalFusionEngine(log_func=self.log_func)

            self.log_func("Loading enhanced optimization modules...")
            self.gaming_optimizer = GamingOptimizer(
                max_effects_per_minute=8,  # Reduced from 10
                min_effect_spacing=1.8,    # Increased from 1.5
                log_func=self.log_func
            )
            self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
            self.file_processor = FileProcessor(log_func=self.log_func)

        except Exception as e:
            self.log_func(f"FATAL: Failed to initialize components: {e}")
            raise

    def _filter_events_with_enhanced_cooldown(self, events: List[Dict]) -> List[Dict]:
        """
        Enhanced cooldown system that adapts to sound types and energy levels.
        """
        if not events:
            return []

        self.log_func(f"⚡ Applying enhanced smart cooldown with adaptive timing...")
        
        # Calculate impact scores and add timing metadata
        for event in events:
            event['impact_score'] = self._calculate_enhanced_impact_score(event)
            event['sound_category'] = self._categorize_sound(event)

        events.sort(key=lambda x: x['time'])
        
        significant_events = []
        last_major_impact_time = -999
        
        for event in events:
            current_time = event['time']
            sound_category = event['sound_category']
            impact_score = event['impact_score']
            
            # Calculate adaptive cooldown based on sound type and previous events
            cooldown_period = self._calculate_adaptive_cooldown(
                event, significant_events, last_major_impact_time
            )
            
            # Check if enough time has passed since last significant event
            if significant_events:
                time_since_last = current_time - significant_events[-1]['time']
                
                # For similar sounds, require longer cooldown
                if self._sounds_are_similar(event, significant_events[-1]):
                    required_cooldown = cooldown_period * 1.5
                else:
                    required_cooldown = cooldown_period
                
                if time_since_last < required_cooldown:
                    # Check if current event is significantly more impactful
                    last_impact = significant_events[-1]['impact_score']
                    if impact_score > last_impact * 1.3:  # 30% more impactful
                        self.log_func(f"  -> OVERRIDE: {event['time']:.2f}s "
                                    f"({sound_category}, impact: {impact_score:.2f}) "
                                    f"overrides {significant_events[-1]['time']:.2f}s "
                                    f"(impact: {last_impact:.2f})")
                        significant_events[-1] = event
                        if impact_score > 2.0:  # High impact event
                            last_major_impact_time = current_time
                    else:
                        self.log_func(f"  -> SKIP: {current_time:.2f}s "
                                    f"({sound_category}) too close to previous event "
                                    f"({time_since_last:.2f}s < {required_cooldown:.2f}s)")
                    continue
            
            # Event passes cooldown check
            significant_events.append(event)
            self.log_func(f"  -> KEEP: {current_time:.2f}s "
                         f"({sound_category}, impact: {impact_score:.2f})")
            
            # Update major impact tracking
            if impact_score > 2.0:
                last_major_impact_time = current_time

        self.log_func(f"⚡ Enhanced cooldown complete. Kept {len(significant_events)} of {len(events)} events.")
        return significant_events

    def _calculate_enhanced_impact_score(self, event: Dict) -> float:
        """Calculate enhanced impact score with multiple factors."""
        energy = event.get('energy', 0.5)
        spectral_flux = event.get('spectral_flux', 0.5)
        tier = event.get('tier', 'medium')
        onset_type = event.get('onset_type', 'GENERAL')
        confidence = event.get('confidence', 0.5)
        
        # Base score from energy and spectral characteristics
        base_score = energy * 2.0 + spectral_flux * 1.5
        
        # Tier multiplier
        tier_multipliers = {'major': 1.5, 'medium': 1.0, 'quick': 0.7}
        tier_mult = tier_multipliers.get(tier, 1.0)
        
        # Onset type bonus
        onset_bonuses = {
            'LOW_FREQ': 1.2,    # Explosions, impacts
            'HIGH_FREQ': 1.1,   # Gunshots, metal
            'BROADBAND': 1.3,   # Complex crashes
            'GENERAL': 1.0
        }
        onset_bonus = onset_bonuses.get(onset_type, 1.0)
        
        # Confidence factor
        confidence_factor = 0.7 + (confidence * 0.3)  # Scale from 0.7 to 1.0
        
        final_score = base_score * tier_mult * onset_bonus * confidence_factor
        return final_score

    def _categorize_sound(self, event: Dict) -> str:
        """Categorize sound for better cooldown management."""
        onset_type = event.get('onset_type', 'GENERAL')
        energy = event.get('energy', 0.5)
        tier = event.get('tier', 'medium')
        
        if tier == 'major' and energy > 0.1:
            if onset_type == 'LOW_FREQ':
                return 'EXPLOSION'
            elif onset_type == 'HIGH_FREQ':
                return 'GUNSHOT'
            else:
                return 'MAJOR_IMPACT'
        elif onset_type == 'BROADBAND':
            return 'CRASH'
        elif energy < 0.05:
            return 'SUBTLE'
        else:
            return 'IMPACT'

    def _calculate_adaptive_cooldown(self, event: Dict, recent_events: List[Dict], 
                                   last_major_time: float) -> float:
        """Calculate adaptive cooldown period based on context."""
        base_cooldown = self.impact_cooldown_base
        sound_category = event['sound_category']
        current_time = event['time']
        
        # Category-specific cooldowns
        category_cooldowns = {
            'EXPLOSION': 2.5,
            'GUNSHOT': 1.5,
            'MAJOR_IMPACT': 2.0,
            'CRASH': 1.8,
            'IMPACT': 1.2,
            'SUBTLE': 0.8
        }
        
        category_cooldown = category_cooldowns.get(sound_category, base_cooldown)
        
        # Reduce cooldown if it's been a while since last major impact
        time_since_major = current_time - last_major_time
        if time_since_major > 10.0:  # Long quiet period
            category_cooldown *= 0.7
        elif time_since_major > 5.0:  # Medium quiet period
            category_cooldown *= 0.85
        
        # Increase cooldown if many recent events
        if len(recent_events) >= 3:
            recent_window = [e for e in recent_events[-3:] if current_time - e['time'] < 10.0]
            if len(recent_window) >= 2:
                category_cooldown *= 1.3
        
        return category_cooldown

    def _sounds_are_similar(self, event1: Dict, event2: Dict) -> bool:
        """Check if two sounds are similar and might be repetitive."""
        # Same category suggests similar sounds
        if event1.get('sound_category') == event2.get('sound_category'):
            # Check onset type similarity
            if event1.get('onset_type') == event2.get('onset_type'):
                # Check energy similarity (within 30%)
                energy1 = event1.get('energy', 0.5)
                energy2 = event2.get('energy', 0.5)
                energy_ratio = min(energy1, energy2) / (max(energy1, energy2) + 1e-8)
                if energy_ratio > 0.7:  # Similar energy levels
                    return True
        return False

    def _analyze_video_file(self, video_path: str) -> List[Dict]:
        """Enhanced video analysis with better timing synchronization."""
        audio_path = None
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING ENHANCED MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")

            # Extract audio
            audio_path = self.file_processor.extract_audio_from_video(video_path, track_index="a:1")

            self.log_func(f"\n📊 PHASE 1: Enhanced Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            # Apply enhanced cooldown filtering
            significant_events = self._filter_events_with_enhanced_cooldown(audio_events)

            if not significant_events:
                self.log_func("No significant audio events detected after enhanced filtering.")
                return []
            
            self.log_func(f"✅ Detected {len(significant_events)} significant audio events.")

            self.log_func(f"\n🎬 PHASE 2: Synchronized Video Analysis")
            
            # Create analysis map with better timing synchronization
            video_analyses_map = self._create_synchronized_video_analysis(
                video_path, significant_events
            )
            
            self.log_func(f"✅ Completed synchronized video analysis for {len(video_analyses_map)} timestamps.")
            
            self.log_func(f"\n🔄 PHASE 3: Enhanced Multimodal Fusion")
            final_effects = self.fusion_engine.process_multimodal_events(
                significant_events, video_analyses_map
            )
            self.log_func(f"✅ Fusion complete. Generated {len(final_effects)} effects.")

            self.log_func(f"\n🎮 PHASE 4: Enhanced Gaming Optimizations")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(f"✅ Final optimization complete. Effect count: {len(optimized_effects)}.")

            self.log_func(f"\n🎉 ENHANCED MULTIMODAL ANALYSIS COMPLETE!")
            return optimized_effects

        except Exception as e:
            self.log_func(f"ERROR during enhanced analysis: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []
        finally:
            if audio_path:
                self.file_processor.cleanup_temp_file(audio_path)

    def _create_synchronized_video_analysis(self, video_path: str, 
                                          audio_events: List[Dict]) -> Dict[float, Dict]:
        """Create video analysis with improved timing synchronization."""
        # Group nearby audio events to reduce video analysis calls
        event_groups = self._group_nearby_events(audio_events, max_group_span=3.0)
        
        video_analyses_map = {}
        
        for group in event_groups:
            # Use the most significant event in the group as the timing reference
            primary_event = max(group, key=lambda e: e['impact_score'])
            analysis_time = primary_event['time']
            
            # Analyze video around this time
            single_analysis = self.video_analyzer.analyze_video_at_timestamps(
                video_path, [primary_event], window_duration=5.0
            )
            
            if single_analysis:
                # Apply this analysis to all events in the group
                for event in group:
                    video_analyses_map[event['time']] = single_analysis[analysis_time]
        
        return video_analyses_map

    def _group_nearby_events(self, events: List[Dict], max_group_span: float = 3.0) -> List[List[Dict]]:
        """Group nearby events to optimize video analysis."""
        if not events:
            return []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e['time'])
        
        groups = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            # Check if event should be in current group
            group_start_time = current_group[0]['time']
            if event['time'] - group_start_time <= max_group_span:
                current_group.append(event)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [event]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups

    def analyze_file(self, input_path: str) -> List[Dict]:
        """Main analysis method with enhanced processing."""
        file_type = self.file_processor.detect_file_type(input_path)

        if file_type == 'video':
            return self._analyze_video_file(video_path=input_path)
        elif file_type == 'audio':
            return self._analyze_audio_file(audio_path=input_path)
        else:
            self.log_func(f"Unsupported file type: {os.path.splitext(input_path)[1]}")
            return []

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Enhanced audio-only analysis."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING ENHANCED AUDIO-ONLY ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")

            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            significant_events = self._filter_events_with_enhanced_cooldown(audio_events)
            
            if not significant_events:
                return []
            
            # For audio-only, use simpler fusion without video context
            final_effects = []
            for event in significant_events:
                # Create basic effect without video context
                word = self.fusion_engine._fallback_audio_effect(event)
                effect = {
                    'word': word,
                    'start_time': event['time'],
                    'end_time': event['time'] + 1.0,
                    'confidence': event.get('confidence', 0.5),
                    'energy': event.get('energy', 0.5),
                    'context': 'audio-only analysis'
                }
                final_effects.append(effect)
            
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)

            self.log_func(f"\n🎉 ENHANCED AUDIO-ONLY ANALYSIS COMPLETE! Found {len(optimized_effects)} effects.")
            return optimized_effects

        except Exception as e:
            self.log_func(f"Error in enhanced audio-only analysis: {e}")
            return []

    def create_subtitle_file(self, input_path: str, output_path: str,
                           animation_type: str = "Random") -> Tuple[bool, List[Dict]]:
        """Enhanced subtitle file creation with better timing."""
        try:
            events = self.analyze_file(input_path)
            if not events:
                self.log_func("No onomatopoeia events detected for subtitle generation.")
                return False, []

            success = self.subtitle_generator.create_subtitle_file(
                events, output_path, animation_type
            )
            return success, events

        except Exception as e:
            self.log_func(f"Error creating enhanced subtitle file: {e}")
            return False, []