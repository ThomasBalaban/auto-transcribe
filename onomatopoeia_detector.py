"""
Improved version of the onomatopoeia detector with better tier handling,
adaptive cooldown propagation and richer event annotation.  This file
extends the timing‑fix implementation supplied by the user by addressing
several logical gaps:

* When a high‑tier event occurs it now updates the last seen time for
  all lower tiers, ensuring that minor sounds don’t immediately follow
  a major one.
* Each detected audio event is annotated with a human readable
  ``tier`` (major/medium/quick/low), an ``impact_score`` and a
  ``sound_category`` prior to any filtering.  This unifies the
  attributes consumed by downstream modules.
* The synchronized video analysis uses the computed ``impact_score``
  for grouping and timing reference.
* Grouping of events defaults to a tighter window (0.5 s) to prevent
  unrelated sounds from sharing the same video analysis.

This file can be dropped in place of the original ``onomatopoeia_detector.py``
for improved behaviour without changing the external API.
"""

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
    Enhanced onomatopoeia detection system with improved timing synchronization
    and richer event annotation.
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

    def _initialize_components(self) -> None:
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

    # -------------------------------------------------------------------------
    # Utility helpers for tier and classification
    # -------------------------------------------------------------------------
    def _classify_impact_tier(self, event: Dict) -> int:
        """Classifies an event into one of four impact tiers (1 is most severe)."""
        word = event.get('word', '').upper()
        context = event.get('context', '').lower()

        # Tier 1: Critical
        if any(w in word for w in ["GUNSHOT", "EXPLOSION", "DEATH"]) or "violence" in context:
            return 1
        # Tier 2: High
        if any(w in word for w in ["PUNCH", "KICK", "CRASH", "FALL"]) or "water" in context:
            return 2
        # Tier 4: Low
        if any(w in word for w in ["FOOTSTEPS", "LADDER", "AMBIENT"]):
            return 4
        # Tier 3: Medium (Default)
        return 3

    def _tier_value_to_name(self, tier_val: int) -> str:
        """Map integer tier values to human readable names used in scoring."""
        mapping = {1: 'major', 2: 'medium', 3: 'quick', 4: 'low'}
        return mapping.get(tier_val, 'medium')

    # -------------------------------------------------------------------------
    # Event scoring and categorization
    # -------------------------------------------------------------------------
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
        tier_multipliers = {'major': 1.5, 'medium': 1.0, 'quick': 0.7, 'low': 0.5}
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

    # -------------------------------------------------------------------------
    # Filtering and cooldown logic
    # -------------------------------------------------------------------------
    def _filter_events_with_enhanced_cooldown(self, events: List[Dict]) -> List[Dict]:
        """
        Applies a hierarchical cooldown system.  When an event of a given tier
        occurs, the cooldown timestamp for that tier **and all lower tiers** is
        updated so that minor events don't immediately follow major ones.
        """
        if not events:
            return []

        self.log_func("Applying hierarchical impact cooldowns...")

        # Already annotated events contain tier information.
        events.sort(key=lambda x: x['time'])

        final_events: List[Dict] = []
        last_event_time = {1: -999.0, 2: -999.0, 3: -999.0, 4: -999.0}
        cooldowns = {1: 10.0, 2: 5.0, 3: 2.5, 4: 1.0}

        for event in events:
            tier_val: int = event['tier_val']
            time: float = event['time']

            # Check cooldown from any higher or equal tier
            on_cooldown = False
            for i in range(1, tier_val + 1):
                if time - last_event_time[i] < cooldowns[i]:
                    self.log_func(
                        f"-> SKIP (Tier {tier_val}): '{event.get('word', 'N/A')}' at {time:.2f}s is on cooldown."
                    )
                    on_cooldown = True
                    break

            if not on_cooldown:
                final_events.append(event)
                # Update last seen time for this tier and all lower tiers
                for t in range(tier_val, 5):
                    last_event_time[t] = time

        self.log_func(
            f"Hierarchical cooldown complete. Kept {len(final_events)} of {len(events)} events."
        )
        return final_events

    # -------------------------------------------------------------------------
    # Core analysis routines
    # -------------------------------------------------------------------------
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

            # Annotate events with tier, tier_val, impact_score and sound_category
            for ev in audio_events:
                tier_val = self._classify_impact_tier(ev)
                ev['tier_val'] = tier_val
                ev['tier'] = self._tier_value_to_name(tier_val)
                ev['impact_score'] = self._calculate_enhanced_impact_score(ev)
                ev['sound_category'] = self._categorize_sound(ev)

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

            self.log_func(
                f"✅ Completed synchronized video analysis for {len(video_analyses_map)} timestamps."
            )

            self.log_func(f"\n🔄 PHASE 3: Enhanced Multimodal Fusion")
            final_effects = self.fusion_engine.process_multimodal_events(
                significant_events, video_analyses_map
            )
            self.log_func(
                f"✅ Fusion complete. Generated {len(final_effects)} effects."
            )

            self.log_func(f"\n🎮 PHASE 4: Enhanced Gaming Optimizations")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(
                f"✅ Final optimization complete. Effect count: {len(optimized_effects)}."
            )

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

    def _create_synchronized_video_analysis(
        self, video_path: str, audio_events: List[Dict]
    ) -> Dict[float, Dict]:
        """Create video analysis with improved timing synchronization."""
        # Group nearby audio events to reduce video analysis calls
        event_groups = self._group_nearby_events(audio_events, max_group_span=0.5)

        video_analyses_map: Dict[float, Dict] = {}

        for group in event_groups:
            # Use the most significant event in the group as the timing reference
            primary_event = max(group, key=lambda e: e.get('impact_score', 0))
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

        groups: List[List[Dict]] = []
        current_group: List[Dict] = [sorted_events[0]]

        for event in sorted_events[1:]:
            group_start_time = current_group[0]['time']
            if event['time'] - group_start_time <= max_group_span:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]

        if current_group:
            groups.append(current_group)

        return groups

    # Public API
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

            # Annotate audio events as in video mode
            for ev in audio_events:
                tier_val = self._classify_impact_tier(ev)
                ev['tier_val'] = tier_val
                ev['tier'] = self._tier_value_to_name(tier_val)
                ev['impact_score'] = self._calculate_enhanced_impact_score(ev)
                ev['sound_category'] = self._categorize_sound(ev)

            significant_events = self._filter_events_with_enhanced_cooldown(audio_events)

            if not significant_events:
                return []

            # For audio-only, use simpler fusion without video context
            final_effects: List[Dict] = []
            for event in significant_events:
                word = self.fusion_engine._fallback_audio_effect(event)
                effect = {
                    'word': word,
                    'start_time': event['time'],
                    'end_time': event['time'] + 1.0,
                    'confidence': event.get('confidence', 0.5),
                    'energy': event.get('energy', 0.5),
                    'context': 'audio-only analysis',
                    'tier': event['tier'],
                    'onset_type': event.get('onset_type', 'GENERAL')
                }
                final_effects.append(effect)

            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)

            self.log_func(
                f"\n🎉 ENHANCED AUDIO-ONLY ANALYSIS COMPLETE! Found {len(optimized_effects)} effects."
            )
            return optimized_effects

        except Exception as e:
            self.log_func(f"Error in enhanced audio-only analysis: {e}")
            return []

    def create_subtitle_file(
        self,
        input_path: str,
        output_path: str,
        animation_type: str = "Random",
    ) -> Tuple[bool, List[Dict]]:
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
