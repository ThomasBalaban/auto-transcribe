"""
Cross‑modal onomatopoeia detector that incorporates event verification,
non‑maximum suppression and global cooldown into the timing‑fix version.

This implementation borrows ideas from the provided `multimodal_events.py`:

* Only generates subtitles for audio events that have a matching video
  analysis within a configurable verification window (default 1 s).
* Maintains a short history of recently emitted events and applies
  non‑maximum suppression (NMS) and a global cooldown to avoid
  duplicate or clustered effects.
* Continues to annotate events with tiers, impact scores and sound
  categories as in the previous improved version.

Use this detector in place of the standard `OnomatopoeiaDetector` when
working with highly noisy, game‑like content where cross‑modal
confirmation is necessary for accurate timing.
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
    Cross‑modal onomatopoeia detection system with verification and
    suppression logic.
    """

    # Verification and suppression parameters (seconds)
    VERIFY_WINDOW_SEC: float = 1.0   # window to match audio events with video analyses
    NMS_RADIUS_SEC: float = 0.4      # non‑maximum suppression window
    COOLDOWN_SEC: float = 0.5        # cooldown after any emitted event

    def __init__(self, sensitivity: float = 0.5, device: str = "cpu", log_func=None):
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print

        # Timing synchronization parameters (same as previous version)
        self.audio_video_sync_window = 2.5
        self.impact_cooldown_base = 1.8
        self.dialogue_detection_window = 4.0

        # Event history for NMS/cooldown
        self.event_history: List[Dict[str, float]] = []

        self.log_func("🚀 Initializing Cross‑modal Onomatopoeia Detector...")
        self._initialize_components()
        self.log_func("✅ Cross‑modal multimodal system ready!")

    def _initialize_components(self) -> None:
        try:
            self.log_func("Loading enhanced detection systems...")
            self.onset_detector = GamingOnsetDetector(
                sensitivity=self.sensitivity,
                log_func=self.log_func,
                min_energy_threshold=0.015,
            )
            self.video_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
            self.fusion_engine = MultimodalFusionEngine(log_func=self.log_func)

            self.log_func("Loading enhanced optimization modules...")
            self.gaming_optimizer = GamingOptimizer(
                max_effects_per_minute=8,
                min_effect_spacing=1.8,
                log_func=self.log_func,
            )
            self.subtitle_generator = SubtitleGenerator(log_func=self.log_func)
            self.file_processor = FileProcessor(log_func=self.log_func)
        except Exception as e:
            self.log_func(f"FATAL: Failed to initialize components: {e}")
            raise

    # ---------------------------------------------------------------------
    # Tier classification and scoring
    # ---------------------------------------------------------------------
    def _classify_impact_tier(self, event: Dict) -> int:
        word = event.get('word', '').upper()
        context = event.get('context', '').lower()
        if any(w in word for w in ["GUNSHOT", "EXPLOSION", "DEATH"]) or "violence" in context:
            return 1
        if any(w in word for w in ["PUNCH", "KICK", "CRASH", "FALL"]) or "water" in context:
            return 2
        if any(w in word for w in ["FOOTSTEPS", "LADDER", "AMBIENT"]):
            return 4
        return 3

    def _tier_value_to_name(self, tier_val: int) -> str:
        mapping = {1: 'major', 2: 'medium', 3: 'quick', 4: 'low'}
        return mapping.get(tier_val, 'medium')

    def _calculate_enhanced_impact_score(self, event: Dict) -> float:
        energy = event.get('energy', 0.5)
        spectral_flux = event.get('spectral_flux', 0.5)
        tier = event.get('tier', 'medium')
        onset_type = event.get('onset_type', 'GENERAL')
        confidence = event.get('confidence', 0.5)
        base_score = energy * 2.0 + spectral_flux * 1.5
        tier_multipliers = {'major': 1.5, 'medium': 1.0, 'quick': 0.7, 'low': 0.5}
        tier_mult = tier_multipliers.get(tier, 1.0)
        onset_bonuses = {
            'LOW_FREQ': 1.2,
            'HIGH_FREQ': 1.1,
            'BROADBAND': 1.3,
            'GENERAL': 1.0,
        }
        onset_bonus = onset_bonuses.get(onset_type, 1.0)
        confidence_factor = 0.7 + (confidence * 0.3)
        return base_score * tier_mult * onset_bonus * confidence_factor

    def _categorize_sound(self, event: Dict) -> str:
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

    # ---------------------------------------------------------------------
    # Hierarchical cooldown
    # ---------------------------------------------------------------------
    def _filter_events_with_enhanced_cooldown(self, events: List[Dict]) -> List[Dict]:
        if not events:
            return []
        events.sort(key=lambda x: x['time'])
        final_events: List[Dict] = []
        last_event_time = {1: -999.0, 2: -999.0, 3: -999.0, 4: -999.0}
        cooldowns = {1: 10.0, 2: 5.0, 3: 2.5, 4: 1.0}
        for event in events:
            tier_val = event['tier_val']
            t = event['time']
            on_cooldown = False
            for i in range(1, tier_val + 1):
                if t - last_event_time[i] < cooldowns[i]:
                    on_cooldown = True
                    break
            if not on_cooldown:
                final_events.append(event)
                for j in range(tier_val, 5):
                    last_event_time[j] = t
        return final_events

    # ---------------------------------------------------------------------
    # Cross‑modal verification and suppression
    # ---------------------------------------------------------------------
    def _verify_and_filter_events(
        self, events: List[Dict], video_map: Dict[float, Dict]
    ) -> List[Dict]:
        """Matches audio events with video analyses and applies NMS/cooldown."""
        verified: List[Dict] = []
        # Iterate in time order
        for event in sorted(events, key=lambda e: e['time']):
            # Find a video analysis within VERIFY_WINDOW_SEC
            matching_analysis = None
            for v_time, analysis in video_map.items():
                if abs(event['time'] - v_time) <= self.VERIFY_WINDOW_SEC:
                    matching_analysis = analysis
                    break
            if not matching_analysis:
                continue  # skip events with no visual corroboration
            # Compute combined score from audio confidence and visual confidence
            audio_conf = event.get('confidence', 0.5)
            visual_conf = matching_analysis.get('confidence', 0.9)
            combined_score = (audio_conf + visual_conf) / 2.0
            # Non‑maximum suppression: skip if too close to a better event
            skip = False
            for existing in verified:
                if abs(event['time'] - existing['time']) < self.NMS_RADIUS_SEC and existing.get('score', 0) >= combined_score:
                    skip = True
                    break
            if skip:
                continue
            # Global cooldown: skip if within COOLDOWN_SEC of history
            for hist in self.event_history:
                if 0 < event['time'] - hist['time'] < self.COOLDOWN_SEC:
                    skip = True
                    break
            if skip:
                continue
            # Annotate score and intensity (use energy as proxy for intensity)
            event['score'] = combined_score
            event['intensity'] = combined_score
            verified.append(event)
            # Update history and truncate
            self.event_history.append({'time': event['time'], 'score': combined_score})
            self.event_history = self.event_history[-50:]
        return verified

    # ---------------------------------------------------------------------
    # Analysis routines
    # ---------------------------------------------------------------------
    def _analyze_video_file(self, video_path: str) -> List[Dict]:
        audio_path = None
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING CROSS‑MODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")
            audio_path = self.file_processor.extract_audio_from_video(video_path, track_index="a:1")
            self.log_func("\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            # Annotate each audio event
            for ev in audio_events:
                tier_val = self._classify_impact_tier(ev)
                ev['tier_val'] = tier_val
                ev['tier'] = self._tier_value_to_name(tier_val)
                ev['impact_score'] = self._calculate_enhanced_impact_score(ev)
                ev['sound_category'] = self._categorize_sound(ev)
            # Apply hierarchical cooldown to raw audio events
            significant_events = self._filter_events_with_enhanced_cooldown(audio_events)
            if not significant_events:
                self.log_func("No significant audio events detected.")
                return []
            self.log_func(f"✅ Detected {len(significant_events)} significant audio events.")
            self.log_func("\n🎬 PHASE 2: Video Analysis")
            video_map = self._create_synchronized_video_analysis(video_path, significant_events)
            self.log_func(f"✅ Generated video analyses for {len(video_map)} timestamps.")
            # Cross‑modal verification and suppression
            verified_events = self._verify_and_filter_events(significant_events, video_map)
            if not verified_events:
                self.log_func("No events survived cross‑modal verification and suppression.")
                return []
            self.log_func(f"✅ {len(verified_events)} events verified across modalities.")
            self.log_func("\n🔄 PHASE 3: Multimodal Fusion")
            final_effects = self.fusion_engine.process_multimodal_events(verified_events, video_map)
            self.log_func(f"✅ Fusion complete. Generated {len(final_effects)} effects.")
            self.log_func("\n🎮 PHASE 4: Gaming Optimizations")
            optimized_effects = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(f"✅ Final optimization complete. Effect count: {len(optimized_effects)}.")
            self.log_func(f"\n🎉 CROSS‑MODAL ANALYSIS COMPLETE!")
            return optimized_effects
        except Exception as e:
            self.log_func(f"ERROR during cross‑modal analysis: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []
        finally:
            if audio_path:
                self.file_processor.cleanup_temp_file(audio_path)

    def _create_synchronized_video_analysis(self, video_path: str, audio_events: List[Dict]) -> Dict[float, Dict]:
        # Group events into 0.5 s windows and analyse around the most significant in each group
        groups = self._group_nearby_events(audio_events, max_group_span=0.5)
        video_map: Dict[float, Dict] = {}
        for group in groups:
            primary_event = max(group, key=lambda e: e.get('impact_score', 0))
            t = primary_event['time']
            analysis = self.video_analyzer.analyze_video_at_timestamps(video_path, [primary_event], window_duration=5.0)
            if analysis:
                for evt in group:
                    video_map[evt['time']] = analysis[t]
        return video_map

    def _group_nearby_events(self, events: List[Dict], max_group_span: float) -> List[List[Dict]]:
        if not events:
            return []
        sorted_events = sorted(events, key=lambda e: e['time'])
        groups: List[List[Dict]] = []
        current_group: List[Dict] = [sorted_events[0]]
        for ev in sorted_events[1:]:
            if ev['time'] - current_group[0]['time'] <= max_group_span:
                current_group.append(ev)
            else:
                groups.append(current_group)
                current_group = [ev]
        groups.append(current_group)
        return groups

    # Public API
    def analyze_file(self, input_path: str) -> List[Dict]:
        file_type = self.file_processor.detect_file_type(input_path)
        if file_type == 'video':
            return self._analyze_video_file(input_path)
        elif file_type == 'audio':
            return self._analyze_audio_file(input_path)
        else:
            self.log_func(f"Unsupported file type: {os.path.splitext(input_path)[1]}")
            return []

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        # Cross‑modal verification doesn't apply in audio‑only mode; just reuse improved logic
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"STARTING CROSS‑MODAL AUDIO ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            for ev in audio_events:
                tier_val = self._classify_impact_tier(ev)
                ev['tier_val'] = tier_val
                ev['tier'] = self._tier_value_to_name(tier_val)
                ev['impact_score'] = self._calculate_enhanced_impact_score(ev)
                ev['sound_category'] = self._categorize_sound(ev)
            significant_events = self._filter_events_with_enhanced_cooldown(audio_events)
            if not significant_events:
                return []
            final_effects: List[Dict] = []
            for ev in significant_events:
                word = self.fusion_engine._fallback_audio_effect(ev)
                final_effects.append({
                    'word': word,
                    'start_time': ev['time'],
                    'end_time': ev['time'] + 1.0,
                    'confidence': ev.get('confidence', 0.5),
                    'energy': ev.get('energy', 0.5),
                    'context': 'audio-only analysis',
                    'tier': ev['tier'],
                    'onset_type': ev.get('onset_type', 'GENERAL'),
                })
            optimized = self.gaming_optimizer.apply_gaming_optimizations(final_effects)
            self.log_func(
                f"\n🎉 CROSS‑MODAL AUDIO ANALYSIS COMPLETE! Found {len(optimized)} effects."
            )
            return optimized
        except Exception as e:
            self.log_func(f"Error in cross‑modal audio analysis: {e}")
            return []

    def create_subtitle_file(self, input_path: str, output_path: str, animation_type: str = "Random") -> Tuple[bool, List[Dict]]:
        try:
            events = self.analyze_file(input_path)
            if not events:
                self.log_func("No events detected for subtitle generation.")
                return False, []
            success = self.subtitle_generator.create_subtitle_file(events, output_path, animation_type)
            return success, events
        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            return False, []
