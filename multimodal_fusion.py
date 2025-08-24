# multimodal_fusion.py

"""
Improved version of the multimodal fusion engine.
This version incorporates AI-driven animation selection and animation-specific
timing offsets to ensure the visual peak of an animation aligns with the
audio-visual event peak.
"""

import random
from typing import List, Dict, Tuple
from animations.core import OnomatopoeiaAnimator
from ollama_integration import OllamaLLM


class MultimodalFusionEngine:
    """
    Enhanced fusion engine with precise timing synchronization, including
    AI-driven animation selection and animation-aware offsets.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.audio_weight = 0.6
        self.video_weight = 0.4
        self.min_confidence_threshold = 0.45
        self.local_llm = OllamaLLM(log_func=self.log_func)
        self.animator = OnomatopoeiaAnimator()

    def process_multimodal_events(
        self, audio_events: List[Dict], video_analyses_map: Dict[float, Dict], animation_setting: str
    ) -> List[Dict]:
        self.log_func(
            f"🔄 Fusing {len(audio_events)} audio events with {len(video_analyses_map)} video analyses..."
        )
        final_effects: List[Dict] = []

        # Determine the base animation type from the UI setting
        base_animation_type = self.animator.get_animation_type_from_setting(animation_setting)

        for audio_event in audio_events:
            matching_video = self._find_closest_video_analysis(audio_event, video_analyses_map)

            if matching_video:
                should_generate, effect_decision = self._make_fusion_decision(
                    audio_event, matching_video
                )
                if should_generate:
                    chosen_animation = base_animation_type
                    # If Intelligent mode, ask the AI
                    if base_animation_type == "Intelligent":
                        ai_choice = self.local_llm.choose_animation(
                            onomatopoeia=effect_decision['text'],
                            audio_context=self._get_audio_context(audio_event),
                            video_caption=effect_decision['context']
                        )
                        # Fallback to Random if AI fails
                        chosen_animation = ai_choice or self.animator.get_random_animation_type()

                    # If Random mode, choose one now
                    elif base_animation_type == "Random":
                        chosen_animation = self.animator.get_random_animation_type()


                    final_effects.append(
                        self._create_final_effect(effect_decision, audio_event, chosen_animation)
                    )

        self._log_timing_debug(final_effects)
        self.log_func(f"🎯 Fusion generated {len(final_effects)} effects.")
        return final_effects

    def _find_closest_video_analysis(
        self,
        audio_event: Dict,
        video_analyses_map: Dict[float, Dict],
        max_time_diff: float = 1.0,
    ) -> Dict:
        # ... (This method remains unchanged)
        audio_time = audio_event['time']
        best_match = None
        best_diff = float('inf')
        for video_time, analysis in video_analyses_map.items():
            time_diff = abs(audio_time - video_time)
            if time_diff <= max_time_diff and time_diff < best_diff:
                best_diff = time_diff
                best_match = analysis
        return best_match


    def _make_fusion_decision(
        self, audio_event: Dict, video_analysis: Dict
    ) -> Tuple[bool, Dict]:
        # ... (This method remains unchanged)
        audio_drama = min(audio_event.get('energy', 0.0) / 0.1, 1.0)
        audio_context = self._get_audio_context(audio_event)
        visual_confidence = video_analysis.get('confidence', 0.9)
        video_caption = video_analysis.get('video_caption', '')
        speech_indicators = [
            "talking", "speaking", "says", "conversation", "dialogue", "reacts",
        ]
        strong_action_indicators = [
            "shoot", "shot", "gun", "fire", "punch", "hit", "kick", "crash", "explode",
        ]
        caption_lower = video_caption.lower()
        speech_score = sum(1 for i in speech_indicators if i in caption_lower)
        strong_action_score = sum(1 for i in strong_action_indicators if i in caption_lower)
        if speech_score > 0 and strong_action_score == 0: return False, {}
        final_confidence = (audio_drama * self.audio_weight) + (visual_confidence * self.video_weight)
        if final_confidence < self.min_confidence_threshold: return False, {}
        effect_text = self.local_llm.generate_onomatopoeia(video_caption, audio_context, video_analysis.get('scene_context', set()))
        if not effect_text:
            effect_text = self._fallback_audio_effect(audio_event)
        return True, {"text": effect_text, "confidence": final_confidence, "context": video_caption}


    def _get_audio_context(self, audio_event: Dict) -> str:
        # ... (This method remains unchanged)
        tier = audio_event.get('tier', 'medium')
        onset_type = audio_event.get('onset_type', 'GENERAL')
        energy = audio_event.get('energy', 0.5)
        context = f"A {tier}-impact sound with {energy:.2f} energy, "
        if onset_type == 'LOW_FREQ': context += "deep, low-frequency."
        elif onset_type == 'HIGH_FREQ': context += "sharp, high-frequency."
        else: context += "mid-range frequency."
        return context

    def _fallback_audio_effect(self, audio_event: Dict) -> str:
        energy = audio_event.get('energy', 0.5)
        onset_type = audio_event.get('onset_type', 'GENERAL')

        if onset_type == 'LOW_FREQ':
            return "BOOM" if energy > 0.6 else "THUD"
        elif onset_type == 'HIGH_FREQ':
            return "CRACK" if energy > 0.6 else "CLICK"
        elif onset_type == 'BROADBAND':
            return "CRASH" if energy > 0.6 else "BUMP"
        else: # GENERAL
            if energy > 0.8: return random.choice(["KABOOM!", "CRACK!", "SLAM!"])
            if energy > 0.4: return random.choice(["THUD", "CLICK", "THWACK"])
            return random.choice(["thump", "tick", "tap"])


    def _create_final_effect(
        self, decision: Dict, audio_event: Dict, animation_type: str
    ) -> Dict:
        """
        Creates the final event, applying animation-specific timing.
        """
        precise_peak_time = audio_event.get('peak_time', audio_event['time'])
        # 1. Base audio offset
        audio_offset = -0.15
        duration = 0.7
        # 2. Animation-specific timing offset
        animation_offset = self.animator.get_animation_timing_offset(animation_type)
        # 3. Combine and finalize
        total_offset = audio_offset + animation_offset
        final_start_time = max(0.0, precise_peak_time + total_offset)
        final_end_time = final_start_time + duration

        return {
            'word': decision['text'],
            'start_time': final_start_time,
            'end_time': final_end_time,
            'confidence': decision['confidence'],
            'energy': audio_event.get('energy', 0.0) * decision['confidence'],
            'context': decision['context'],
            'tier': audio_event.get('tier', 'medium'),
            'onset_type': audio_event.get('onset_type', 'GENERAL'),
            'animation_type': animation_type, # Pass the final choice
            'precise_peak_time': precise_peak_time,
            'timing_offset': total_offset,
        }

    def _log_timing_debug(self, effects: List[Dict]) -> None:
        # ... (This method remains unchanged)
        if not effects: return
        self.log_func("\n🎯 PRECISE TIMING DEBUG:")
        for effect in effects:
            word = effect.get('word', 'UNKNOWN')
            peak_time = effect.get('precise_peak_time', 0.0)
            final_time = effect.get('start_time', 0.0)
            offset = effect.get('timing_offset', 0.0)
            anim = effect.get('animation_type', 'N/A')
            self.log_func(
                f"  {word:12} | Peak:{peak_time:6.2f}s | Start:{final_time:6.2f}s | Offset:{offset:+5.2f}s | Anim: {anim}"
            )
        self.log_func("")