# multimodal_fusion.py

import numpy as np
from typing import List, Dict, Tuple
import random

class MultimodalFusionEngine:
    """
    Fuses audio and video analysis with final polished logic for timing,
    context, and confidence.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.audio_weight = 0.55
        self.video_weight = 0.45
        # Lowered threshold one last time to be a bit more permissive
        self.min_confidence_threshold = 0.42

    def process_multimodal_events(self, audio_events: List[Dict],
                                video_analyses: List[Dict]) -> List[Dict]:
        self.log_func(f"🔄 Fusing {len(audio_events)} audio events with {len(video_analyses)} video analyses...")
        final_effects = []
        for audio_event in audio_events:
            audio_time = audio_event.get('time', 0)
            matching_video = self._find_matching_video_analysis(audio_time, video_analyses)
            should_generate, effect_decision = self._make_fusion_decision(audio_event, matching_video)
            if should_generate:
                final_effects.append(self._create_final_effect(effect_decision, audio_event))
        self.log_func(f"🎯 Fusion generated {len(final_effects)} effects.")
        return final_effects

    def _find_matching_video_analysis(self, audio_time: float, video_analyses: List[Dict]) -> Dict:
        for analysis in video_analyses:
            if analysis['start_time'] <= audio_time <= (analysis['start_time'] + analysis['duration']):
                return analysis
        return None

    def _make_fusion_decision(self, audio_event: Dict, video_analysis: Dict) -> Tuple[bool, Dict]:
        audio_drama = min(audio_event.get('energy', 0) / 0.1, 1.0)

        if not video_analysis:
            if audio_drama > 0.7:
                # Use peak_time for timing
                return True, {"text": self._fallback_audio_effect(audio_event), "timing": audio_event['peak_time'], "confidence": audio_drama}
            return False, {}

        visual_action_score = video_analysis.get('action_score', 0)
        action = video_analysis['action_classification']['primary_action']
        
        base_score = (audio_drama * self.audio_weight) + (visual_action_score * self.video_weight)
        
        context_bonus = 0.0
        high_impact_actions = ["punching", "explosion", "gunfire", "damage", "breaking", "shattering", "underwater"]
        low_impact_actions = ["breathing", "calm"]

        if any(keyword in action for keyword in high_impact_actions): context_bonus = 0.25
        elif any(keyword in action for keyword in low_impact_actions): context_bonus = -0.1
        if audio_event.get('tier') == 'major': context_bonus += 0.1

        final_confidence = base_score + context_bonus

        if final_confidence < self.min_confidence_threshold:
            self.log_func(f"📉 Event at {audio_event['time']:.2f}s failed. Score: {final_confidence:.2f} (Base: {base_score:.2f}, Bonus: {context_bonus:+.2f})")
            return False, {}

        effect_text = self._select_video_driven_effect(action, audio_event)
        if not effect_text: return False, {}

        self.log_func(f"✅ Event at {audio_event['time']:.2f}s PASSED. Score: {final_confidence:.2f} (Base: {base_score:.2f}, Bonus: {context_bonus:+.2f})")
        return True, {
            "text": effect_text,
            "timing": audio_event['peak_time'], # Use the peak time for perfect timing
            "confidence": final_confidence,
            "context": action
        }

    def _select_video_driven_effect(self, action: str, audio_event: Dict) -> str:
        action_map = {
            "kicking or punching": ["WHAM!", "POW!", "SMACK!"],
            "explosion or a large blast": ["KABOOM!", "BOOM!"],
            "gunfire from a weapon": ["BANG!", "BLAM!", "CRACK!"],
            "being hit or taking damage": ["THUD!", "WHACK!", "OOF!"],
            "breaking or shattering": ["CRASH!", "SHATTER!"],
            "fighting underwater": ["BLUB!", "GLUG!", "FWOOSH!"], # Specific underwater words
            "falling into water": ["SPLASH!", "SPLISH!"],
            "emerging from water": ["FWOOSH!", "SPLASH!"],
        }
        for keyword, words in action_map.items():
            if keyword in action:
                return random.choice(words)
        if audio_event.get('energy', 0) > 0.06:
             return self._fallback_audio_effect(audio_event)
        return None

    def _fallback_audio_effect(self, audio_event: Dict) -> str:
        onset_type = audio_event.get('onset_type', 'GENERAL')
        if onset_type == 'LOW_FREQ': return "THOOM"
        if onset_type == 'HIGH_FREQ': return "CRACK"
        return "BAM"

    def _create_final_effect(self, decision: Dict, audio_event: Dict) -> Dict:
        return {
            'word': decision['text'],
            'start_time': decision['timing'], # This is now the peak_time
            'end_time': decision['timing'] + random.uniform(0.7, 1.5),
            'confidence': decision['confidence'],
            'energy': audio_event.get('energy', 0) * decision['confidence'],
            'context': decision['context'],
        }