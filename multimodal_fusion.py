# multimodal_fusion.py

import numpy as np
from typing import List, Dict, Tuple
import random
from ollama_integration import OllamaLLM

class MultimodalFusionEngine:
    """
    Fuses audio analysis and Gemini vision captions, then uses a local LLM
    for the final onomatopoeia generation.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.audio_weight = 0.6
        self.video_weight = 0.4
        self.min_confidence_threshold = 0.45
        self.local_llm = OllamaLLM(log_func=self.log_func)

    def process_multimodal_events(self, audio_events: List[Dict],
                                video_analyses_map: Dict[float, Dict]) -> List[Dict]:
        self.log_func(f"🔄 Fusing {len(audio_events)} audio events with {len(video_analyses_map)} video analyses...")
        final_effects = []
        
        for audio_event in audio_events:
            matching_video = video_analyses_map.get(audio_event['time'])
            
            if matching_video:
                should_generate, effect_decision = self._make_fusion_decision(audio_event, matching_video)
                if should_generate:
                    final_effects.append(self._create_final_effect(effect_decision, audio_event))

        self.log_func(f"🎯 Fusion generated {len(final_effects)} effects.")
        return final_effects

    def _make_fusion_decision(self, audio_event: Dict, video_analysis: Dict) -> Tuple[bool, Dict]:
        audio_drama = min(audio_event.get('energy', 0) / 0.1, 1.0)
        audio_context = self._get_audio_context(audio_event)

        visual_confidence = video_analysis.get('confidence', 0.9)
        video_caption = video_analysis.get('video_caption', '')
        scene_context = video_analysis.get('scene_context', set())

        # === FINAL UPDATE: Stricter speech and action detection ===
        speech_keywords = ["talking", "speaking", "narrating", "shouting", "whispering", "saying", "character speaks", "tells", "conversation", "dialogue", "expresses"]
        action_keywords = ["climb", "fall", "shoot", "jump", "perform", "struggle", "examine", "retrieve", "gunpoint", "interact", "hit", "punch", "kick"]

        is_speech = any(keyword in video_caption.lower() for keyword in speech_keywords)
        is_action = any(keyword in video_caption.lower() for keyword in action_keywords)

        # If speech is detected AND no clear action is present, reject the event.
        if is_speech and not is_action:
            self.log_func(f"🤫 Event at {audio_event['time']:.2f}s REJECTED. Caption indicates dialogue with no overriding action: '{video_caption}'")
            return False, {}

        final_confidence = (audio_drama * self.audio_weight) + (visual_confidence * self.video_weight)

        if final_confidence < self.min_confidence_threshold:
            self.log_func(f"📉 Event at {audio_event['time']:.2f}s failed confidence check. Score: {final_confidence:.2f}")
            return False, {}

        effect_text = self.local_llm.generate_onomatopoeia(video_caption, audio_context, scene_context)
        
        if not effect_text:
            self.log_func(f"⚠️ Local LLM failed to generate onomatopoeia. Using fallback.")
            effect_text = self._fallback_audio_effect(audio_event)

        self.log_func(f"✅ Event at {audio_event['time']:.2f}s PASSED. Score: {final_confidence:.2f}, Caption: '{video_caption}', Effect: '{effect_text}'")
        return True, {
            "text": effect_text,
            "confidence": final_confidence,
            "context": video_caption
        }

    def _get_audio_context(self, audio_event: Dict) -> str:
        """Create a descriptive string of the audio event's characteristics."""
        tier = audio_event.get('tier', 'medium')
        onset_type = audio_event.get('onset_type', 'GENERAL')
        
        context = f"A {tier}-impact, "
        if onset_type == 'LOW_FREQ':
            context += "deep, low-frequency sound."
        elif onset_type == 'HIGH_FREQ':
            context += "sharp, high-frequency sound."
        else:
            context += "mid-range sound."
            
        return context

    def _fallback_audio_effect(self, audio_event: Dict) -> str:
        """Fallback onomatopoeia generation based on audio properties."""
        onset_type = audio_event.get('onset_type', 'GENERAL')
        if onset_type == 'LOW_FREQ': return random.choice(["THOOM", "BUMP", "SPLOOSH"])
        if onset_type == 'HIGH_FREQ': return random.choice(["CRACK", "CLICK", "TINK"])
        return random.choice(["BAM", "POP", "THWACK"])

    def _create_final_effect(self, decision: Dict, audio_event: Dict) -> Dict:
        """Creates the final event object with timing centered around the audio peak."""
        duration = random.uniform(0.7, 1.2)
        start_time = audio_event['time']
        end_time = start_time + duration
        
        return {
            'word': decision['text'],
            'start_time': max(0, start_time),
            'end_time': end_time,
            'confidence': decision['confidence'],
            'energy': audio_event.get('energy', 0) * decision['confidence'],
            'context': decision['context'],
        }