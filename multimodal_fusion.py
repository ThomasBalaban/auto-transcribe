# multimodal_fusion.py

import numpy as np
from typing import List, Dict, Tuple
import random
from ollama_integration import OllamaLLM

class MultimodalFusionEngine:
    """
    Fuses audio analysis and video captions with final polished logic for timing,
    context, and confidence.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.audio_weight = 0.6
        self.video_weight = 0.4
        self.min_confidence_threshold = 0.45
        self.ollama_llm = OllamaLLM(log_func=self.log_func)

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
        audio_context = self._get_audio_context(audio_event)

        if not video_analysis or not video_analysis.get('video_caption'):
            # Fallback for audio-only events
            if audio_drama > 0.7:
                effect_text = self._fallback_audio_effect(audio_event)
                return True, {"text": effect_text, "timing": audio_event['peak_time'], "confidence": audio_drama, "context": "audio_only"}
            return False, {}

        visual_confidence = video_analysis.get('confidence', 0)
        video_caption = video_analysis.get('video_caption', '')
        scene_context = video_analysis.get('scene_context', set())
        
        # New: Check for speech-related keywords in the caption
        speech_keywords = ["talking", "speaking", "narrating", "shouting", "whispering", "saying"]
        if any(keyword in video_caption.lower() for keyword in speech_keywords):
            self.log_func(f"🤫 Event at {audio_event['time']:.2f}s ignored due to speech in caption: '{video_caption}'")
            return False, {}

        base_score = (audio_drama * self.audio_weight) + (visual_confidence * self.video_weight)
        
        context_bonus = 0.1 if audio_event.get('tier') == 'major' else 0.0
        final_confidence = base_score + context_bonus

        if final_confidence < self.min_confidence_threshold:
            self.log_func(f"📉 Event at {audio_event['time']:.2f}s failed. Score: {final_confidence:.2f}")
            return False, {}

        # Generate onomatopoeia using video caption, audio context, and scene context
        effect_text = self.ollama_llm.generate_onomatopoeia(video_caption, audio_context, scene_context)
        
        if not effect_text:
            self.log_func(f"⚠️ LLM failed to generate onomatopoeia. Using fallback.")
            effect_text = self._fallback_audio_effect(audio_event)

        self.log_func(f"✅ Event at {audio_event['time']:.2f}s PASSED. Score: {final_confidence:.2f}, Caption: '{video_caption}', Effect: '{effect_text}'")
        return True, {
            "text": effect_text,
            "timing": audio_event['peak_time'],
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
        if onset_type == 'LOW_FREQ': return random.choice(["THOOM", "BUMP"])
        if onset_type == 'HIGH_FREQ': return random.choice(["CRACK", "CLICK"])
        return random.choice(["BAM", "POP"])

    def _create_final_effect(self, decision: Dict, audio_event: Dict) -> Dict:
        return {
            'word': decision['text'],
            'start_time': decision['timing'],
            'end_time': decision['timing'] + random.uniform(0.7, 1.5),
            'confidence': decision['confidence'],
            'energy': audio_event.get('energy', 0) * decision['confidence'],
            'context': decision['context'],
        }