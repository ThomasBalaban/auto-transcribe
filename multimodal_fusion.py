# multimodal_fusion.py - TIMING PRECISION FIX

import numpy as np
from typing import List, Dict, Tuple
import random
from ollama_integration import OllamaLLM

class MultimodalFusionEngine:
    """
    Enhanced fusion engine with precise timing synchronization.
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
            # Find closest video analysis within reasonable time window
            matching_video = self._find_closest_video_analysis(audio_event, video_analyses_map)
            
            if matching_video:
                should_generate, effect_decision = self._make_fusion_decision(audio_event, matching_video)
                if should_generate:
                    final_effects.append(self._create_final_effect(effect_decision, audio_event))

        # Debug timing information
        self._log_timing_debug(final_effects)
        
        self.log_func(f"🎯 Fusion generated {len(final_effects)} effects.")
        return final_effects

    def _find_closest_video_analysis(self, audio_event: Dict, video_analyses_map: Dict[float, Dict], 
                                   max_time_diff: float = 3.0) -> Dict:
        """Find the closest video analysis within time window."""
        audio_time = audio_event['time']
        best_match = None
        best_diff = float('inf')
        
        for video_time, analysis in video_analyses_map.items():
            time_diff = abs(audio_time - video_time)
            if time_diff <= max_time_diff and time_diff < best_diff:
                best_diff = time_diff
                best_match = analysis
        
        return best_match

    def _make_fusion_decision(self, audio_event: Dict, video_analysis: Dict) -> Tuple[bool, Dict]:
        audio_drama = min(audio_event.get('energy', 0) / 0.1, 1.0)
        audio_context = self._get_audio_context(audio_event)

        visual_confidence = video_analysis.get('confidence', 0.9)
        video_caption = video_analysis.get('video_caption', '')
        scene_context = video_analysis.get('scene_context', set())

        # UNIVERSAL SPEECH DETECTION - No character-specific terms
        speech_indicators = [
            # Direct speech/dialogue
            "talking", "speaking", "says", "tells", "asks", "responds", "replies",
            "conversation", "dialogue", "discusses", "mentions", "explains",
            
            # Emotional reactions (often dialogue scenes)
            "reacts", "shocked", "surprise", "looks concerned", "expression",
            "stares", "gazes", "worry", "fear", "relief", "emotional",
            
            # Scene types that are typically dialogue-heavy
            "cutscene", "conversation", "discussion", "interview",
            "narrative", "story", "commentary"
        ]
        
        # STRONG ACTION INDICATORS - Clear physical events
        strong_action_indicators = [
            # Violent actions
            "shoot", "shot", "gun", "fire", "weapon", "killed", "attack",
            "punch", "hit", "kick", "strike", "fight", "combat", "battle",
            
            # Impact/collision
            "crash", "slam", "smash", "break", "destroy", "explode", "blast",
            "impact", "collision", "knocked", "falls", "collapses",
            
            # Movement with impact
            "jump", "leap", "throw", "drop", "slam", "dive"
        ]
        
        # WEAK ACTION INDICATORS - Might be part of dialogue scenes
        weak_action_indicators = [
            # Generic interactions
            "interact", "perform", "examine", "retrieve", "climb",
            "walk", "move", "run", "sneak", "crouch", "emerge"
        ]

        caption_lower = video_caption.lower()
        
        # Count indicators
        speech_score = sum(1 for indicator in speech_indicators if indicator in caption_lower)
        strong_action_score = sum(1 for indicator in strong_action_indicators if indicator in caption_lower)
        weak_action_score = sum(1 for indicator in weak_action_indicators if indicator in caption_lower)
        
        # UNIVERSAL LOGIC - Works for any content
        # Rule 1: If there are speech indicators and no strong actions, likely dialogue
        if speech_score > 0 and strong_action_score == 0:
            self.log_func(f"🤫 Event at {audio_event['time']:.2f}s REJECTED. "
                         f"Speech indicators ({speech_score}) without strong action: '{video_caption}'")
            return False, {}
        
        # Rule 2: If multiple speech indicators vs weak actions, likely dialogue
        if speech_score >= 2 and strong_action_score == 0 and weak_action_score <= 1:
            self.log_func(f"🤫 Event at {audio_event['time']:.2f}s REJECTED. "
                         f"Multiple speech indicators ({speech_score}) vs weak action ({weak_action_score}): '{video_caption}'")
            return False, {}
        
        # Rule 3: Audio energy check - very low energy + any speech = dialogue
        audio_energy = audio_event.get('energy', 0.5)
        if audio_energy < 0.03 and speech_score > 0:
            self.log_func(f"🤫 Event at {audio_event['time']:.2f}s REJECTED. "
                         f"Low audio energy ({audio_energy:.3f}) + speech indicators: '{video_caption}'")
            return False, {}

        final_confidence = (audio_drama * self.audio_weight) + (visual_confidence * self.video_weight)

        if final_confidence < self.min_confidence_threshold:
            self.log_func(f"📉 Event at {audio_event['time']:.2f}s failed confidence check. Score: {final_confidence:.2f}")
            return False, {}

        effect_text = self.local_llm.generate_onomatopoeia(video_caption, audio_context, scene_context)
        
        if not effect_text:
            self.log_func(f"⚠️ Local LLM failed to generate onomatopoeia. Using fallback.")
            effect_text = self._fallback_audio_effect(audio_event)

        self.log_func(f"✅ Event at {audio_event['time']:.2f}s PASSED. "
                     f"Speech: {speech_score}, Strong Action: {strong_action_score}, Weak Action: {weak_action_score}, "
                     f"Caption: '{video_caption}', Effect: '{effect_text}'")
        return True, {
            "text": effect_text,
            "confidence": final_confidence,
            "context": video_caption
        }

    def _get_audio_context(self, audio_event: Dict) -> str:
        """Create a descriptive string of the audio event's characteristics."""
        tier = audio_event.get('tier', 'medium')
        onset_type = audio_event.get('onset_type', 'GENERAL')
        energy = audio_event.get('energy', 0.5)
        
        context = f"A {tier}-impact sound with {energy:.2f} energy, "
        if onset_type == 'LOW_FREQ':
            context += "deep, low-frequency characteristics (possible impact, explosion, or bass-heavy sound)."
        elif onset_type == 'HIGH_FREQ':
            context += "sharp, high-frequency characteristics (possible gunshot, metal, or breaking sound)."
        else:
            context += "mid-range frequency characteristics (general impact or collision sound)."
            
        return context

    def _fallback_audio_effect(self, audio_event: Dict) -> str:
        """Enhanced fallback onomatopoeia generation based on audio properties."""
        onset_type = audio_event.get('onset_type', 'GENERAL')
        tier = audio_event.get('tier', 'medium')
        energy = audio_event.get('energy', 0.5)
        
        # High energy sounds
        if energy > 0.8:
            if onset_type == 'LOW_FREQ': 
                return random.choice(["KABOOM!", "THOOM!", "WHAM!"])
            elif onset_type == 'HIGH_FREQ': 
                return random.choice(["CRACK!", "BANG!", "SNAP!"])
            else: 
                return random.choice(["SLAM!", "CRASH!", "BOOM!"])
        
        # Medium energy sounds  
        elif energy > 0.4:
            if onset_type == 'LOW_FREQ': 
                return random.choice(["THUD", "BUMP", "WHUMP"])
            elif onset_type == 'HIGH_FREQ': 
                return random.choice(["CLICK", "TINK", "PING"])
            else: 
                return random.choice(["THWACK", "POP", "CLUNK"])
        
        # Low energy sounds
        else:
            if onset_type == 'LOW_FREQ': 
                return random.choice(["thump", "bump"])
            elif onset_type == 'HIGH_FREQ': 
                return random.choice(["tick", "click"])
            else: 
                return random.choice(["tap", "pat"])

    def _create_final_effect(self, decision: Dict, audio_event: Dict) -> Dict:
        """Creates the final event with BALANCED timing for universal content."""
        # Use the precise peak time, not the detection time
        precise_peak_time = audio_event.get('peak_time', audio_event['time'])
        
        # Determine timing adjustment based on sound characteristics
        onset_type = audio_event.get('onset_type', 'GENERAL')
        tier = audio_event.get('tier', 'medium')
        context_lower = decision['context'].lower()
        
        # BALANCED TIMING ADJUSTMENTS that work universally
        if (onset_type == 'HIGH_FREQ' or 
            any(keyword in context_lower for keyword in ['shot', 'gun', 'fire', 'shoot'])):
            # Gunshots: Sharp, precise timing
            start_offset = -0.25  # 250ms before peak
            duration = 0.5        # Short, sharp effect
            
        elif (any(keyword in context_lower for keyword in ['punch', 'hit', 'kick', 'strike', 'attack', 'killed']) or
              tier == 'major'):
            # Combat actions: Medium timing  
            start_offset = -0.15  # 150ms before peak
            duration = 0.7        # Medium duration
            
        elif (any(keyword in context_lower for keyword in ['water', 'splash', 'swim', 'dive']) or
              'underwater' in str(decision.get('scene_context', ''))):
            # Water sounds: Splash anticipation
            start_offset = -0.2   # 200ms before peak
            duration = 1.0        # Longer for water sounds
            
        elif (onset_type == 'LOW_FREQ' or
              any(keyword in context_lower for keyword in ['boom', 'explode', 'blast', 'crash'])):
            # Explosions/crashes: Earlier for impact anticipation
            start_offset = -0.18  # 180ms before peak
            duration = 1.0        # Longer for low frequency
            
        else:
            # General impacts: Conservative timing
            start_offset = -0.12  # 120ms before peak
            duration = 0.8
        
        # Calculate final timing
        final_start_time = max(0, precise_peak_time + start_offset)
        final_end_time = final_start_time + duration
        
        return {
            'word': decision['text'],
            'start_time': final_start_time,
            'end_time': final_end_time,
            'confidence': decision['confidence'],
            'energy': audio_event.get('energy', 0) * decision['confidence'],
            'context': decision['context'],
            'tier': audio_event.get('tier', 'medium'),
            'onset_type': audio_event.get('onset_type', 'GENERAL'),
            
            # Debugging fields
            'original_detection_time': audio_event['time'],
            'precise_peak_time': precise_peak_time,
            'timing_offset': start_offset
        }

    def _log_timing_debug(self, effects: List[Dict]):
        """Log timing information for debugging synchronization."""
        if not effects:
            return
            
        self.log_func("\n🎯 PRECISE TIMING DEBUG:")
        for effect in effects:
            word = effect.get('word', 'UNKNOWN')
            detection_time = effect.get('original_detection_time', 0)
            peak_time = effect.get('precise_peak_time', 0)
            final_time = effect.get('start_time', 0)
            offset = effect.get('timing_offset', 0)
            onset_type = effect.get('onset_type', 'UNKNOWN')
            
            self.log_func(f"  {word:12} | Det:{detection_time:6.2f}s | Peak:{peak_time:6.2f}s | "
                         f"Final:{final_time:6.2f}s | Offset:{offset:+5.2f}s | {onset_type}")
        self.log_func("")