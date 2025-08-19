
"""
Multimodal Fusion Engine for intelligent onomatopoeia generation (Phase 3).
Combines audio onset detection + video analysis for context-aware effects.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import random
import os


class MultimodalFusionEngine:
    """Intelligent fusion of audio and video analysis for onomatopoeia decisions"""
    
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        
        # Fusion parameters
        self.min_confidence_threshold = 0.5
        self.audio_weight = 0.6  # Audio slightly preferred for timing
        self.video_weight = 0.4  # Video provides context
        
        # Effect selection mappings
        self.action_effect_map = self._create_action_effect_mapping()
        
        self.log_func("🔄 Multimodal fusion engine initialized")
        self.log_func(f"   Audio weight: {self.audio_weight}, Video weight: {self.video_weight}")

    def _create_action_effect_mapping(self) -> Dict[str, Dict]:
        """Create mapping from video actions to appropriate onomatopoeia effects"""
        return {
            "character attacking with weapon": {
                "melee": ["WHACK!", "THWACK!", "SLAM!", "SMASH!"],
                "ranged": ["BANG!", "BLAST!", "SHOT!", "FIRE!"],
                "generic": ["ATTACK!", "STRIKE!", "HIT!"]
            },
            "explosion or blast occurring": {
                "major": ["KABOOM!", "BLAST!", "BOOM!", "EXPLODE!"],
                "medium": ["BANG!", "CRASH!", "BOOM!"],
                "minor": ["POP!", "CRACK!", "SNAP!"]
            },
            "character taking damage or being hit": {
                "heavy": ["WHAM!", "SMASH!", "CRUSH!", "SLAM!"],
                "medium": ["THUD!", "WHACK!", "OOF!", "HURT!"],
                "light": ["TAP!", "BUMP!", "OUCH!", "HIT!"]
            },
            "object breaking or destruction": {
                "glass": ["SHATTER!", "CRASH!", "SMASH!", "BREAK!"],
                "wood": ["CRACK!", "SNAP!", "SPLINTER!", "BREAK!"],
                "metal": ["CLANG!", "CRASH!", "CLANK!", "BANG!"],
                "generic": ["CRASH!", "BREAK!", "SMASH!", "DESTROY!"]
            },
            "character moving or running quickly": {
                "footsteps": ["STOMP!", "THUD!", "STEP!", "POUND!"],
                "rushing": ["WHOOSH!", "DASH!", "RUSH!", "ZOOM!"],
                "jumping": ["LEAP!", "BOUND!", "JUMP!", "HOP!"]
            },
            "mechanical or electronic sounds happening": {
                "electronic": ["BEEP!", "BUZZ!", "ZAP!", "BLEEP!"],
                "mechanical": ["WHIRR!", "CLICK!", "CLANK!", "TICK!"],
                "hydraulic": ["HISS!", "WHOOSH!", "PRESS!", "PUMP!"]
            },
            "monster or enemy appearing": {
                "large": ["ROAR!", "GROWL!", "BELLOW!", "SNARL!"],
                "small": ["HISS!", "SCREECH!", "SQUEAK!", "CHITTER!"],
                "supernatural": ["WAIL!", "MOAN!", "HOWL!", "SHRIEK!"]
            },
            "environmental destruction or collapse": {
                "building": ["RUMBLE!", "COLLAPSE!", "CRUMBLE!", "CRASH!"],
                "natural": ["CRACK!", "SPLIT!", "BREAK!", "FALL!"],
                "massive": ["THUNDER!", "BOOM!", "ROAR!", "QUAKE!"]
            },
            "gunfire or shooting": {
                "single": ["BANG!", "SHOT!", "FIRE!", "BLAST!"],
                "automatic": ["RATATA!", "SPRAY!", "RAPID!", "BURST!"],
                "heavy": ["BOOM!", "CANNON!", "BLAST!", "THUNDER!"]
            },
            "quiet or peaceful scene with no action": {
                "suppress": []  # No effects for quiet scenes
            }
        }

    def calculate_audio_drama_score(self, audio_event: Dict) -> float:
        """Calculate drama score from audio event characteristics"""
        if not audio_event:
            return 0.0
        
        # Base score from energy
        energy_score = min(audio_event.get('energy', 0) / 0.1, 1.0)
        
        # Tier bonus
        tier = audio_event.get('tier', 'quick')
        if tier == 'major':
            tier_bonus = 0.3
        elif tier == 'medium':
            tier_bonus = 0.2
        else:  # quick
            tier_bonus = 0.1
        
        # Onset type bonus
        onset_type = audio_event.get('onset_type', 'GENERAL')
        type_bonus = {
            'LOW_FREQ': 0.2,    # Explosions, bass impacts
            'HIGH_FREQ': 0.15,  # Gunshots, sharp sounds
            'SHARP': 0.1,       # Clicks, taps
            'GENERAL': 0.05     # General impacts
        }.get(onset_type, 0.05)
        
        # Combine scores
        total_score = min(energy_score + tier_bonus + type_bonus, 1.0)
        return total_score

    def calculate_temporal_alignment(self, audio_event: Dict, video_analysis: Dict) -> float:
        """Calculate how well audio and video events are temporally aligned"""
        if not audio_event or not video_analysis:
            return 0.5  # Neutral if missing data
        
        audio_time = audio_event.get('time', 0)
        video_start = video_analysis.get('start_time', 0)
        video_end = video_start + video_analysis.get('duration', 2.0)
        
        # Check if audio event falls within video analysis window
        if video_start <= audio_time <= video_end:
            # Perfect alignment - audio event is within video window
            return 1.0
        else:
            # Calculate distance penalty
            if audio_time < video_start:
                distance = video_start - audio_time
            else:
                distance = audio_time - video_end
            
            # Exponential decay based on distance
            alignment_score = max(0.1, np.exp(-distance))
            return alignment_score

    def determine_effect_intensity(self, audio_drama: float, visual_drama: float, 
                                 temporal_alignment: float) -> float:
        """Determine the intensity/size of the effect"""
        # Weighted combination
        combined_drama = (audio_drama * self.audio_weight + 
                         visual_drama * self.video_weight)
        
        # Temporal alignment bonus
        aligned_drama = combined_drama * (0.7 + 0.3 * temporal_alignment)
        
        return min(aligned_drama, 1.0)

    def select_contextual_effect(self, audio_event: Dict, video_analysis: Dict) -> str:
        """Select appropriate onomatopoeia based on audio + video context"""
        
        # Get video action classification
        if not video_analysis or 'action_classification' not in video_analysis:
            return self._fallback_audio_effect(audio_event)
        
        action_data = video_analysis['action_classification']
        primary_action = action_data.get('primary_action', '')
        action_confidence = action_data.get('confidence', 0)
        
        # If video action confidence is low, fall back to audio
        if action_confidence < 0.6:
            return self._fallback_audio_effect(audio_event)
        
        # Check if action suggests no effects (quiet scenes)
        if "quiet" in primary_action or "peaceful" in primary_action:
            return None  # Suppress effect for quiet scenes
        
        # Get effect category from action
        effect_category = self._map_action_to_effect_category(primary_action, audio_event)
        
        if primary_action in self.action_effect_map:
            effect_options = self.action_effect_map[primary_action]
            
            # Select sub-category based on audio characteristics
            if effect_category in effect_options:
                effects = effect_options[effect_category]
            else:
                # Use first available category
                effects = list(effect_options.values())[0]
                if isinstance(effects, list):
                    pass  # effects is already a list
                else:
                    effects = list(effects.values())[0]  # Go deeper if nested
        else:
            # Fallback to audio-based effect
            return self._fallback_audio_effect(audio_event)
        
        if not effects:  # Empty list means suppress
            return None
        
        return random.choice(effects)

    def _map_action_to_effect_category(self, action: str, audio_event: Dict) -> str:
        """Map specific action + audio to effect sub-category"""
        energy = audio_event.get('energy', 0)
        onset_type = audio_event.get('onset_type', 'GENERAL')
        
        if "attacking with weapon" in action:
            if onset_type == 'HIGH_FREQ':
                return "ranged"  # Gunfire, projectiles
            elif onset_type == 'LOW_FREQ' or energy > 0.05:
                return "melee"   # Heavy impacts
            else:
                return "generic"
        
        elif "explosion" in action:
            if energy > 0.08:
                return "major"   # Big explosions
            elif energy > 0.03:
                return "medium"  # Medium explosions
            else:
                return "minor"   # Small explosions
        
        elif "taking damage" in action:
            if energy > 0.06:
                return "heavy"   # Heavy damage
            elif energy > 0.03:
                return "medium"  # Medium damage
            else:
                return "light"   # Light damage
        
        elif "breaking" in action:
            if onset_type == 'HIGH_FREQ':
                return "glass"   # Sharp, brittle sounds
            elif onset_type == 'LOW_FREQ':
                return "metal"   # Heavy, metallic
            else:
                return "wood"    # General breaking
        
        elif "moving" in action:
            if energy > 0.04:
                return "footsteps"  # Clear footsteps
            elif onset_type == 'SHARP':
                return "jumping"    # Quick movements
            else:
                return "rushing"    # General movement
        
        elif "mechanical" in action:
            if onset_type == 'HIGH_FREQ':
                return "electronic"  # Beeps, electronic
            elif onset_type == 'SHARP':
                return "mechanical"  # Clicks, mechanical
            else:
                return "hydraulic"   # Smooth mechanical
        
        elif "monster" in action or "enemy" in action:
            if energy > 0.06:
                return "large"       # Big monsters
            elif onset_type == 'HIGH_FREQ':
                return "small"       # Small creatures
            else:
                return "supernatural" # Spooky sounds
        
        elif "destruction" in action or "collapse" in action:
            if energy > 0.08:
                return "massive"     # Huge destruction
            elif "building" in action or "structure" in action:
                return "building"    # Structural collapse
            else:
                return "natural"     # Natural destruction
        
        elif "gunfire" in action or "shooting" in action:
            if energy > 0.07:
                return "heavy"       # Heavy weapons
            elif audio_event.get('tier') == 'quick':
                return "automatic"   # Rapid fire
            else:
                return "single"      # Single shots
        
        return "generic"

    def _fallback_audio_effect(self, audio_event: Dict) -> str:
        """Fallback to audio-only effect selection"""
        if not audio_event:
            return "THUD!"
        
        onset_type = audio_event.get('onset_type', 'GENERAL')
        tier = audio_event.get('tier', 'medium')
        energy = audio_event.get('energy', 0)
        
        # Intensity-based effects
        if tier == 'major' or energy > 0.08:
            if onset_type == 'LOW_FREQ':
                return random.choice(['BOOM!', 'CRASH!', 'SLAM!'])
            elif onset_type == 'HIGH_FREQ':
                return random.choice(['BANG!', 'CLANG!', 'CRACK!'])
            else:
                return random.choice(['WHAM!', 'SMASH!', 'BLAST!'])
        
        elif tier == 'medium' or energy > 0.03:
            if onset_type == 'LOW_FREQ':
                return random.choice(['THUD!', 'BUMP!', 'THUMP!'])
            elif onset_type == 'HIGH_FREQ':
                return random.choice(['CLICK!', 'SNAP!', 'TAP!'])
            else:
                return random.choice(['WHACK!', 'HIT!', 'STRIKE!'])
        
        else:  # quick/light
            return random.choice(['TAP!', 'CLICK!', 'POP!'])

    def determine_effect_timing(self, audio_event: Dict, video_analysis: Dict, 
                              temporal_alignment: float) -> float:
        """Determine optimal timing for effect placement"""
        audio_time = audio_event.get('time', 0)
        
        if not video_analysis or temporal_alignment < 0.7:
            # Low alignment - use audio timing
            return audio_time
        
        # High alignment - check for visual peak within window
        video_features = video_analysis.get('video_features', {})
        motion_score = video_features.get('motion_score', 0)
        
        if motion_score > 0.5:  # High motion - use visual timing
            video_start = video_analysis.get('start_time', audio_time)
            video_duration = video_analysis.get('duration', 1.0)
            
            # Place effect at visual peak (assume middle of high-motion window)
            visual_peak_time = video_start + video_duration * 0.4  # Slightly before middle
            
            # Don't drift too far from audio
            max_drift = 0.5  # 500ms max drift
            if abs(visual_peak_time - audio_time) <= max_drift:
                return visual_peak_time
        
        # Default to audio timing
        return audio_time

    def should_generate_effect(self, audio_event: Dict, video_analysis: Dict) -> Tuple[bool, Dict]:
        """Main fusion decision: should we generate an onomatopoeia effect?"""
        try:
            self.log_func(f"\n🔄 Multimodal fusion analysis:")
            
            # Calculate component scores
            audio_drama = self.calculate_audio_drama_score(audio_event)
            
            if video_analysis:
                visual_drama = video_analysis.get('visual_drama_score', 0)
                temporal_alignment = self.calculate_temporal_alignment(audio_event, video_analysis)
            else:
                visual_drama = 0.0
                temporal_alignment = 0.5
            
            self.log_func(f"   Audio drama: {audio_drama:.2f}")
            self.log_func(f"   Visual drama: {visual_drama:.2f}")
            self.log_func(f"   Temporal alignment: {temporal_alignment:.2f}")
            
            # Calculate combined confidence
            weighted_drama = (audio_drama * self.audio_weight + 
                            visual_drama * self.video_weight)
            
            # Temporal alignment affects final confidence
            final_confidence = weighted_drama * (0.5 + 0.5 * temporal_alignment)
            
            self.log_func(f"   Combined confidence: {final_confidence:.2f}")
            
            # Decision threshold
            if final_confidence < self.min_confidence_threshold:
                self.log_func(f"   ❌ Below threshold ({self.min_confidence_threshold})")
                return False, {}
            
            # Generate effect details
            effect_text = self.select_contextual_effect(audio_event, video_analysis)
            
            if not effect_text:  # Explicitly suppressed (quiet scene)
                self.log_func(f"   🔇 Effect suppressed (quiet scene)")
                return False, {}
            
            effect_intensity = self.determine_effect_intensity(
                audio_drama, visual_drama, temporal_alignment
            )
            
            effect_timing = self.determine_effect_timing(
                audio_event, video_analysis, temporal_alignment
            )
            
            # Create effect decision
            effect_decision = {
                "generate": True,
                "text": effect_text,
                "intensity": effect_intensity,
                "timing": effect_timing,
                "confidence": final_confidence,
                "audio_drama": audio_drama,
                "visual_drama": visual_drama,
                "temporal_alignment": temporal_alignment,
                "context": video_analysis.get('action_classification', {}).get('primary_action', 'audio-only') if video_analysis else 'audio-only'
            }
            
            self.log_func(f"   ✅ Effect: '{effect_text}' at {effect_timing:.2f}s (intensity: {effect_intensity:.2f})")
            self.log_func(f"   Context: {effect_decision['context']}")
            
            return True, effect_decision
            
        except Exception as e:
            self.log_func(f"   💥 Fusion error: {e}")
            return False, {}

    def process_multimodal_events(self, audio_events: List[Dict], 
                                video_analyses: List[Dict]) -> List[Dict]:
        """Process all audio events with video context for final effects"""
        
        self.log_func(f"\n🔄 Processing {len(audio_events)} audio events with video context...")
        
        final_effects = []
        
        for audio_event in audio_events:
            audio_time = audio_event.get('time', 0)
            
            # Find matching video analysis (if any)
            matching_video = None
            best_overlap = 0
            
            for video_analysis in video_analyses:
                video_start = video_analysis.get('start_time', 0)
                video_end = video_start + video_analysis.get('duration', 2.0)
                
                # Calculate overlap
                if video_start <= audio_time <= video_end:
                    overlap = 1.0  # Perfect overlap
                else:
                    # Calculate proximity
                    if audio_time < video_start:
                        distance = video_start - audio_time
                    else:
                        distance = audio_time - video_end
                    overlap = max(0, 1.0 - distance / 2.0)  # Decay over 2 seconds
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    matching_video = video_analysis
            
            # Make fusion decision
            should_generate, effect_decision = self.should_generate_effect(
                audio_event, matching_video
            )
            
            if should_generate:
                # Create final effect event
                effect_event = {
                    'word': effect_decision['text'],
                    'start_time': effect_decision['timing'],
                    'end_time': effect_decision['timing'] + random.uniform(0.5, 1.5),
                    'confidence': effect_decision['confidence'],
                    'energy': audio_event.get('energy', 0) * effect_decision['intensity'],
                    'context': effect_decision['context'],
                    'multimodal': True,
                    'audio_source': audio_event,
                    'video_source': matching_video
                }
                
                final_effects.append(effect_event)
        
        self.log_func(f"🎯 Multimodal fusion complete: {len(final_effects)} effects generated")
        return final_effects


def test_multimodal_fusion():
    """Test the multimodal fusion engine"""
    print("Testing multimodal fusion engine...")
    
    fusion_engine = MultimodalFusionEngine(log_func=print)
    
    # Create test audio event
    test_audio_event = {
        'time': 5.2,
        'tier': 'major',
        'energy': 0.08,
        'onset_type': 'LOW_FREQ',
        'confidence': 0.8
    }
    
    # Create test video analysis
    test_video_analysis = {
        'start_time': 4.5,
        'duration': 2.0,
        'visual_drama_score': 0.7,
        'action_classification': {
            'primary_action': 'explosion or blast occurring',
            'confidence': 0.85
        }
    }
    
    # Test fusion
    should_generate, effect_decision = fusion_engine.should_generate_effect(
        test_audio_event, test_video_analysis
    )
    
    if should_generate:
        print(f"✅ Fusion test successful!")
        print(f"   Effect: {effect_decision['text']}")
        print(f"   Confidence: {effect_decision['confidence']:.2f}")
        print(f"   Context: {effect_decision['context']}")
        return True
    else:
        print("❌ Fusion test failed - no effect generated")
        return False


if __name__ == "__main__":
    test_multimodal_fusion()