"""
Core animation system for onomatopoeia effects.
Contains the main OnomatopoeiaAnimator class and shared constants.
"""

import random
from .animation_types import AnimationType
from .position_calculator import PositionCalculator
from .renderer import ASSRenderer
from .utils import AnimationConstants


class OnomatopoeiaAnimator:
    """Main animator class that coordinates all animation functionality"""
    
    def __init__(self):
        self.position_calculator = PositionCalculator()
        self.renderer = ASSRenderer()
    
    @classmethod
    def get_all_animation_types(cls):
        """Get list of all available animation types."""
        return [
            AnimationType.DRIFT_FADE,
            AnimationType.WIGGLE,
            AnimationType.POP_SHRINK,
            AnimationType.SHAKE,
            AnimationType.PULSE,
            AnimationType.WAVE,
            AnimationType.EXPLODE_OUT,
            AnimationType.HYPER_BOUNCE
        ]
    
    @classmethod
    def get_random_animation_type(cls):
        """Randomly select an animation type from all available options."""
        return random.choice(cls.get_all_animation_types())
    
    @classmethod
    def get_animation_type_from_setting(cls, animation_setting):
        """Get animation type based on UI setting."""
        animation_map = {
            "Drift & Fade": AnimationType.DRIFT_FADE,
            "Wiggle": AnimationType.WIGGLE,
            "Pop & Shrink": AnimationType.POP_SHRINK,
            "Shake": AnimationType.SHAKE,
            "Pulse": AnimationType.PULSE,
            "Wave": AnimationType.WAVE,
            "Explode-Out": AnimationType.EXPLODE_OUT,
            "Hyper Bounce": AnimationType.HYPER_BOUNCE,
            "Random": cls.get_random_animation_type()
        }
        return animation_map.get(animation_setting, cls.get_random_animation_type())
    
    def create_animated_ass_events(self, events, animation_setting="Random"):
        """Create ASS dialogue events for all onomatopoeia with various animation styles."""
        from subtitle_styles import OnomatopoeiaStyle
        dialogue_lines = []
        
        for event in events:
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']
            font_size = OnomatopoeiaStyle.BASE_FONT_SIZE
            
            # Use position from style settings
            base_x = OnomatopoeiaStyle.MIN_MARGIN_L
            base_y = OnomatopoeiaStyle.MIN_MARGIN_V
            
            # Select animation type
            animation_type = self.get_animation_type_from_setting(animation_setting)
            
            # Handle per-letter animations differently
            if animation_type == AnimationType.WAVE:
                # Use per-letter wave animation
                letter_lines = self.position_calculator.create_wave_per_letter_events(
                    start_time, end_time, word, base_x, base_y, font_size
                )
                dialogue_lines.extend(letter_lines)
                
            elif animation_type == AnimationType.EXPLODE_OUT:
                # Use per-letter explode animation
                letter_lines = self.position_calculator.create_explode_per_letter_events(
                    start_time, end_time, word, base_x, base_y, font_size
                )
                dialogue_lines.extend(letter_lines)
                
            else:
                # Use standard whole-word animations
                positions = self.position_calculator.calculate_animation_positions(
                    animation_type, base_x, base_y, font_size, len(word)
                )
                
                # Create dialogue lines for each frame
                for frame, position_data in enumerate(positions):
                    if len(position_data) == 5:  # Has rotation
                        x, y, alpha, frame_font_size, rotation = position_data
                    else:  # No rotation (legacy format)
                        x, y, alpha, frame_font_size = position_data
                        rotation = 0
                    
                    frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                    frame_end = frame_start + AnimationConstants.FRAME_DURATION
                    
                    if frame_end > end_time:
                        frame_end = end_time
                    if frame_start >= end_time:
                        break
                    
                    # Use frame-specific font size if provided, otherwise use base
                    final_font_size = frame_font_size if frame_font_size is not None else font_size
                    
                    dialogue_line = self.renderer.create_ass_dialogue_line_with_rotation(
                        frame_start, frame_end, word, x, y, alpha, final_font_size, rotation
                    )
                    dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
    def generate_animated_ass_content(self, events, animation_setting="Random"):
        """Generate complete ASS file content with various animation styles."""
        if not events:
            return self.renderer.create_ass_header()
        
        ass_content = [self.renderer.create_ass_header()]
        dialogue_lines = self.create_animated_ass_events(events, animation_setting)
        
        for line in dialogue_lines:
            ass_content.append(line)
        
        return "\n".join(ass_content)