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
        return AnimationType.get_all_types()

    @classmethod
    def get_animation_type_from_setting(cls, animation_setting):
        """Get animation type based on UI setting."""
        display_names = AnimationType.get_display_names()
        name_to_type = {v: k for k, v in display_names.items()}

        # Add special cases for Random and the new Intelligent mode
        name_to_type["Auto"] = "Intelligent"
        name_to_type["Intelligent"] = "Intelligent" 

        return name_to_type.get(animation_setting, "Intelligent")

    @classmethod
    def get_animation_timing_offset(cls, animation_type):
        """
        Calculate the timing offset needed to align the animation's peak
        with the event's peak time.
        """
        peak_frame = AnimationType.get_peak_frame(animation_type)
        return - (peak_frame * AnimationConstants.FRAME_DURATION)


    def create_animated_ass_events(self, events):
        """Create ASS dialogue events for all onomatopoeia with various animation styles."""
        from utils.subtitle_styles import OnomatopoeiaStyle
        dialogue_lines = []

        for event in events:
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']
            animation_type = event['animation_type'] # This is now decided in the fusion engine
            font_size = OnomatopoeiaStyle.BASE_FONT_SIZE
            base_x = OnomatopoeiaStyle.MIN_MARGIN_L
            base_y = OnomatopoeiaStyle.MIN_MARGIN_V

            if AnimationType.requires_per_letter_rendering(animation_type):
                letter_lines = []
                if animation_type == AnimationType.WAVE:
                    letter_lines = self.position_calculator.create_wave_per_letter_events(
                        start_time, end_time, word, base_x, base_y, font_size
                    )
                elif animation_type == AnimationType.EXPLODE_OUT:
                    letter_lines = self.position_calculator.create_explode_per_letter_events(
                        start_time, end_time, word, base_x, base_y, font_size
                    )
                dialogue_lines.extend(letter_lines)
            else:
                positions = self.position_calculator.calculate_animation_positions(
                    animation_type, base_x, base_y, font_size, len(word)
                )

                for frame, position_data in enumerate(positions):
                    x, y, alpha, frame_font_size, rotation = position_data
                    frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                    frame_end = frame_start + AnimationConstants.FRAME_DURATION
                    if frame_end > end_time: frame_end = end_time
                    if frame_start >= end_time: break

                    final_font_size = frame_font_size if frame_font_size is not None else font_size
                    dialogue_line = self.renderer.create_ass_dialogue_line_with_rotation(
                        frame_start, frame_end, word, x, y, alpha, final_font_size, rotation
                    )
                    dialogue_lines.append(dialogue_line)

        return dialogue_lines

    def generate_animated_ass_content(self, events):
        """Generate complete ASS file content with various animation styles."""
        if not events:
            return self.renderer.create_ass_header()

        ass_content = [self.renderer.create_ass_header()]
        dialogue_lines = self.create_animated_ass_events(events)
        ass_content.extend(dialogue_lines)

        return "\n".join(ass_content)