"""
Position calculation algorithms for different animation types.
Handles movement physics, easing, and position generation for animations.
"""

import random
import math
from .animation_types import AnimationType
from .utils import AnimationConstants
from .renderer import ASSRenderer


class PositionCalculator:
    """Calculates positions and movements for different animation types"""

    def __init__(self):
        self.renderer = ASSRenderer()

    def calculate_drift_positions(self, base_x, base_y):
        """Calculate smooth positions for drift and fade animation."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
            ease_progress = 1 - (1 - progress) ** 2

            x = base_x
            y = base_y - int(AnimationConstants.DRIFT_DISTANCE * ease_progress)
            alpha = int(255 * (1.0 - 0.8 * progress))
            font_size = None
            rotation = 0  # FIX: Added missing rotation value

            positions.append((x, y, alpha, font_size, rotation))
        return positions

    def calculate_wiggle_positions(self, base_x, base_y):
        """Calculate smooth positions for wiggle animation using sine wave."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 4 * math.pi
            x_offset = AnimationConstants.WIGGLE_AMPLITUDE * math.sin(angle)

            x = base_x + int(x_offset)
            y = base_y
            alpha = 255
            font_size = None
            rotation = 0  # FIX: Added missing rotation value

            positions.append((x, y, alpha, font_size, rotation))
        return positions

    def calculate_pop_shrink_positions(self, base_x, base_y, base_font_size):
        """Calculate positions for pop and shrink animation with rubber band elasticity."""
        positions = []

        sizes = [
            base_font_size, int(base_font_size * 1.8), int(base_font_size * 1.9),
            int(base_font_size * 0.7), int(base_font_size * 0.6), int(base_font_size * 1.1),
            int(base_font_size * 0.9), int(base_font_size * 1.05), int(base_font_size * 0.95),
            int(base_font_size * 1.02), int(base_font_size * 0.98), base_font_size,
            base_font_size, base_font_size, base_font_size
        ]
        rotations = [0, 5, 5, -3, -3, 2, -1, 1, -0.5, 0.5, 0, 0, 0, 0, 0]

        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            font_size = sizes[frame]
            rotation = rotations[frame]
            alpha = 255
            positions.append((base_x, base_y, alpha, font_size, rotation))
        return positions

    def calculate_shake_positions(self, base_x, base_y):
        """Calculate positions for shake animation with exponential decay and rotation."""
        positions = []
        max_shake = AnimationConstants.SHAKE_AMPLITUDE
        max_rotation = 15

        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
            intensity = (1 - progress) ** 2
            shake_amount = max_shake * intensity
            rotation_amount = max_rotation * intensity

            x_offset = random.randint(-int(shake_amount), int(shake_amount))
            y_offset = random.randint(-int(shake_amount // 2), int(shake_amount // 2))
            rotation = random.uniform(-rotation_amount, rotation_amount)
            font_size = None
            alpha = 255

            positions.append((base_x + x_offset, base_y + y_offset, alpha, font_size, rotation))
        return positions

    def calculate_pulse_positions(self, base_x, base_y, base_font_size):
        """Calculate positions for pulse animation with smoother size changes."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 4 * math.pi
            size_variation = (math.sin(angle) + 1) / 2
            size_multiplier = 1.0 + (AnimationConstants.PULSE_SCALE_FACTOR - 1.0) * size_variation
            font_size = int(base_font_size * size_multiplier)
            alpha = 255
            rotation = 0  # FIX: Added missing rotation value

            positions.append((base_x, base_y, alpha, font_size, rotation))
        return positions

    def calculate_wave_positions(self, base_x, base_y, word_length):
        """Calculate positions for wave animation (simplified single-position version)."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 6 * math.pi
            y_offset = AnimationConstants.WAVE_AMPLITUDE * math.sin(angle)

            x = base_x
            y = base_y + int(y_offset)
            alpha = 255
            font_size = None
            rotation = 0  # FIX: Added missing rotation value

            positions.append((x, y, alpha, font_size, rotation))
        return positions

    def calculate_explode_out_positions(self, base_x, base_y):
        """Calculate positions for explode-out animation (fallback for whole word)."""
        positions = []
        angle = random.uniform(0, 2 * math.pi)

        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
            ease_progress = 1 - (1 - progress) ** 2
            distance = AnimationConstants.EXPLODE_DISTANCE * ease_progress
            x_offset = distance * math.cos(angle)
            y_offset = distance * math.sin(angle)

            x = base_x + int(x_offset)
            y = base_y + int(y_offset)
            alpha = int(255 * (1.0 - 0.9 * progress))
            font_size = None
            rotation = 0  # FIX: Added missing rotation value

            positions.append((x, y, alpha, font_size, rotation))
        return positions

    def calculate_hyper_bounce_positions(self, base_x, base_y):
        """Calculate positions for hyper bounce animation."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            bounce_cycles = 3
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * bounce_cycles * 2 * math.pi
            amplitude_decay = 1.0 - (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 0.3
            bounce_height = AnimationConstants.BOUNCE_HEIGHT * amplitude_decay

            y_offset = bounce_height * abs(math.sin(angle))
            x_jitter = random.randint(-3, 3)
            font_size = None
            alpha = 255
            rotation = 0  # FIX: Added missing rotation value

            positions.append((base_x + x_jitter, base_y - int(y_offset), alpha, font_size, rotation))
        return positions

    def calculate_animation_positions(self, animation_type, base_x, base_y, base_font_size=140, word_length=5):
        """Calculate positions for any animation type."""
        if animation_type == AnimationType.DRIFT_FADE:
            return self.calculate_drift_positions(base_x, base_y)
        elif animation_type == AnimationType.WIGGLE:
            return self.calculate_wiggle_positions(base_x, base_y)
        elif animation_type == AnimationType.POP_SHRINK:
            return self.calculate_pop_shrink_positions(base_x, base_y, base_font_size)
        elif animation_type == AnimationType.SHAKE:
            return self.calculate_shake_positions(base_x, base_y)
        elif animation_type == AnimationType.PULSE:
            return self.calculate_pulse_positions(base_x, base_y, base_font_size)
        elif animation_type == AnimationType.WAVE:
            return self.calculate_wave_positions(base_x, base_y, word_length)
        elif animation_type == AnimationType.EXPLODE_OUT:
            return self.calculate_explode_out_positions(base_x, base_y)
        elif animation_type == AnimationType.HYPER_BOUNCE:
            return self.calculate_hyper_bounce_positions(base_x, base_y)
        else:
            return self.calculate_drift_positions(base_x, base_y)

    def create_wave_per_letter_events(self, start_time, end_time, word, base_x, base_y, base_font_size):
        """Create wave animation with per-letter lean physics - single wave pass."""
        dialogue_lines = []
        letter_spacing = base_font_size * 0.6

        for letter_index, letter in enumerate(word):
            if letter.isspace():
                continue

            letter_x = base_x + int(letter_index * letter_spacing)

            for frame in range(AnimationConstants.ANIMATION_FRAMES):
                frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                frame_end = frame_start + AnimationConstants.FRAME_DURATION
                if frame_end > end_time: frame_end = end_time
                if frame_start >= end_time: break

                time_progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
                wave_position = -1 + (len(word) + 2) * time_progress
                distance_from_wave = abs(wave_position - letter_index)
                wave_width = 1.5

                if distance_from_wave <= wave_width:
                    wave_intensity = math.cos((distance_from_wave / wave_width) * math.pi / 2)
                    y_offset = AnimationConstants.WAVE_AMPLITUDE * wave_intensity
                    wave_direction = wave_position - letter_index
                    rotation = max(-30, min(30, -30 * wave_direction * wave_intensity / wave_width))
                else:
                    y_offset = 0
                    rotation = 0

                dialogue_line = self.renderer.create_wave_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, base_y + int(y_offset), 255, base_font_size, rotation
                )
                dialogue_lines.append(dialogue_line)

        return dialogue_lines

    def create_explode_per_letter_events(self, start_time, end_time, word, base_x, base_y, base_font_size):
        """Create explode-out animation with per-letter control - true explosion effect."""
        dialogue_lines = []
        letter_spacing = base_font_size * 0.6
        word_center_x = base_x + (len(word) * letter_spacing) / 2

        for letter_index, letter in enumerate(word):
            if letter.isspace():
                continue

            letter_start_x = base_x + int(letter_index * letter_spacing)
            dx = letter_start_x - word_center_x
            angle = math.atan2(0, dx) + random.uniform(-0.5, 0.5) if abs(dx) >= 5 else random.uniform(0, 2 * math.pi)
            rotation_speed = random.uniform(-180, 180)

            for frame in range(AnimationConstants.ANIMATION_FRAMES):
                frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                frame_end = frame_start + AnimationConstants.FRAME_DURATION
                if frame_end > end_time: frame_end = end_time
                if frame_start >= end_time: break

                progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
                ease_progress = progress ** 0.5
                explosion_distance = (AnimationConstants.EXPLODE_DISTANCE * 1.5) * ease_progress
                x_offset = explosion_distance * math.cos(angle)
                y_offset = explosion_distance * math.sin(angle)
                scale_factor = 1.0 + (1.5 * progress)
                explosion_font_size = int(base_font_size * scale_factor)
                rotation_angle = rotation_speed * progress
                alpha = int(255 * (1.0 - 0.8 * progress))

                dialogue_line = self.renderer.create_exploding_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_start_x + int(x_offset), base_y + int(y_offset),
                    alpha, explosion_font_size, rotation_angle
                )
                dialogue_lines.append(dialogue_line)

        return dialogue_lines