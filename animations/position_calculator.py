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
            font_size = None  # No size change
            
            positions.append((x, y, alpha, font_size))
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
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    def calculate_pop_shrink_positions(self, base_x, base_y, base_font_size):
        """Calculate positions for pop and shrink animation with rubber band elasticity."""
        positions = []
        
        # Rubber band sequence: Pop → Snapback → Quick oscillations → Settle
        sizes = [
            base_font_size,           # Frame 0: Normal
            int(base_font_size * 1.8), # Frame 1: POP!
            int(base_font_size * 1.9), # Frame 2: Peak pop
            int(base_font_size * 0.7), # Frame 3: Big snapback
            int(base_font_size * 0.6), # Frame 4: Deeper snapback
            int(base_font_size * 1.1), # Frame 5: First bounce up
            int(base_font_size * 0.9), # Frame 6: Quick down
            int(base_font_size * 1.05),# Frame 7: Smaller bounce
            int(base_font_size * 0.95),# Frame 8: Smaller down
            int(base_font_size * 1.02),# Frame 9: Tiny bounce
            int(base_font_size * 0.98),# Frame 10: Tiny down
            base_font_size,           # Frame 11: Settle
            base_font_size,           # Frame 12-14: Hold steady
            base_font_size,
            base_font_size
        ]
        
        # Tiny rotation sequence (5° max)
        rotations = [
            0,    # Frame 0: Normal
            5,    # Frame 1: Pop rotation
            5,    # Frame 2: Hold pop rotation
            -3,   # Frame 3: Counter-rotate on snapback
            -3,   # Frame 4: Hold counter
            2,    # Frame 5: Small rotation
            -1,   # Frame 6: Small counter
            1,    # Frame 7: Tiny rotation
            -0.5, # Frame 8: Tiny counter
            0.5,  # Frame 9: Micro rotation
            0,    # Frame 10: Back to normal
            0, 0, 0, 0  # Frames 11-14: Steady
        ]
        
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            font_size = sizes[frame] if frame < len(sizes) else base_font_size
            rotation = rotations[frame] if frame < len(rotations) else 0
            
            x = base_x
            y = base_y
            alpha = 255
            
            positions.append((x, y, alpha, font_size, rotation))
        return positions
    
    def calculate_shake_positions(self, base_x, base_y):
        """Calculate positions for shake animation with exponential decay and rotation."""
        positions = []
        max_shake = AnimationConstants.SHAKE_AMPLITUDE
        max_rotation = 15  # degrees
        
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
            
            # Exponential decay - starts violent, calms down quickly
            intensity = (1 - progress) ** 2  # Exponential decay
            
            # Random shake within decreasing bounds
            shake_amount = max_shake * intensity
            rotation_amount = max_rotation * intensity
            
            x_offset = random.randint(-int(shake_amount), int(shake_amount))
            y_offset = random.randint(-int(shake_amount//2), int(shake_amount//2))
            rotation = random.uniform(-rotation_amount, rotation_amount)
            
            x = base_x + x_offset
            y = base_y + y_offset
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size, rotation))
        return positions
    
    def calculate_pulse_positions(self, base_x, base_y, base_font_size):
        """Calculate positions for pulse animation with smoother size changes."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            # Smoother sine wave for pulsing size with multiple cycles
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 4 * math.pi  # 2 full cycles
            # Use smoother sine curve
            size_variation = (math.sin(angle) + 1) / 2  # Normalize to 0-1
            size_multiplier = 1.0 + (AnimationConstants.PULSE_SCALE_FACTOR - 1.0) * size_variation
            font_size = int(base_font_size * size_multiplier)
            
            x = base_x
            y = base_y
            alpha = 255
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    def calculate_wave_positions(self, base_x, base_y, word_length):
        """Calculate positions for wave animation (simplified single-position version)."""
        # This is a fallback for when per-letter control isn't available
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            # Create wave effect by varying Y position
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 6 * math.pi  # 3 full cycles
            y_offset = AnimationConstants.WAVE_AMPLITUDE * math.sin(angle)
            
            x = base_x
            y = base_y + int(y_offset)
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    def calculate_explode_out_positions(self, base_x, base_y):
        """Calculate positions for explode-out animation (fallback for whole word)."""
        # This is a fallback when per-letter control isn't available
        positions = []
        angle = random.uniform(0, 2 * math.pi)
        
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
            
            # Move outward in random direction with ease-out
            ease_progress = 1 - (1 - progress) ** 2
            distance = AnimationConstants.EXPLODE_DISTANCE * ease_progress
            x_offset = distance * math.cos(angle)
            y_offset = distance * math.sin(angle)
            
            x = base_x + int(x_offset)
            y = base_y + int(y_offset)
            alpha = int(255 * (1.0 - 0.9 * progress))  # Fade out
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    def calculate_hyper_bounce_positions(self, base_x, base_y):
        """Calculate positions for hyper bounce animation."""
        positions = []
        for frame in range(AnimationConstants.ANIMATION_FRAMES):
            # Multiple bounces with decreasing amplitude
            bounce_cycles = 3  # Number of bounces
            angle = (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * bounce_cycles * 2 * math.pi
            
            # Decreasing amplitude over time
            amplitude_decay = 1.0 - (frame / (AnimationConstants.ANIMATION_FRAMES - 1)) * 0.3
            bounce_height = AnimationConstants.BOUNCE_HEIGHT * amplitude_decay
            
            y_offset = bounce_height * abs(math.sin(angle))
            x_jitter = random.randint(-3, 3)  # Small horizontal jitter
            
            x = base_x + x_jitter
            y = base_y - int(y_offset)  # Negative for upward bounce
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
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
            # Default fallback
            return self.calculate_drift_positions(base_x, base_y)
    
    def create_wave_per_letter_events(self, start_time, end_time, word, base_x, base_y, base_font_size):
        """Create wave animation with per-letter lean physics - single wave pass."""
        dialogue_lines = []
        letter_spacing = base_font_size * 0.6
        
        for letter_index, letter in enumerate(word):
            if letter.isspace():
                continue
                
            # Calculate letter position
            letter_x = base_x + int(letter_index * letter_spacing)
            
            # Create frames for this letter
            for frame in range(AnimationConstants.ANIMATION_FRAMES):
                frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                frame_end = frame_start + AnimationConstants.FRAME_DURATION
                
                if frame_end > end_time:
                    frame_end = end_time
                if frame_start >= end_time:
                    break
                
                # Single wave pass: wave travels from left to right through the word
                time_progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
                
                # Wave position moves from -1 (before first letter) to word_length+1 (after last letter)
                wave_position = -1 + (len(word) + 2) * time_progress
                
                # Distance from wave center to this letter
                distance_from_wave = abs(wave_position - letter_index)
                
                # Wave affects letters within a certain range
                wave_width = 1.5  # How wide the wave influence is
                
                if distance_from_wave <= wave_width:
                    # Letter is within wave influence
                    wave_intensity = math.cos((distance_from_wave / wave_width) * math.pi / 2)
                    
                    # Y position (vertical wave motion)
                    y_offset = AnimationConstants.WAVE_AMPLITUDE * wave_intensity
                    
                    # Rotation: lean based on wave direction
                    wave_direction = wave_position - letter_index
                    rotation = -30 * wave_direction * wave_intensity / wave_width
                    rotation = max(-30, min(30, rotation))  # Clamp to ±30°
                    
                else:
                    # Letter is not affected by wave
                    y_offset = 0
                    rotation = 0
                
                letter_y = base_y + int(y_offset)
                
                dialogue_line = self.renderer.create_wave_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, letter_y, 255, base_font_size, rotation
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
                
            # Calculate letter starting position
            letter_start_x = base_x + int(letter_index * letter_spacing)
            
            # Explosion direction - radiate outward from center
            dx = letter_start_x - word_center_x
            dy = 0  # Start at same Y level
            
            # If letter is at center, give it a random direction
            if abs(dx) < 5:
                angle = random.uniform(0, 2 * math.pi)
            else:
                # Calculate angle from center, add some randomness
                angle = math.atan2(dy, dx) + random.uniform(-0.5, 0.5)
            
            # Reduced rotation - max 180 degrees in either direction
            rotation_speed = random.uniform(-180, 180)  # degrees total rotation
            
            # Create frames for this letter
            for frame in range(AnimationConstants.ANIMATION_FRAMES):
                frame_start = start_time + (frame * AnimationConstants.FRAME_DURATION)
                frame_end = frame_start + AnimationConstants.FRAME_DURATION
                
                if frame_end > end_time:
                    frame_end = end_time
                if frame_start >= end_time:
                    break
                
                # Explosion calculation
                progress = frame / (AnimationConstants.ANIMATION_FRAMES - 1)
                
                # Use different easing for explosion - fast start, slow end
                ease_progress = progress ** 0.5  # Square root for explosion feel
                
                # Move outward from center with increased distance
                explosion_distance = (AnimationConstants.EXPLODE_DISTANCE * 1.5) * ease_progress  # 1.5x more separation
                x_offset = explosion_distance * math.cos(angle)
                y_offset = explosion_distance * math.sin(angle)
                
                letter_x = letter_start_x + int(x_offset)
                letter_y = base_y + int(y_offset)
                
                # Scale up as it explodes (letters get bigger)
                scale_factor = 1.0 + (1.5 * progress)  # Grow to 250% size
                explosion_font_size = int(base_font_size * scale_factor)
                
                # Rotation calculation - max 180 degrees total
                rotation_angle = rotation_speed * progress
                
                # Fade out as it explodes
                alpha = int(255 * (1.0 - 0.8 * progress))  # Keep more visible longer
                
                # Create ASS line with rotation and scaling
                dialogue_line = self.renderer.create_exploding_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, letter_y, 
                    alpha, explosion_font_size, rotation_angle
                )
                dialogue_lines.append(dialogue_line)
        
        return dialogue_lines