"""
Enhanced Onomatopoeia animation system for comic book-style effects.
Creates smooth animations using ASS (Advanced SubStation Alpha) format.
Now includes 8 different animation styles for varied visual effects.
"""

import random
import math

class OnomatopoeiaAnimator:
    """Handles animation generation for onomatopoeia subtitles using ASS format with multiple animation styles"""
    
    # Animation constants
    DRIFT_DISTANCE = 50
    WIGGLE_AMPLITUDE = 15
    SHAKE_AMPLITUDE = 8
    BOUNCE_HEIGHT = 30
    PULSE_SCALE_FACTOR = 1.3  # Reduced from 1.5 for smoother effect
    EXPLODE_DISTANCE = 80
    WAVE_AMPLITUDE = 25
    ANIMATION_FRAMES = 15  # Increased from 10 for smoother animations
    FRAME_DURATION = 0.033  # ~30fps for smoother motion (15 frames × 0.033s = 0.5s total)
    
    # Animation types
    DRIFT_FADE = "drift_fade"
    WIGGLE = "wiggle"
    POP_SHRINK = "pop_shrink"
    SHAKE = "shake"
    PULSE = "pulse"
    WAVE = "wave"
    EXPLODE_OUT = "explode_out"
    HYPER_BOUNCE = "hyper_bounce"
    
    @classmethod
    def get_all_animation_types(cls):
        """Get list of all available animation types."""
        return [
            cls.DRIFT_FADE,
            cls.WIGGLE,
            cls.POP_SHRINK,
            cls.SHAKE,
            cls.PULSE,
            cls.WAVE,
            cls.EXPLODE_OUT,
            cls.HYPER_BOUNCE
        ]
    
    @classmethod
    def get_random_animation_type(cls):
        """Randomly select an animation type from all available options."""
        return random.choice(cls.get_all_animation_types())
    
    @classmethod
    def get_animation_type_from_setting(cls, animation_setting):
        """Get animation type based on UI setting."""
        animation_map = {
            "Drift & Fade": cls.DRIFT_FADE,
            "Wiggle": cls.WIGGLE,
            "Pop & Shrink": cls.POP_SHRINK,
            "Shake": cls.SHAKE,
            "Pulse": cls.PULSE,
            "Wave": cls.WAVE,
            "Explode-Out": cls.EXPLODE_OUT,
            "Hyper Bounce": cls.HYPER_BOUNCE,
            "Random": cls.get_random_animation_type()
        }
        return animation_map.get(animation_setting, cls.get_random_animation_type())
    
    @classmethod
    def create_ass_header(cls):
        """Create ASS file header with proper video resolution."""
        return """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Onomatopoeia,Bold Marker,140,&H00FFFFFF,&H00000000,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,2,5,50,50,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    @classmethod
    def calculate_drift_positions(cls, base_x, base_y):
        """Calculate smooth positions for drift and fade animation."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            progress = frame / (cls.ANIMATION_FRAMES - 1)
            ease_progress = 1 - (1 - progress) ** 2
            
            x = base_x
            y = base_y - int(cls.DRIFT_DISTANCE * ease_progress)
            alpha = int(255 * (1.0 - 0.8 * progress))
            font_size = None  # No size change
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def calculate_wiggle_positions(cls, base_x, base_y):
        """Calculate smooth positions for wiggle animation using sine wave."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            angle = (frame / (cls.ANIMATION_FRAMES - 1)) * 4 * math.pi
            x_offset = cls.WIGGLE_AMPLITUDE * math.sin(angle)
            
            x = base_x + int(x_offset)
            y = base_y
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def calculate_pop_shrink_positions(cls, base_x, base_y, base_font_size):
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
        
        for frame in range(cls.ANIMATION_FRAMES):
            font_size = sizes[frame] if frame < len(sizes) else base_font_size
            rotation = rotations[frame] if frame < len(rotations) else 0
            
            x = base_x
            y = base_y
            alpha = 255
            
            positions.append((x, y, alpha, font_size, rotation))
        return positions
    
    @classmethod
    def calculate_shake_positions(cls, base_x, base_y):
        """Calculate positions for shake animation with exponential decay and rotation."""
        positions = []
        max_shake = cls.SHAKE_AMPLITUDE
        max_rotation = 15  # degrees
        
        for frame in range(cls.ANIMATION_FRAMES):
            progress = frame / (cls.ANIMATION_FRAMES - 1)
            
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
    
    @classmethod
    def calculate_pulse_positions(cls, base_x, base_y, base_font_size):
        """Calculate positions for pulse animation with smoother size changes."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            # Smoother sine wave for pulsing size with multiple cycles
            angle = (frame / (cls.ANIMATION_FRAMES - 1)) * 4 * math.pi  # 2 full cycles
            # Use smoother sine curve
            size_variation = (math.sin(angle) + 1) / 2  # Normalize to 0-1
            size_multiplier = 1.0 + (cls.PULSE_SCALE_FACTOR - 1.0) * size_variation
            font_size = int(base_font_size * size_multiplier)
            
            x = base_x
            y = base_y
            alpha = 255
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def calculate_wave_positions(cls, base_x, base_y, word_length):
        """Calculate positions for wave animation (simplified single-position version)."""
        # This is a fallback for when per-letter control isn't available
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            # Create wave effect by varying Y position
            angle = (frame / (cls.ANIMATION_FRAMES - 1)) * 6 * math.pi  # 3 full cycles
            y_offset = cls.WAVE_AMPLITUDE * math.sin(angle)
            
            x = base_x
            y = base_y + int(y_offset)
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def create_wave_per_letter_events(cls, start_time, end_time, word, base_x, base_y, base_font_size):
        """Create wave animation with per-letter lean physics - single wave pass."""
        dialogue_lines = []
        letter_spacing = base_font_size * 0.6
        
        for letter_index, letter in enumerate(word):
            if letter.isspace():
                continue
                
            # Calculate letter position
            letter_x = base_x + int(letter_index * letter_spacing)
            
            # Create frames for this letter
            for frame in range(cls.ANIMATION_FRAMES):
                frame_start = start_time + (frame * cls.FRAME_DURATION)
                frame_end = frame_start + cls.FRAME_DURATION
                
                if frame_end > end_time:
                    frame_end = end_time
                if frame_start >= end_time:
                    break
                
                # Single wave pass: wave travels from left to right through the word
                time_progress = frame / (cls.ANIMATION_FRAMES - 1)
                
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
                    y_offset = cls.WAVE_AMPLITUDE * wave_intensity
                    
                    # Rotation: lean based on wave direction
                    # If wave is approaching (wave_position < letter_index): lean away (+)
                    # If wave is leaving (wave_position > letter_index): lean toward (-)
                    wave_direction = wave_position - letter_index
                    rotation = -30 * wave_direction * wave_intensity / wave_width
                    rotation = max(-30, min(30, rotation))  # Clamp to ±30°
                    
                else:
                    # Letter is not affected by wave
                    y_offset = 0
                    rotation = 0
                
                letter_y = base_y + int(y_offset)
                
                dialogue_line = cls.create_wave_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, letter_y, 255, base_font_size, rotation
                )
                dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
    @classmethod
    def create_wave_ass_dialogue_line(cls, start_time, end_time, text, x, y, alpha, font_size, rotation_degrees):
        """Create an ASS dialogue line with rotation for wave lean effect."""
        start_formatted = cls.format_ass_time(start_time)
        end_formatted = cls.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags with rotation
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&\\fs{font_size}\\frz{rotation_degrees:.1f}}}"
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @classmethod
    def calculate_explode_out_positions(cls, base_x, base_y):
        """Calculate positions for explode-out animation (fallback for whole word)."""
        # This is a fallback when per-letter control isn't available
        positions = []
        angle = random.uniform(0, 2 * math.pi)
        
        for frame in range(cls.ANIMATION_FRAMES):
            progress = frame / (cls.ANIMATION_FRAMES - 1)
            
            # Move outward in random direction with ease-out
            ease_progress = 1 - (1 - progress) ** 2
            distance = cls.EXPLODE_DISTANCE * ease_progress
            x_offset = distance * math.cos(angle)
            y_offset = distance * math.sin(angle)
            
            x = base_x + int(x_offset)
            y = base_y + int(y_offset)
            alpha = int(255 * (1.0 - 0.9 * progress))  # Fade out
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def create_explode_per_letter_events(cls, start_time, end_time, word, base_x, base_y, base_font_size):
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
            initial_scale = 1.0
            
            # Create frames for this letter
            for frame in range(cls.ANIMATION_FRAMES):
                frame_start = start_time + (frame * cls.FRAME_DURATION)
                frame_end = frame_start + cls.FRAME_DURATION
                
                if frame_end > end_time:
                    frame_end = end_time
                if frame_start >= end_time:
                    break
                
                # Explosion calculation
                progress = frame / (cls.ANIMATION_FRAMES - 1)
                
                # Use different easing for explosion - fast start, slow end
                ease_progress = progress ** 0.5  # Square root for explosion feel
                
                # Move outward from center with increased distance
                explosion_distance = (cls.EXPLODE_DISTANCE * 1.5) * ease_progress  # 1.5x more separation
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
                dialogue_line = cls.create_exploding_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, letter_y, 
                    alpha, explosion_font_size, rotation_angle
                )
                dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
    @classmethod
    def create_exploding_ass_dialogue_line(cls, start_time, end_time, text, x, y, alpha, font_size, rotation_degrees):
        """Create an ASS dialogue line with rotation and scaling for explosion effect."""
        start_formatted = cls.format_ass_time(start_time)
        end_formatted = cls.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags with rotation and scaling
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&\\fs{font_size}\\frz{rotation_degrees:.1f}}}"
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @classmethod
    def calculate_hyper_bounce_positions(cls, base_x, base_y):
        """Calculate positions for hyper bounce animation."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            # Multiple bounces with decreasing amplitude
            bounce_cycles = 3  # Number of bounces
            angle = (frame / (cls.ANIMATION_FRAMES - 1)) * bounce_cycles * 2 * math.pi
            
            # Decreasing amplitude over time
            amplitude_decay = 1.0 - (frame / (cls.ANIMATION_FRAMES - 1)) * 0.3
            bounce_height = cls.BOUNCE_HEIGHT * amplitude_decay
            
            y_offset = bounce_height * abs(math.sin(angle))
            x_jitter = random.randint(-3, 3)  # Small horizontal jitter
            
            x = base_x + x_jitter
            y = base_y - int(y_offset)  # Negative for upward bounce
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def calculate_animation_positions(cls, animation_type, base_x, base_y, base_font_size=140, word_length=5):
        """Calculate positions for any animation type."""
        if animation_type == cls.DRIFT_FADE:
            return cls.calculate_drift_positions(base_x, base_y)
        elif animation_type == cls.WIGGLE:
            return cls.calculate_wiggle_positions(base_x, base_y)
        elif animation_type == cls.POP_SHRINK:
            return cls.calculate_pop_shrink_positions(base_x, base_y, base_font_size)
        elif animation_type == cls.SHAKE:
            return cls.calculate_shake_positions(base_x, base_y)
        elif animation_type == cls.PULSE:
            return cls.calculate_pulse_positions(base_x, base_y, base_font_size)
        elif animation_type == cls.WAVE:
            return cls.calculate_wave_positions(base_x, base_y, word_length)
        elif animation_type == cls.EXPLODE_OUT:
            return cls.calculate_explode_out_positions(base_x, base_y)
        elif animation_type == cls.HYPER_BOUNCE:
            return cls.calculate_hyper_bounce_positions(base_x, base_y)
        else:
            # Default fallback
            return cls.calculate_drift_positions(base_x, base_y)
    
    @classmethod
    def create_ass_dialogue_line_with_rotation(cls, start_time, end_time, text, x, y, alpha, font_size, rotation=0):
        """Create an ASS dialogue line with optional rotation."""
        start_formatted = cls.format_ass_time(start_time)
        end_formatted = cls.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&"
        if font_size is not None:
            override_tags += f"\\fs{font_size}"
        if rotation != 0:
            override_tags += f"\\frz{rotation:.1f}"
        override_tags += "}"
        
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @classmethod
    def format_ass_time(cls, seconds):
        """Format time for ASS format: H:MM:SS.CC"""
        centiseconds = int((seconds - int(seconds)) * 100)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    @classmethod
    def create_ass_dialogue_line(cls, start_time, end_time, text, x, y, alpha, font_size):
        """Create an ASS dialogue line with absolute positioning and optional font size."""
        start_formatted = cls.format_ass_time(start_time)
        end_formatted = cls.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags properly
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&"
        if font_size is not None:
            override_tags += f"\\fs{font_size}"
        override_tags += "}"  # Single closing brace, not double
        
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @classmethod
    def create_animated_ass_events(cls, events, animation_setting="Random"):
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
            animation_type = cls.get_animation_type_from_setting(animation_setting)
            
            # Handle per-letter animations differently
            if animation_type == cls.WAVE:
                # Use per-letter wave animation
                letter_lines = cls.create_wave_per_letter_events(
                    start_time, end_time, word, base_x, base_y, font_size
                )
                dialogue_lines.extend(letter_lines)
                
            elif animation_type == cls.EXPLODE_OUT:
                # Use per-letter explode animation
                letter_lines = cls.create_explode_per_letter_events(
                    start_time, end_time, word, base_x, base_y, font_size
                )
                dialogue_lines.extend(letter_lines)
                
            else:
                # Use standard whole-word animations
                positions = cls.calculate_animation_positions(
                    animation_type, base_x, base_y, font_size, len(word)
                )
                
                # Create dialogue lines for each frame
                for frame, position_data in enumerate(positions):
                    if len(position_data) == 5:  # Has rotation
                        x, y, alpha, frame_font_size, rotation = position_data
                    else:  # No rotation (legacy format)
                        x, y, alpha, frame_font_size = position_data
                        rotation = 0
                    
                    frame_start = start_time + (frame * cls.FRAME_DURATION)
                    frame_end = frame_start + cls.FRAME_DURATION
                    
                    if frame_end > end_time:
                        frame_end = end_time
                    if frame_start >= end_time:
                        break
                    
                    # Use frame-specific font size if provided, otherwise use base
                    final_font_size = frame_font_size if frame_font_size is not None else font_size
                    
                    dialogue_line = cls.create_ass_dialogue_line_with_rotation(
                        frame_start, frame_end, word, x, y, alpha, final_font_size, rotation
                    )
                    dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
    @classmethod
    def generate_animated_ass_content(cls, events, animation_setting="Random"):
        """Generate complete ASS file content with various animation styles."""
        if not events:
            return cls.create_ass_header()
        
        ass_content = [cls.create_ass_header()]
        dialogue_lines = cls.create_animated_ass_events(events, animation_setting)
        
        for line in dialogue_lines:
            ass_content.append(line)
        
        return "\n".join(ass_content)


def create_animated_onomatopoeia_ass(audio_path, output_ass_path, animation_setting="Random", log_func=None):
    """Create an animated onomatopoeia ASS file from an audio file with enhanced animation options."""
    try:
        from onomatopoeia_detector import OnomatopoeiaDetector
        
        detector = OnomatopoeiaDetector(log_func=log_func)
        events = detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("No onomatopoeia events detected for animation")
            return False, []
        
        if log_func:
            log_func(f"Creating animated onomatopoeia ASS file with {len(events)} events...")
            log_func(f"Animation type setting: {animation_setting}")
            log_func(f"Available animations: {', '.join(OnomatopoeiaAnimator.get_all_animation_types())}")
            log_func(f"Each effect will have {OnomatopoeiaAnimator.ANIMATION_FRAMES} animation frames")
        
        animated_ass_content = OnomatopoeiaAnimator.generate_animated_ass_content(events, animation_setting)
        
        with open(output_ass_path, 'w', encoding='utf-8') as f:
            f.write(animated_ass_content)
        
        if log_func:
            log_func(f"Enhanced animated onomatopoeia ASS file created: {output_ass_path}")
            log_func(f"Generated {len(events)} animated sound effects")
            log_func(f"Total dialogue entries: {len(events) * OnomatopoeiaAnimator.ANIMATION_FRAMES}")
        
        return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating animated onomatopoeia ASS: {e}")
        return False, []