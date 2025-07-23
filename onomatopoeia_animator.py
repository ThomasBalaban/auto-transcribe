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
        """Calculate positions for pop and shrink animation with smoother transitions."""
        positions = []
        max_size = int(base_font_size * 1.4)  # Reduced from 1.8 for less jarring effect
        
        for frame in range(cls.ANIMATION_FRAMES):
            progress = frame / (cls.ANIMATION_FRAMES - 1)
            
            # Smoother pop and shrink using ease-out curve
            if frame <= 2:  # Pop phase - first 3 frames
                pop_progress = frame / 2
                font_size = int(base_font_size + (max_size - base_font_size) * pop_progress)
            else:  # Shrink phase - remaining frames with ease-out
                shrink_progress = (frame - 2) / (cls.ANIMATION_FRAMES - 3)
                # Ease-out curve for smooth shrinking
                ease_progress = 1 - (1 - shrink_progress) ** 3
                font_size = int(max_size - (max_size - base_font_size) * ease_progress)
            
            x = base_x
            y = base_y
            alpha = 255
            
            positions.append((x, y, alpha, font_size))
        return positions
    
    @classmethod
    def calculate_shake_positions(cls, base_x, base_y):
        """Calculate positions for shake animation with random jitter."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            # Random shake within amplitude
            x_offset = random.randint(-cls.SHAKE_AMPLITUDE, cls.SHAKE_AMPLITUDE)
            y_offset = random.randint(-cls.SHAKE_AMPLITUDE//2, cls.SHAKE_AMPLITUDE//2)
            
            x = base_x + x_offset
            y = base_y + y_offset
            alpha = 255
            font_size = None
            
            positions.append((x, y, alpha, font_size))
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
        """Create wave animation with per-letter control."""
        dialogue_lines = []
        letter_spacing = base_font_size * 0.6  # Approximate letter width
        
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
                
                # Wave calculation: each letter follows the wave with a phase offset
                time_progress = frame / (cls.ANIMATION_FRAMES - 1)
                wave_phase = (time_progress * 4 * math.pi) + (letter_index * math.pi / 2)  # Phase offset per letter
                y_offset = cls.WAVE_AMPLITUDE * math.sin(wave_phase)
                
                letter_y = base_y + int(y_offset)
                
                dialogue_line = cls.create_ass_dialogue_line(
                    frame_start, frame_end, letter, letter_x, letter_y, 255, base_font_size
                )
                dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
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
                    x, y, alpha, frame_font_size = position_data
                    
                    frame_start = start_time + (frame * cls.FRAME_DURATION)
                    frame_end = frame_start + cls.FRAME_DURATION
                    
                    if frame_end > end_time:
                        frame_end = end_time
                    if frame_start >= end_time:
                        break
                    
                    # Use frame-specific font size if provided, otherwise use base
                    final_font_size = frame_font_size if frame_font_size is not None else font_size
                    
                    dialogue_line = cls.create_ass_dialogue_line(
                        frame_start, frame_end, word, x, y, alpha, final_font_size
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