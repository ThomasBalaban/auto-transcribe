"""
Onomatopoeia animation system for comic book-style effects.
Creates smooth animations using ASS (Advanced SubStation Alpha) format.
"""

import random
import math

class OnomatopoeiaAnimator:
    """Handles animation generation for onomatopoeia subtitles using ASS format with smooth animations"""
    
    # Animation constants
    DRIFT_DISTANCE = 50
    WIGGLE_AMPLITUDE = 15
    ANIMATION_FRAMES = 10  # Increased for smoother animation
    FRAME_DURATION = 0.05  # 10 frames × 0.05s = 0.5s total
    
    # Animation types
    DRIFT_FADE = "drift_fade"
    WIGGLE = "wiggle"
    
    @classmethod
    def get_random_animation_type(cls):
        """Randomly select an animation type with 50/50 probability."""
        return random.choice([cls.DRIFT_FADE, cls.WIGGLE])
    
    @classmethod
    def get_animation_type_from_setting(cls, animation_setting):
        """Get animation type based on UI setting."""
        if animation_setting == "Drift & Fade":
            return cls.DRIFT_FADE
        elif animation_setting == "Wiggle":
            return cls.WIGGLE
        else:  # "Random" or any other value
            return cls.get_random_animation_type()
    
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
            # Smooth progress from 0 to 1
            progress = frame / (cls.ANIMATION_FRAMES - 1)
            
            # Smooth upward drift using easing
            # Using ease-out curve for natural deceleration
            ease_progress = 1 - (1 - progress) ** 2
            
            x = base_x  # No horizontal movement
            y = base_y - int(cls.DRIFT_DISTANCE * ease_progress)  # Smooth upward drift
            alpha = int(255 * (1.0 - 0.8 * progress))  # Gradual fade out
            
            positions.append((x, y, alpha))
        return positions
    
    @classmethod
    def calculate_wiggle_positions(cls, base_x, base_y):
        """Calculate smooth positions for wiggle animation using sine wave."""
        positions = []
        for frame in range(cls.ANIMATION_FRAMES):
            # Calculate angle for sine wave (2 full cycles over the animation)
            angle = (frame / (cls.ANIMATION_FRAMES - 1)) * 4 * math.pi
            
            # Smooth sine wave motion
            x_offset = cls.WIGGLE_AMPLITUDE * math.sin(angle)
            
            x = base_x + int(x_offset)
            y = base_y  # No vertical movement
            alpha = 255  # No fading for wiggle
            
            positions.append((x, y, alpha))
        return positions
    
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
        """Create an ASS dialogue line with absolute positioning."""
        start_formatted = cls.format_ass_time(start_time)
        end_formatted = cls.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&\\fs{font_size}}}"
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @classmethod
    def create_animated_ass_events(cls, events, animation_setting="Random"):
        """Create ASS dialogue events for all onomatopoeia with smooth animations."""
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
            
            # Select and calculate animation based on UI setting
            animation_type = cls.get_animation_type_from_setting(animation_setting)
            if animation_type == cls.DRIFT_FADE:
                positions = cls.calculate_drift_positions(base_x, base_y)
            elif animation_type == cls.WIGGLE:
                positions = cls.calculate_wiggle_positions(base_x, base_y)
            else:
                positions = [(base_x, base_y, 255)] * cls.ANIMATION_FRAMES
            
            # Create dialogue lines for each frame
            for frame, (x, y, alpha) in enumerate(positions):
                frame_start = start_time + (frame * cls.FRAME_DURATION)
                frame_end = frame_start + cls.FRAME_DURATION
                
                if frame_end > end_time:
                    frame_end = end_time
                if frame_start >= end_time:
                    break
                
                dialogue_line = cls.create_ass_dialogue_line(
                    frame_start, frame_end, word, x, y, alpha, font_size
                )
                dialogue_lines.append(dialogue_line)
        
        return dialogue_lines
    
    @classmethod
    def generate_animated_ass_content(cls, events, animation_setting="Random"):
        """Generate complete ASS file content with smooth animations."""
        if not events:
            return cls.create_ass_header()
        
        ass_content = [cls.create_ass_header()]
        dialogue_lines = cls.create_animated_ass_events(events, animation_setting)
        
        for line in dialogue_lines:
            ass_content.append(line)
        
        return "\n".join(ass_content)


def create_animated_onomatopoeia_ass(audio_path, output_ass_path, animation_setting="Random", log_func=None):
    """Create an animated onomatopoeia ASS file from an audio file."""
    try:
        from onomatopoeia_detector import OnomatopoeiaDetector
        
        detector = OnomatopoeiaDetector(log_func=log_func)
        events = detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("No onomatopoeia events detected for animation")
            return False, []
        
        if log_func:
            log_func(f"Creating smooth animated onomatopoeia ASS file with {len(events)} events...")
            log_func(f"Animation type setting: {animation_setting}")
            log_func(f"Each effect will have {OnomatopoeiaAnimator.ANIMATION_FRAMES} smooth animation frames")
        
        animated_ass_content = OnomatopoeiaAnimator.generate_animated_ass_content(events, animation_setting)
        
        with open(output_ass_path, 'w', encoding='utf-8') as f:
            f.write(animated_ass_content)
        
        if log_func:
            log_func(f"Smooth animated onomatopoeia ASS file created: {output_ass_path}")
            log_func(f"Generated {len(events)} smooth animated sound effects")
            log_func(f"Total dialogue entries: {len(events) * OnomatopoeiaAnimator.ANIMATION_FRAMES}")
        
        return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating animated onomatopoeia ASS: {e}")
        return False, []