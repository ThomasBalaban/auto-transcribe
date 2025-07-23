"""
ASS file rendering and formatting utilities.
Handles the creation of ASS subtitle files with proper formatting.
"""

from .utils import TimeFormatter


class ASSRenderer:
    """Handles ASS file generation and formatting"""
    
    @staticmethod
    def create_ass_header():
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
    
    @staticmethod
    def create_ass_dialogue_line(start_time, end_time, text, x, y, alpha, font_size):
        """Create an ASS dialogue line with absolute positioning and optional font size."""
        start_formatted = TimeFormatter.format_ass_time(start_time)
        end_formatted = TimeFormatter.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags properly
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&"
        if font_size is not None:
            override_tags += f"\\fs{font_size}"
        override_tags += "}"  # Single closing brace, not double
        
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @staticmethod
    def create_ass_dialogue_line_with_rotation(start_time, end_time, text, x, y, alpha, font_size, rotation=0):
        """Create an ASS dialogue line with optional rotation."""
        start_formatted = TimeFormatter.format_ass_time(start_time)
        end_formatted = TimeFormatter.format_ass_time(end_time)
        
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
    
    @staticmethod
    def create_wave_ass_dialogue_line(start_time, end_time, text, x, y, alpha, font_size, rotation_degrees):
        """Create an ASS dialogue line with rotation for wave lean effect."""
        start_formatted = TimeFormatter.format_ass_time(start_time)
        end_formatted = TimeFormatter.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags with rotation
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&\\fs{font_size}\\frz{rotation_degrees:.1f}}}"
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"
    
    @staticmethod
    def create_exploding_ass_dialogue_line(start_time, end_time, text, x, y, alpha, font_size, rotation_degrees):
        """Create an ASS dialogue line with rotation and scaling for explosion effect."""
        start_formatted = TimeFormatter.format_ass_time(start_time)
        end_formatted = TimeFormatter.format_ass_time(end_time)
        
        alpha_hex = f"{255 - alpha:02X}"
        
        # Build override tags with rotation and scaling
        override_tags = f"{{\\pos({x},{y})\\alpha&H{alpha_hex}&\\fs{font_size}\\frz{rotation_degrees:.1f}}}"
        styled_text = f"{override_tags}{text}"
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},Onomatopoeia,,0,0,0,,{styled_text}"


def create_animated_onomatopoeia_ass(audio_path, output_ass_path, animation_setting="Random", log_func=None):
    """Create an animated onomatopoeia ASS file from an audio file with enhanced animation options."""
    try:
        from onomatopoeia_detector import OnomatopoeiaDetector
        from .core import OnomatopoeiaAnimator
        
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
            from .utils import AnimationConstants
            log_func(f"Each effect will have {AnimationConstants.ANIMATION_FRAMES} animation frames")
        
        animator = OnomatopoeiaAnimator()
        animated_ass_content = animator.generate_animated_ass_content(events, animation_setting)
        
        with open(output_ass_path, 'w', encoding='utf-8') as f:
            f.write(animated_ass_content)
        
        if log_func:
            log_func(f"Enhanced animated onomatopoeia ASS file created: {output_ass_path}")
            log_func(f"Generated {len(events)} animated sound effects")
            from .utils import AnimationConstants
            log_func(f"Total dialogue entries: {len(events) * AnimationConstants.ANIMATION_FRAMES}")
        
        return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating animated onomatopoeia ASS: {e}")
        return False, []