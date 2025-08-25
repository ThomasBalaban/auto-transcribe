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
    def create_mic_dialogue_line(start_time, end_time, text, font_size, style):
        """Creates a dialogue line for animated mic subs, using style for position."""
        start_formatted = TimeFormatter.format_ass_time(start_time)
        end_formatted = TimeFormatter.format_ass_time(end_time)
        
        # Only override the font size for the pop animation
        override_tags = f"{{\\fs{font_size}}}"
        styled_text = f"{override_tags}{text}"
        
        # Margins are 0,0,0 so the style's own margins are used for positioning
        return f"Dialogue: 0,{start_formatted},{end_formatted},{style},,0,0,0,,{styled_text}"

    @staticmethod
    def create_ass_dialogue_line_with_rotation(start_time, end_time, text, x, y, alpha, font_size, rotation=0, style="Onomatopoeia"):
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
        
        return f"Dialogue: 0,{start_formatted},{end_formatted},{style},,0,0,0,,{styled_text}"
    
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