"""
Subtitle style definitions for different tracks.
Contains configuration settings for various subtitle styles
that can be applied when embedding subtitles into videos.
"""

class MicrophoneStyle:
    """Style for Track 2 (Microphone) - Teal text with darker gray stroke, using BubbleGum font"""
    FONT_NAME = "BubbleGum"
    FONT_SIZE = 14
    PRIMARY_COLOR = "&H00d2ff00"  # Teal color
    OUTLINE_COLOR = "&H00171717"  # Dark gray
    BACKGROUND_COLOR = "&H00000000"  # Transparent
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1  # Outline
    OUTLINE = 3
    SHADOW = 1
    ALIGNMENT = 2  # Bottom-center alignment
    MARGIN_V = 70  # Higher value = higher position from bottom
    MARGIN_L = 40  # Left margin
    MARGIN_R = 40  # Right margin
    
    @classmethod
    def get_style_string(cls):
        """Generate the FFmpeg style string for subtitles"""
        return (
            f"FontName={cls.FONT_NAME},FontSize={cls.FONT_SIZE},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={cls.MARGIN_V},MarginL={cls.MARGIN_L},MarginR={cls.MARGIN_R}"
        )


class DesktopStyle:
    """Style for Track 3 (Desktop) - Black text on white background, positioned below mic subtitles"""
    FONT_NAME = "Arial"
    FONT_SIZE = 12
    PRIMARY_COLOR = "&H00000000"  # Black
    OUTLINE_COLOR = "&H00FFFFFF"  # White
    BACKGROUND_COLOR = "&HFFFFFFFF"  # White background
    BOLD = 1
    ITALIC = 0
    BORDER_STYLE = 1  # Background box
    OUTLINE = 3
    SHADOW = 1
    ALIGNMENT = 2  # Bottom-center
    MARGIN_V = 55  # Lower value = lower position (below mic track)
    MARGIN_L = 40  # Left margin
    MARGIN_R = 40  # Right margin
    
    @classmethod
    def get_style_string(cls):
        """Generate the FFmpeg style string for subtitles"""
        return (
            f"FontName={cls.FONT_NAME},FontSize={cls.FONT_SIZE},"
            f"PrimaryColour={cls.PRIMARY_COLOR},OutlineColour={cls.OUTLINE_COLOR},"
            f"BackColour={cls.BACKGROUND_COLOR},Bold={cls.BOLD},Italic={cls.ITALIC},"
            f"BorderStyle={cls.BORDER_STYLE},Outline={cls.OUTLINE},Shadow={cls.SHADOW},"
            f"Alignment={cls.ALIGNMENT},MarginV={cls.MARGIN_V},MarginL={cls.MARGIN_L},MarginR={cls.MARGIN_R}"
        )


class IntroTitleStyle:
    """Style for intro title text overlay - White box with black text, larger and higher"""
    # Font settings
    FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
    FONT_SIZE = 54
    FONT_COLOR = "black"
    
    # Box settings
    BOX_ENABLED = True
    BOX_COLOR = "white@0.9"
    BOX_BORDER_WIDTH = 15
    
    # Position settings
    POSITION_X = "(w-text_w)/2"  # Centered horizontally
    POSITION_Y = "500"  # Fixed position from top of the screen
    
    # Duration in seconds
    DEFAULT_DURATION = 5.0
    
    @classmethod
    def get_ffmpeg_filter(cls, text, duration=None):
        """Generate the FFmpeg filter string for the intro title"""
        if duration is None:
            duration = cls.DEFAULT_DURATION
            
        return (
            f"drawtext=fontfile={cls.FONT}:text='{text}':"
            f"fontsize={cls.FONT_SIZE}:fontcolor={cls.FONT_COLOR}:"
            f"box={1 if cls.BOX_ENABLED else 0}:boxcolor={cls.BOX_COLOR}:"
            f"boxborderw={cls.BOX_BORDER_WIDTH}:"
            f"x={cls.POSITION_X}:y={cls.POSITION_Y}:"
            f"enable='between(t,0,{duration})'"
        )

    @classmethod
    def format_title_text(cls, text, max_chars_per_line=20):
        """
        Split long title text into multiple lines with roughly equal length
        to ensure it doesn't extend beyond 75% of screen width
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # If adding this word would exceed max length and we have content
            if current_length + len(word) + 1 > max_chars_per_line and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1  # +1 for space
        
        # Add the last line if it has content
        if current_line:
            lines.append(" ".join(current_line))
        
        # Join lines with newline characters
        return "\\n".join(lines)


# For backward compatibility with existing code
TRACK2_STYLE = MicrophoneStyle.get_style_string()
TRACK3_STYLE = DesktopStyle.get_style_string()