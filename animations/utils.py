"""
Utility functions and constants for the animation system.
Contains shared constants, time formatting, and other helper functions.
"""


class AnimationConstants:
    """Animation constants and configuration values"""
    
    # Animation parameters
    DRIFT_DISTANCE = 50
    WIGGLE_AMPLITUDE = 15
    SHAKE_AMPLITUDE = 8
    BOUNCE_HEIGHT = 30
    PULSE_SCALE_FACTOR = 1.3  # Reduced from 1.5 for smoother effect
    EXPLODE_DISTANCE = 80
    WAVE_AMPLITUDE = 25
    
    # Timing constants
    ANIMATION_FRAMES = 15  # Increased from 10 for smoother animations
    FRAME_DURATION = 0.033  # ~30fps for smoother motion (15 frames × 0.033s = 0.5s total)


class TimeFormatter:
    """Handles time formatting for different subtitle formats"""
    
    @staticmethod
    def format_ass_time(seconds):
        """Format time for ASS format: H:MM:SS.CC"""
        centiseconds = int((seconds - int(seconds)) * 100)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    @staticmethod
    def format_srt_time(seconds):
        """Format time for SRT format: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


class MathUtils:
    """Mathematical utility functions for animations"""
    
    @staticmethod
    def ease_in_out(t):
        """Standard ease-in-out function"""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def ease_out(t):
        """Ease-out function"""
        return 1 - (1 - t) ** 2
    
    @staticmethod
    def ease_in(t):
        """Ease-in function"""
        return t * t
    
    @staticmethod
    def bounce_ease(t):
        """Bouncing ease function"""
        if t < 1/2.75:
            return 7.5625 * t * t
        elif t < 2/2.75:
            t -= 1.5/2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625/2.75
            return 7.5625 * t * t + 0.984375
    
    @staticmethod
    def clamp(value, min_val, max_val):
        """Clamp a value between min and max"""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def lerp(a, b, t):
        """Linear interpolation between a and b"""
        return a + (b - a) * t


class ColorUtils:
    """Color utility functions for subtitle styling"""
    
    @staticmethod
    def rgb_to_ass_color(r, g, b):
        """Convert RGB values (0-255) to ASS color format"""
        return f"&H00{b:02X}{g:02X}{r:02X}"
    
    @staticmethod
    def alpha_to_ass_alpha(alpha):
        """Convert alpha value (0-255) to ASS alpha format"""
        return f"&H{255 - alpha:02X}&"
    
    @staticmethod
    def get_vibrant_colors():
        """Get a list of vibrant colors for onomatopoeia"""
        return [
            "&H0000FFFF",  # Bright yellow
            "&H0000FF00",  # Bright green  
            "&H00FF0000",  # Bright blue
            "&H00FF00FF",  # Bright magenta
            "&H0000CCFF",  # Orange
            "&H00FFFF00",  # Cyan
            "&H00FFFFFF",  # White
            "&H0000A5FF",  # Orange-red
        ]


class ValidationUtils:
    """Validation utilities for animation parameters"""
    
    @staticmethod
    def validate_time_range(start_time, end_time):
        """Validate that time range is valid"""
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time")
        return True
    
    @staticmethod
    def validate_position(x, y, max_x=1920, max_y=1080):
        """Validate that position is within screen bounds"""
        if x < 0 or x > max_x:
            raise ValueError(f"X position {x} is out of bounds (0-{max_x})")
        if y < 0 or y > max_y:
            raise ValueError(f"Y position {y} is out of bounds (0-{max_y})")
        return True
    
    @staticmethod
    def validate_alpha(alpha):
        """Validate that alpha value is in valid range"""
        if alpha < 0 or alpha > 255:
            raise ValueError(f"Alpha value {alpha} is out of range (0-255)")
        return True
    
    @staticmethod
    def validate_font_size(font_size, min_size=8, max_size=500):
        """Validate that font size is reasonable"""
        if font_size < min_size or font_size > max_size:
            raise ValueError(f"Font size {font_size} is out of range ({min_size}-{max_size})")
        return True


class DebugUtils:
    """Debug and logging utilities"""
    
    @staticmethod
    def log_animation_info(animation_type, frame_count, duration, log_func=None):
        """Log information about an animation"""
        if log_func:
            log_func(f"Animation: {animation_type}")
            log_func(f"Frames: {frame_count}")
            log_func(f"Duration: {duration:.3f}s")
            log_func(f"FPS: {frame_count/duration:.1f}")
    
    @staticmethod
    def log_position_data(positions, log_func=None):
        """Log position data for debugging"""
        if log_func and positions:
            log_func(f"Position data: {len(positions)} frames")
            for i, pos in enumerate(positions[:3]):  # Show first 3
                log_func(f"  Frame {i}: {pos}")
            if len(positions) > 3:
                log_func(f"  ... and {len(positions) - 3} more frames")