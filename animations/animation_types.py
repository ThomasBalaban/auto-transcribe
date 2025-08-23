"""
Animation type definitions and constants.
Contains the enum-like class for animation types and their string representations.
"""

class AnimationType:
    """Animation type constants - acts like an enum for animation types"""

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
    def get_all_types(cls):
        """Get all animation type constants."""
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
    def get_display_names(cls):
        """Get human-readable display names for animation types."""
        return {
            cls.DRIFT_FADE: "Drift & Fade",
            cls.WIGGLE: "Wiggle",
            cls.POP_SHRINK: "Pop & Shrink",
            cls.SHAKE: "Shake",
            cls.PULSE: "Pulse",
            cls.WAVE: "Wave",
            cls.EXPLODE_OUT: "Explode-Out",
            cls.HYPER_BOUNCE: "Hyper Bounce"
        }

    @classmethod
    def get_peak_frame(cls, animation_type):
        """
        Returns the frame number where the animation has the most visual impact.
        This is used to align the animation's peak with the audio event's peak.
        """
        peak_frames = {
            cls.DRIFT_FADE: 0,   # Impact is immediate
            cls.WIGGLE: 2,       # Peak of the first wiggle
            cls.POP_SHRINK: 2,   # The largest "pop"
            cls.SHAKE: 0,        # Impact is immediate
            cls.PULSE: 2,        # Peak of the first pulse
            cls.WAVE: 7,         # Middle of the wave crest
            cls.EXPLODE_OUT: 0,  # Impact is immediate
            cls.HYPER_BOUNCE: 2  # Peak of the first bounce
        }
        return peak_frames.get(animation_type, 0)


    @classmethod
    def get_description(cls, animation_type):
        """Get description of what each animation type does."""
        descriptions = {
            cls.DRIFT_FADE: "Text drifts upward while fading out",
            cls.WIGGLE: "Text wiggles side to side using sine wave",
            cls.POP_SHRINK: "Text pops larger then bounces back with rubber band effect",
            cls.SHAKE: "Text shakes violently with exponential decay and rotation",
            cls.PULSE: "Text pulses larger and smaller in smooth cycles",
            cls.WAVE: "Letters lean and move in a wave pattern (per-letter)",
            cls.EXPLODE_OUT: "Letters explode outward from center with rotation (per-letter)",
            cls.HYPER_BOUNCE: "Text bounces multiple times with decreasing amplitude"
        }
        return descriptions.get(animation_type, "Unknown animation type")

    @classmethod
    def requires_per_letter_rendering(cls, animation_type):
        """Check if animation type requires per-letter rendering."""
        per_letter_types = [cls.WAVE, cls.EXPLODE_OUT]
        return animation_type in per_letter_types

    @classmethod
    def is_valid_type(cls, animation_type):
        """Check if the given string is a valid animation type."""
        return animation_type in cls.get_all_types()