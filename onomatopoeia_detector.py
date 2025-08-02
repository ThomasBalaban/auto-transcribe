"""
Modern Onomatopoeia Detection System
Clean interface that imports the modern CLAP + LLM system.
"""

# Import the modern system
from modern_onomatopoeia_detector import (
    ModernOnomatopoeiaDetector as OnomatopoeiaDetector,
    create_modern_onomatopoeia_srt
)

# Backward compatibility functions
def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, 
                           use_animation=True, animation_setting="Random", 
                           ai_sensitivity=0.5):
    """Create onomatopoeia SRT using modern system."""
    return create_modern_onomatopoeia_srt(
        audio_path=audio_path,
        output_srt_path=output_srt_path,
        sensitivity=ai_sensitivity,
        animation_setting=animation_setting,
        log_func=log_func,
        use_animation=use_animation
    )

def detect_onomatopoeia(audio_path, log_func=None, confidence_threshold=0.7):
    """Detect onomatopoeia using modern system."""
    detector = OnomatopoeiaDetector(sensitivity=confidence_threshold, log_func=log_func)
    return detector.analyze_audio_file(audio_path)

# For any legacy imports
YAMNET_AVAILABLE = True  # Always report as available for compatibility