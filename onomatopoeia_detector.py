"""
Enhanced Onomatopoeia Detection System
Clean interface that imports the enhanced CLAP + Ollama system.
FIXED: Updated imports to match the new enhanced function names.
"""

# Import the enhanced system with correct function names
from integration_update import (
    AdaptiveOnomatopoeiaDetector as OnomatopoeiaDetector,
    create_onomatopoeia_srt as create_enhanced_onomatopoeia_srt
)

# Backward compatibility functions with updated calls
def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, 
                           use_animation=True, animation_setting="Random", 
                           ai_sensitivity=0.5):
    """Create onomatopoeia SRT using enhanced Ollama system."""
    return create_enhanced_onomatopoeia_srt(  # FIXED: Updated function name
        audio_path=audio_path,
        output_srt_path=output_srt_path,
        sensitivity=ai_sensitivity,
        animation_setting=animation_setting,
        log_func=log_func,
        use_animation=use_animation
    )

def create_modern_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, 
                                  use_animation=True, animation_setting="Random", 
                                  sensitivity=0.5):
    """Alias for backward compatibility."""
    return create_enhanced_onomatopoeia_srt(
        audio_path=audio_path,
        output_srt_path=output_srt_path,
        sensitivity=sensitivity,
        animation_setting=animation_setting,
        log_func=log_func,
        use_animation=use_animation
    )

def detect_onomatopoeia(audio_path, log_func=None, confidence_threshold=0.7):
    """Detect onomatopoeia using enhanced Ollama system."""
    detector = OnomatopoeiaDetector(sensitivity=confidence_threshold, log_func=log_func)
    return detector.analyze_audio_file(audio_path)

# For any legacy imports
YAMNET_AVAILABLE = True  # Always report as available for compatibility