"""
Onomatopoeia Animation System
Enhanced animation system for comic book-style effects.
"""

# Import main classes and functions for public API
from .core import OnomatopoeiaAnimator
from .animation_types import AnimationType
from .renderer import create_animated_onomatopoeia_ass

# Backward compatibility - maintain existing interface
__all__ = [
    'OnomatopoeiaAnimator',
    'AnimationType', 
    'create_animated_onomatopoeia_ass'
]

# Version info
__version__ = "2.0.0"