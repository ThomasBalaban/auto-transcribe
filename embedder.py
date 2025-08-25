"""
Simplified embedder module for backward compatibility.
Points to the main subtitle processing functions.
"""

from core.subtitle_converter import convert_to_srt
from core.subtitle_embedder import embed_dual_subtitles, embed_single_subtitles, embed_subtitles
import utils.subtitle_styles as subtitle_styles
from onomatopoeia_detector import OnomatopoeiaDetector

__all__ = [
    "convert_to_srt",
    "embed_dual_subtitles", 
    "embed_single_subtitles",
    "embed_subtitles", 
    "subtitle_styles",
    "OnomatopoeiaDetector",
]