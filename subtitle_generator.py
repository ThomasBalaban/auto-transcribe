"""
Subtitle Generation Module for Onomatopoeia Detection.
Handles creation of SRT and ASS subtitle files.
"""

import os
from typing import List, Dict


class SubtitleGenerator:
    """
    Handles creation of subtitle files with onomatopoeia effects.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print

    def create_subtitle_file(self, events: List[Dict], output_path: str,
                           animation_type: str = "Random") -> bool:
        """
        Create subtitle file with onomatopoeia effects.

        Args:
            events: List of onomatopoeia events
            output_path: Output subtitle file (.srt or .ass)
            animation_type: Animation type for effects (can be "Intelligent")

        Returns:
            Success status
        """
        try:
            if not events:
                self.log_func("No onomatopoeia events to write")
                return False

            # Create output file
            file_ext = os.path.splitext(output_path)[1].lower()

            if file_ext == '.ass' or animation_type != "Static":
                # Create animated ASS file
                ass_path = os.path.splitext(output_path)[0] + '.ass'
                # FIX: Pass the animation_type to the creation method
                return self._create_animated_subtitle_file(events, ass_path, animation_type)
            else:
                # Create static SRT file
                return self._create_static_subtitle_file(events, output_path)

        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return False

    def _create_animated_subtitle_file(self, events: List[Dict], output_path: str,
                                     animation_type: str) -> bool:
        """Create animated ASS subtitle file."""
        try:
            from animations.core import OnomatopoeiaAnimator

            animator = OnomatopoeiaAnimator()
            # FIX: The animation type is now decided within the fusion engine and stored
            # in each event. The animator reads it from there, so we no longer
            # need to pass animation_type as an argument here.
            animated_content = animator.generate_animated_ass_content(events)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(animated_content)

            self.log_func(f"✅ Animated subtitle file created: {output_path}")
            return True

        except Exception as e:
            self.log_func(f"Error creating animated subtitle file: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return False

    def _create_static_subtitle_file(self, events: List[Dict], output_path: str) -> bool:
        """Create static SRT subtitle file."""
        try:
            srt_content = self._generate_srt_content(events)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            self.log_func(f"✅ Static subtitle file created: {output_path}")
            return True

        except Exception as e:
            self.log_func(f"Error creating static subtitle file: {e}")
            return False

    def _generate_srt_content(self, events: List[Dict]) -> str:
        """Generate SRT content from events"""
        if not events:
            return ""

        srt_lines = []
        for i, event in enumerate(events, 1):
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']

            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)

            srt_lines.extend([
                str(i),
                f"{start_formatted} --> {end_formatted}",
                word,
                ""
            ])

        return "\n".join(srt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

_onomatopoeia_lines: List[Dict] = []

def clear_onomatopoeia() -> None:
    _onomatopoeia_lines.clear()

def add_onomatopoeia(word, start, peak, end, cls, ctx) -> None:
    _onomatopoeia_lines.append({
        "text": str(word),
        "start": float(start),
        "peak": float(peak),
        "end": float(end),
        "class": str(cls),
        "context": list(ctx) if ctx is not None else [],
    })

def get_onomatopoeia_lines() -> List[Dict]:
    return list(_onomatopoeia_lines)

def get_events_for_generator() -> List[Dict]:
    """
    Adapt the collected lines into the shape SubtitleGenerator.create_subtitle_file expects,
    and also include fields useful for the ASS animator.
    """
    events = []
    for it in _onomatopoeia_lines:
        events.append({
            "word": it["text"],
            "start_time": it["start"],
            "end_time": it["end"],
            "peak_time": it.get("peak"),
            "class": it.get("class"),
            "context": it.get("context", []),
        })
    return events