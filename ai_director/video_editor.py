# ai_director/video_editor.py - FINAL CORRECTED VERSION
"""
The Video Editor for the AI Director system.

This module is responsible for executing the decisions made by the Master Director.
It takes the final Decision Timeline and uses ffmpeg to apply all specified
zoom and pan effects to the video, creating the final edited output.
"""

import subprocess
import os
from typing import List
from ai_director.data_models import TimelineEvent

class VideoEditor:
    """Executes the editing timeline to produce the final video."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.log_func("📹 AI Director Video Editor initialized.")

    def apply_edits(self, input_video: str, output_video: str, timeline: List[TimelineEvent]):
        """
        Applies the entire series of edits from the timeline to the input video.
        """
        if not timeline:
            self.log_func("No edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return

        applicable_events = [e for e in timeline if e.action in ["zoom_to_cam", "zoom_to_game"]]

        if not applicable_events:
            self.log_func("No applicable zoom edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return
            
        self.log_func(f"Applying {len(applicable_events)} edits to the video...")

        filter_chains = []
        video_streams = "[0:v]"

        for i, event in enumerate(applicable_events):
            start_time = event.timestamp
            end_time = start_time + event.duration
            
            if event.action == "zoom_to_cam":
                # 2x zoom on the top-left corner, maintains aspect ratio
                crop_filter = "crop=iw/2:ih/2:0:0"
            elif event.action == "zoom_to_game":
                # --- CORRECTED ZOOM ---
                # 1.5x zoom, centered on the screen, maintains aspect ratio
                crop_filter = "crop=iw/1.5:ih/1.5:iw/6:ih/6"
            
            from_stream = video_streams
            base_stream = f"[base{i}]"
            to_zoom_stream = f"[to_zoom{i}]"
            zoomed_stream = f"[zoomed{i}]"
            video_streams = f"[v{i+1}]"

            chain = (
                f"{from_stream}split={base_stream}{to_zoom_stream};"
                f"{to_zoom_stream}{crop_filter},scale=1080:1920{zoomed_stream};"
                f"{base_stream}{zoomed_stream}overlay=0:0:enable='between(t,{start_time},{end_time})'{video_streams}"
            )
            filter_chains.append(chain)

        full_filtergraph = ";".join(filter_chains)

        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-filter_complex', full_filtergraph,
            '-map', video_streams,
            # --- CORRECTED AUDIO MAPPING ---
            # Map all audio streams from the first input (0)
            '-map', '0:a', 
            '-c:a', 'copy',
            output_video
        ]

        try:
            self.log_func("Executing ffmpeg command for multi-edit video processing...")
            self.log_func(f"Command: {' '.join(ffmpeg_command)}")
            result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            self.log_func("✅ Video editing complete.")
            self.log_func(f"FFmpeg output:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            self.log_func("❌ ERROR: ffmpeg command timed out.")
        except subprocess.CalledProcessError as e:
            self.log_func("❌ ERROR during video editing.")
            self.log_func(f"FFmpeg stderr:\n{e.stderr}")
            import shutil
            shutil.copy(input_video, output_video)