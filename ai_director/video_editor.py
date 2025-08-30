# ai_director/video_editor.py - CORRECTED FOR AUDIO AND STRETCHING
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

        applicable_events = [e for e in timeline if e.action in ["zoom_to_cam", "zoom_to_game", "zoom_out"]]

        if not applicable_events:
            self.log_func("No applicable edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return
            
        self.log_func(f"Applying {len(applicable_events)} edits to the video...")

        filter_chains = []
        video_streams = "[0:v]"

        for i, event in enumerate(applicable_events):
            start_time = event.timestamp
            end_time = start_time + event.duration
            
            effect_filter = ""
            
            if event.action == "zoom_to_cam":
                # 2x zoom on the top-left corner, then scale back to full screen
                effect_filter = "crop=iw/2:ih/2:0:0,scale=1080:1920"
            elif event.action == "zoom_to_game":
                # --- CORRECTED ZOOM (NO STRETCH) ---
                # 1.5x zoom, centered on the screen, maintains aspect ratio before scaling
                effect_filter = "crop=iw/1.5:ih/1.5:iw/6:ih/6,scale=1080:1920"
            elif event.action == "zoom_out":
                # This logic is now handled in the overlay section
                pass

            from_stream = video_streams
            base_stream = f"[base{i}]"
            to_effect_stream = f"[to_effect{i}]"
            effect_stream = f"[effect{i}]"
            video_streams = f"[v{i+1}]"

            # Build the filter chain for this event
            chain = ""
            if event.action == "zoom_out":
                # For zoom out, we create a special composition
                temp_stream = f"[temp{i}]"
                chain = (
                    f"{from_stream}split[main_base][main_for_effect];"
                    f"[main_for_effect]boxblur=50[blurred];"
                    f"[main_for_effect]scale=iw*0.8:ih*0.8[scaled_fg];"
                    f"[main_base][blurred]overlay=0:0:enable='between(t,{start_time},{end_time})'{temp_stream};"
                    f"{temp_stream}[scaled_fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:enable='between(t,{start_time},{end_time})'{video_streams}"
                )
            else:
                 # Standard logic for zoom-in effects
                chain = (
                    f"{from_stream}split={base_stream}{to_effect_stream};"
                    f"{to_effect_stream}{effect_filter}{effect_stream};"
                    f"{base_stream}{effect_stream}overlay=0:0:enable='between(t,{start_time},{end_time})'{video_streams}"
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
            # Map all audio streams from the first input (0), if they exist (?)
            '-map', '0:a?', 
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
        except subprocess.TimeoutExpired:
            self.log_func("❌ ERROR: ffmpeg command timed out.")
        except subprocess.CalledProcessError as e:
            self.log_func("❌ ERROR during video editing.")
            self.log_func(f"FFmpeg stderr:\n{e.stderr}")
            import shutil
            shutil.copy(input_video, output_video)