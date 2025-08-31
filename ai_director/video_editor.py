# ai_director/video_editor.py
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
        if not timeline:
            self.log_func("No edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return

        applicable_events = [e for e in timeline if e.action in ["zoom_to_cam", "zoom_to_game", "zoom_out", "zoom_to_cam_reaction"]]

        if not applicable_events:
            self.log_func("No applicable edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return
            
        self.log_func(f"Applying {len(applicable_events)} edits to the video...")

        filter_parts = []
        
        # Stage 1: Create a single blurred background for all zoom-out events
        filter_parts.append(f"[0:v]boxblur=50[blur_bg]")
        
        # Stage 2: Create each 'effect' clip from the original video stream
        for i, event in enumerate(applicable_events):
            effect_stream = f"[eff_{i}]"
            
            effect_filter = ""
            if event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
                effect_filter = f"[0:v]crop=iw/2:ih/2:0:0,scale=1080:1920{effect_stream}"
            elif event.action == "zoom_to_game":
                effect_filter = f"[0:v]crop=iw/1.5:ih/1.5:iw/6:ih/6,scale=1080:1920{effect_stream}"
            elif event.action == "zoom_out":
                # This stream is the scaled-down foreground video
                effect_filter = f"[0:v]scale=iw*0.8:ih*0.8{effect_stream}"
            
            filter_parts.append(effect_filter)

        # Stage 3: Chain the overlays sequentially, starting with the original video stream
        last_stream = "[0:v]"
        for i, event in enumerate(applicable_events):
            start_time = event.timestamp
            end_time = start_time + event.duration
            effect_stream = f"[eff_{i}]"
            next_stream = f"[v_out_{i}]"
            
            if event.action == "zoom_out":
                # For zoom out, we need a two-step overlay: first the blur, then the centered foreground
                temp_stream = f"[temp_{i}]"
                filter_parts.append(f"{last_stream}[blur_bg]overlay=0:0:enable='between(t,{start_time},{end_time})'{temp_stream}")
                filter_parts.append(f"{temp_stream}{effect_stream}overlay=(W-w)/2:(H-h)/2:enable='between(t,{start_time},{end_time})'{next_stream}")
            else:
                # For zoom in, it's a simple one-step overlay
                filter_parts.append(f"{last_stream}{effect_stream}overlay=0:0:enable='between(t,{start_time},{end_time})'{next_stream}")
            
            last_stream = next_stream
            
        full_filtergraph = ";".join(filter_parts)

        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-filter_complex', full_filtergraph,
            '-map', last_stream,
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