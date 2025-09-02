# ai_director/video_editor.py - SIMPLIFIED WORKING VERSION
import subprocess
import os
from typing import List
from ai_director.data_models import TimelineEvent

class VideoEditor:
    """Executes the editing timeline with working animated zoom effects."""

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

        # Check if we have zoom_out events - use different approach for them
        has_zoom_out = any(event.action == "zoom_out" for event in applicable_events)
        
        if has_zoom_out:
            self._apply_edits_with_smooth_zoom(input_video, output_video, applicable_events)
        else:
            self._apply_standard_edits(input_video, output_video, applicable_events)

    def _apply_standard_edits(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Apply standard crop/zoom effects without animation."""
        filter_parts = []
        
        # Stage 1: Create a single blurred background for all zoom-out events
        filter_parts.append(f"[0:v]boxblur=50[blur_bg]")
        
        # Stage 2: Create each 'effect' clip from the original video stream
        for i, event in enumerate(events):
            effect_stream = f"[eff_{i}]"
            
            if event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
                effect_filter = f"[0:v]crop=iw/2:ih/2:0:0,scale=1080:1920{effect_stream}"
            elif event.action == "zoom_to_game":
                effect_filter = f"[0:v]crop=iw/1.5:ih/1.5:iw/6:ih/6,scale=1080:1920{effect_stream}"
            
            filter_parts.append(effect_filter)

        # Stage 3: Chain the overlays sequentially
        last_stream = "[0:v]"
        for i, event in enumerate(events):
            start_time = event.timestamp
            end_time = start_time + event.duration
            effect_stream = f"[eff_{i}]"
            next_stream = f"[v_out_{i}]"
            
            filter_parts.append(f"{last_stream}{effect_stream}overlay=0:0:enable='between(t,{start_time},{end_time})'{next_stream}")
            last_stream = next_stream
            
        full_filtergraph = ";".join(filter_parts)
        
        self._run_ffmpeg(input_video, output_video, full_filtergraph, last_stream)

    def _apply_edits_with_smooth_zoom(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Apply edits with smooth zoom out animation using simpler approach."""
        
        # For now, let's use a simpler approach that definitely works
        # We'll create the zoom out effect using multiple overlay steps
        
        filter_parts = []
        filter_parts.append(f"[0:v]boxblur=50[blur_bg]")
        
        last_stream = "[0:v]"
        
        for i, event in enumerate(events):
            next_stream = f"[v_out_{i}]"
            
            if event.action == "zoom_out":
                # Create smooth zoom out effect
                zoom_filter = self._create_working_zoom_out(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
                
            elif event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
                # Standard camera zoom
                crop_filter = (
                    f"[0:v]crop=iw/2:ih/2:0:0,scale=1080:1920[cropped_{i}];"
                    f"{last_stream}[cropped_{i}]overlay=0:0:enable="
                    f"'between(t,{event.timestamp},{event.timestamp + event.duration})'"
                    f"{next_stream}"
                )
                filter_parts.append(crop_filter)
                
            elif event.action == "zoom_to_game":
                # Standard game zoom
                crop_filter = (
                    f"[0:v]crop=iw/1.5:ih/1.5:iw/6:ih/6,scale=1080:1920[cropped_{i}];"
                    f"{last_stream}[cropped_{i}]overlay=0:0:enable="
                    f"'between(t,{event.timestamp},{event.timestamp + event.duration})'"
                    f"{next_stream}"
                )
                filter_parts.append(crop_filter)
            
            last_stream = next_stream
        
        full_filtergraph = ";".join(filter_parts)
        self._run_ffmpeg(input_video, output_video, full_filtergraph, last_stream)

    def _create_working_zoom_out(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create smooth animated zoom out using reliable multi-step approach."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Always use the multi-step approach for reliability
        return self._create_multi_step_zoom_out(input_stream, output_stream, event)

    def _create_multi_step_zoom_out(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Multi-step zoom animation with very smooth animation."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Use many more steps for ultra-smooth animation
        # About 24-30 steps per second for cinema-quality smoothness
        steps_per_second = 24
        num_steps = max(int(duration * steps_per_second), 15)  # Minimum 15 steps
        num_steps = min(num_steps, 90)  # Maximum 90 steps (should still be manageable)
        
        step_duration = duration / num_steps
        
        start_zoom = 0.8
        zoom_out_rate = 0.02 * duration  # 2% per second instead of 5%
        end_zoom = max(start_zoom - zoom_out_rate, 0.4)
        
        self.log_func(f"Creating ultra-smooth zoom out: {num_steps} steps over {duration:.1f}s ({start_zoom:.2f} → {end_zoom:.2f})")
        
        effects = []
        
        # Apply blur background
        effects.append(
            f"{input_stream}[blur_bg]overlay=0:0:enable="
            f"'between(t,{start_time},{end_time})'[bg_applied]"
        )
        
        # Create zoom levels with very smooth progression
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            scale = start_zoom + (end_zoom - start_zoom) * progress
            effects.append(f"[0:v]scale=iw*{scale:.5f}:ih*{scale:.5f}[zoom_{step}]")
        
        # Apply each zoom level sequentially
        last_stream = "[bg_applied]"
        for step in range(num_steps):
            step_start = start_time + (step * step_duration)
            step_end = start_time + ((step + 1) * step_duration) if step < num_steps - 1 else end_time
            next_stream = f"[step_{step}]" if step < num_steps - 1 else output_stream
            
            effects.append(
                f"{last_stream}[zoom_{step}]overlay=(W-w)/2:(H-h)/2:enable="
                f"'between(t,{step_start:.5f},{step_end:.5f})'{next_stream}"
            )
            
            last_stream = next_stream
        
        return ";".join(effects)

    def _run_ffmpeg(self, input_video: str, output_video: str, filtergraph: str, final_stream: str):
        """Run the ffmpeg command with error handling."""
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-filter_complex', filtergraph,
            '-map', final_stream,
            '-map', '0:a?', 
            '-c:a', 'copy',
            output_video
        ]

        try:
            self.log_func("Executing ffmpeg command...")
            self.log_func(f"Filter: {filtergraph}")
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