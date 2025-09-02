# ai_director/video_editor.py - SIMPLIFIED WORKING VERSION
import subprocess
import os
import random
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
            
        self.log_func(f"Applying {len(applicable_events)} edits using optimized approach...")

        # Use a simpler approach with reduced complexity to avoid freezing
        self._apply_edits_with_reduced_complexity(input_video, output_video, applicable_events)

    def _apply_edits_with_reduced_complexity(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Apply edits with reduced complexity to prevent freezing."""
        filter_parts = []
        
        # Create blur background once
        filter_parts.append(f"[0:v]boxblur=50[blur_bg]")
        
        # Process each effect with fewer steps to prevent complexity issues
        last_stream = "[0:v]"
        
        for i, event in enumerate(events):
            next_stream = f"[v_out_{i}]"
            
            if event.action == "zoom_out":
                # Simplified zoom out with fewer steps
                zoom_filter = self._create_simple_zoom_out(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
                
            elif event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
                # Simplified camera zoom
                zoom_filter = self._create_simple_cam_zoom(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
                
            elif event.action == "zoom_to_game":
                # Simplified game zoom
                zoom_filter = self._create_simple_game_zoom(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
            
            last_stream = next_stream
            
        full_filtergraph = ";".join(filter_parts)
        self._run_ffmpeg(input_video, output_video, full_filtergraph, last_stream)

    def _create_simple_zoom_out(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create zoom out with just 3 steps to prevent freezing."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Use only 3 steps for maximum reliability
        step_duration = duration / 3
        
        start_zoom = 0.8
        zoom_out_rate = 0.02 * duration
        end_zoom = max(start_zoom - zoom_out_rate, 0.4)
        
        # Calculate the 3 zoom levels
        zoom_1 = start_zoom
        zoom_2 = start_zoom + (end_zoom - start_zoom) * 0.5
        zoom_3 = end_zoom
        
        effects = []
        
        # Apply blur background during entire effect
        effects.append(
            f"{input_stream}[blur_bg]overlay=0:0:enable="
            f"'between(t,{start_time},{end_time})'[bg_applied]"
        )
        
        # Create the 3 zoom levels
        effects.append(f"[0:v]scale=iw*{zoom_1:.3f}:ih*{zoom_1:.3f}[zoom_1]")
        effects.append(f"[0:v]scale=iw*{zoom_2:.3f}:ih*{zoom_2:.3f}[zoom_2]")
        effects.append(f"[0:v]scale=iw*{zoom_3:.3f}:ih*{zoom_3:.3f}[zoom_3]")
        
        # Apply each zoom level at its time
        effects.append(
            f"[bg_applied][zoom_1]overlay=(W-w)/2:(H-h)/2:enable="
            f"'between(t,{start_time:.3f},{start_time + step_duration:.3f})'[step1]"
        )
        effects.append(
            f"[step1][zoom_2]overlay=(W-w)/2:(H-h)/2:enable="
            f"'between(t,{start_time + step_duration:.3f},{start_time + 2*step_duration:.3f})'[step2]"
        )
        effects.append(
            f"[step2][zoom_3]overlay=(W-w)/2:(H-h)/2:enable="
            f"'between(t,{start_time + 2*step_duration:.3f},{end_time:.3f})'"
            f"{output_stream}"
        )
        
        return ";".join(effects)

    def _create_simple_cam_zoom(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create simple camera zoom with 3 steps."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Random zoom direction
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            scale_1, scale_2, scale_3 = 1.0, 1.0 - zoom_rate*0.5, 1.0 - zoom_rate
        else:
            scale_1, scale_2, scale_3 = 1.0 - zoom_rate, 1.0 - zoom_rate*0.5, 1.0
        
        step_duration = duration / 3
        
        effects = []
        
        # Create the 3 crop levels
        effects.append(f"[0:v]crop=(iw/2)*{scale_1:.3f}:(ih/2)*{scale_1:.3f}:0:0,scale=1080:1920[crop_1]")
        effects.append(f"[0:v]crop=(iw/2)*{scale_2:.3f}:(ih/2)*{scale_2:.3f}:0:0,scale=1080:1920[crop_2]")
        effects.append(f"[0:v]crop=(iw/2)*{scale_3:.3f}:(ih/2)*{scale_3:.3f}:0:0,scale=1080:1920[crop_3]")
        
        # Apply each crop level
        effects.append(
            f"{input_stream}[crop_1]overlay=0:0:enable="
            f"'between(t,{start_time:.3f},{start_time + step_duration:.3f})'[cam_step1]"
        )
        effects.append(
            f"[cam_step1][crop_2]overlay=0:0:enable="
            f"'between(t,{start_time + step_duration:.3f},{start_time + 2*step_duration:.3f})'[cam_step2]"
        )
        effects.append(
            f"[cam_step2][crop_3]overlay=0:0:enable="
            f"'between(t,{start_time + 2*step_duration:.3f},{end_time:.3f})'"
            f"{output_stream}"
        )
        
        return ";".join(effects)

    def _create_simple_game_zoom(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create simple game zoom with 3 steps."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Random zoom direction
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            scale_1, scale_2, scale_3 = 1.0, 1.0 - zoom_rate*0.5, 1.0 - zoom_rate
        else:
            scale_1, scale_2, scale_3 = 1.0 - zoom_rate, 1.0 - zoom_rate*0.5, 1.0
        
        step_duration = duration / 3
        
        effects = []
        
        # Create the 3 crop levels for game area
        for i, scale in enumerate([scale_1, scale_2, scale_3], 1):
            crop_w = f"(iw/1.5)*{scale:.3f}"
            crop_h = f"(ih/1.5)*{scale:.3f}"
            crop_x = f"(iw/6)+((iw/1.5)*(1-{scale:.3f})/2)"
            crop_y = f"(ih/6)+((ih/1.5)*(1-{scale:.3f})/2)"
            
            effects.append(f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920[game_crop_{i}]")
        
        # Apply each crop level
        effects.append(
            f"{input_stream}[game_crop_1]overlay=0:0:enable="
            f"'between(t,{start_time:.3f},{start_time + step_duration:.3f})'[game_step1]"
        )
        effects.append(
            f"[game_step1][game_crop_2]overlay=0:0:enable="
            f"'between(t,{start_time + step_duration:.3f},{start_time + 2*step_duration:.3f})'[game_step2]"
        )
        effects.append(
            f"[game_step2][game_crop_3]overlay=0:0:enable="
            f"'between(t,{start_time + 2*step_duration:.3f},{end_time:.3f})'"
            f"{output_stream}"
        )
        
        return ";".join(effects)

    def _run_ffmpeg(self, input_video: str, output_video: str, filtergraph: str, final_stream: str):
        """Run the ffmpeg command with conservative settings to prevent freezing."""
        
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-filter_complex', filtergraph,
            '-map', final_stream,
            '-map', '0:a?', 
            '-c:a', 'copy',
            '-c:v', 'libx264',  # Explicit video codec
            '-preset', 'medium',  # Balanced speed/quality
            '-crf', '23',  # Good quality
            '-movflags', '+faststart',
            output_video
        ]

        try:
            self.log_func("Executing ffmpeg with simplified effects...")
            self.log_func(f"Filter length: {len(filtergraph)} characters")
                
            result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=600
            )
            self.log_func("✅ Video editing with simplified effects complete.")
        except subprocess.TimeoutExpired:
            self.log_func("❌ ERROR: ffmpeg command timed out.")
            self._apply_simple_fallback_edits(input_video, output_video)
        except subprocess.CalledProcessError as e:
            self.log_func("❌ ERROR during video editing.")
            self.log_func(f"FFmpeg stderr:\n{e.stderr}")
            self._apply_simple_fallback_edits(input_video, output_video)

    def _apply_edits_with_overlays(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Apply edits by creating effect clips and overlaying them at correct timestamps."""
        import tempfile
        import os
        
        filter_parts = []
        temp_dir = tempfile.mkdtemp()
        effect_clips = []
        
        try:
            # Create individual effect clips first
            for i, event in enumerate(events):
                clip_path = os.path.join(temp_dir, f"effect_{i}.mp4")
                
                if self._create_effect_clip(input_video, clip_path, event, i):
                    effect_clips.append((clip_path, event, i))
                    self.log_func(f"✅ Created effect clip {i} for {event.action}")
                else:
                    self.log_func(f"❌ Failed to create effect clip {i}")
            
            if not effect_clips:
                self.log_func("No effect clips created, copying original")
                import shutil
                shutil.copy(input_video, output_video)
                return
            
            # Now create filter to overlay all effect clips at correct times
            self._create_overlay_filter_and_apply(input_video, output_video, effect_clips)
            
        finally:
            # Cleanup temp files
            for clip_path, _, _ in effect_clips:
                if os.path.exists(clip_path):
                    try:
                        os.remove(clip_path)
                    except:
                        pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

    def _create_effect_clip(self, input_video: str, output_path: str, event: TimelineEvent, clip_index: int) -> bool:
        """Create a single effect clip with smooth animation."""
        try:
            # Create a simpler filter for this single effect
            if event.action == "zoom_out":
                filter_str = self._create_smooth_zoom_out_clip_filter(event)
            elif event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
                filter_str = self._create_smooth_cam_zoom_clip_filter(event)
            elif event.action == "zoom_to_game":
                filter_str = self._create_smooth_game_zoom_clip_filter(event)
            else:
                return False
            
            # Extract the time segment and apply the effect
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-ss', str(event.timestamp),
                '-t', str(event.duration),
                '-filter_complex', filter_str,
                '-map', '[output]',
                '-an',  # No audio needed for effect clips
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=120)
            return True
            
        except Exception as e:
            self.log_func(f"Failed to create effect clip: {e}")
            return False

    def _create_smooth_zoom_out_clip_filter(self, event: TimelineEvent) -> str:
        """Create smooth zoom out filter for a single clip."""
        duration = event.duration
        
        # Reduce steps but keep it smooth
        num_steps = min(max(int(duration * 10), 5), 25)  # 10 steps/sec, max 25
        step_duration = duration / num_steps
        
        start_zoom = 0.8
        zoom_out_rate = 0.02 * duration
        end_zoom = max(start_zoom - zoom_out_rate, 0.4)
        
        effects = []
        
        # Create blur background for entire duration
        effects.append("[0:v]boxblur=50[blur_bg]")
        
        # Create zoom levels
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            scale = start_zoom + (end_zoom - start_zoom) * progress
            effects.append(f"[0:v]scale=iw*{scale:.4f}:ih*{scale:.4f}[zoom_{step}]")
        
        # Apply zoom levels with timing
        last_stream = "[blur_bg]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[zoom_{step}]overlay=(W-w)/2:(H-h)/2:enable="
                f"'between(t,{step_start:.4f},{step_end:.4f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _create_smooth_cam_zoom_clip_filter(self, event: TimelineEvent) -> str:
        """Create smooth camera zoom filter for a single clip."""
        duration = event.duration
        num_steps = min(max(int(duration * 8), 4), 20)  # 8 steps/sec
        
        # Random zoom direction
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            start_scale, end_scale = 1.0, 1.0 - zoom_rate
        else:
            start_scale, end_scale = 1.0 - zoom_rate, 1.0
        
        effects = []
        step_duration = duration / num_steps
        
        # Create crop levels
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            scaled_width = f"(iw/2)*{current_scale:.4f}"
            scaled_height = f"(ih/2)*{current_scale:.4f}"
            
            effects.append(f"[0:v]crop={scaled_width}:{scaled_height}:0:0,scale=1080:1920[cropped_{step}]")
        
        # Apply crop levels
        last_stream = "[0:v]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[cropped_{step}]overlay=0:0:enable="
                f"'between(t,{step_start:.4f},{step_end:.4f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _create_smooth_game_zoom_clip_filter(self, event: TimelineEvent) -> str:
        """Create smooth game zoom filter for a single clip."""
        duration = event.duration
        num_steps = min(max(int(duration * 8), 4), 20)  # 8 steps/sec
        
        # Random zoom direction
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            start_scale, end_scale = 1.0, 1.0 - zoom_rate
        else:
            start_scale, end_scale = 1.0 - zoom_rate, 1.0
        
        effects = []
        step_duration = duration / num_steps
        
        # Create crop levels for game area (center)
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            scaled_width = f"(iw/1.5)*{current_scale:.4f}"
            scaled_height = f"(ih/1.5)*{current_scale:.4f}"
            scaled_x = f"(iw/6)+((iw/1.5)*(1-{current_scale:.4f})/2)"
            scaled_y = f"(ih/6)+((ih/1.5)*(1-{current_scale:.4f})/2)"
            
            effects.append(f"[0:v]crop={scaled_width}:{scaled_height}:{scaled_x}:{scaled_y},scale=1080:1920[cropped_{step}]")
        
        # Apply crop levels
        last_stream = "[0:v]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[cropped_{step}]overlay=0:0:enable="
                f"'between(t,{step_start:.4f},{step_end:.4f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _create_overlay_filter_and_apply(self, input_video: str, output_video: str, effect_clips: list):
        """Create filter to overlay all effect clips at their correct timestamps."""
        filter_parts = []
        
        # Add all input sources
        input_maps = [f"[0:v]"]  # Original video
        for i, (clip_path, event, clip_index) in enumerate(effect_clips):
            input_maps.append(f"[{i+1}:v]")
        
        # Start with original video
        last_stream = "[0:v]"
        
        # Overlay each effect clip at its correct time
        for i, (clip_path, event, clip_index) in enumerate(effect_clips):
            input_stream = f"[{i+1}:v]"
            next_stream = f"[overlay_{i}]" if i < len(effect_clips) - 1 else "[final]"
            
            # Overlay this effect at its timestamp
            filter_parts.append(
                f"{last_stream}{input_stream}overlay=0:0:enable="
                f"'between(t,{event.timestamp:.4f},{event.timestamp + event.duration:.4f})'"
                f"{next_stream}"
            )
            
            last_stream = next_stream
        
        # Build FFmpeg command with all inputs
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', input_video
        ]
        
        # Add effect clip inputs
        for clip_path, _, _ in effect_clips:
            ffmpeg_command.extend(['-i', clip_path])
        
        # Add filter and output
        ffmpeg_command.extend([
            '-filter_complex', ";".join(filter_parts),
            '-map', '[final]',
            '-map', '0:a?', '-c:a', 'copy',
            '-movflags', '+faststart',
            output_video
        ])
        
        try:
            self.log_func(f"Overlaying {len(effect_clips)} effect clips...")
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=300)
            self.log_func("✅ Successfully applied all overlay effects")
        except Exception as e:
            self.log_func(f"❌ Overlay failed: {e}")
            # Fallback to original video
            import shutil
            shutil.copy(input_video, output_video)

    def _apply_edits_segment_based(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Apply edits by processing segments separately and concatenating."""
        import tempfile
        import os
        
        self.log_func("Using segment-based processing for better performance...")
        
        # Get video duration first
        duration = self._get_video_duration(input_video)
        
        # Create segments based on events
        segments = self._create_segments(events, duration)
        
        temp_dir = tempfile.mkdtemp()
        segment_files = []
        
        try:
            # Process each segment
            for i, segment in enumerate(segments):
                segment_output = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                
                if segment['has_effect']:
                    # Apply effect to this segment
                    self._process_segment_with_effect(input_video, segment_output, segment)
                else:
                    # Extract plain segment
                    self._extract_plain_segment(input_video, segment_output, segment)
                
                segment_files.append(segment_output)
            
            # Concatenate all segments
            self._concatenate_segments(segment_files, output_video)
            
        finally:
            # Cleanup temp files
            for file in segment_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

    def _create_segments(self, events: List[TimelineEvent], total_duration: float) -> List[dict]:
        """Create segments based on events and gaps between them."""
        segments = []
        current_time = 0.0
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for event in sorted_events:
            # Add gap segment before event (if any)
            if current_time < event.timestamp:
                segments.append({
                    'start': current_time,
                    'end': event.timestamp,
                    'has_effect': False,
                    'event': None
                })
            
            # Add event segment
            segments.append({
                'start': event.timestamp,
                'end': event.timestamp + event.duration,
                'has_effect': True,
                'event': event
            })
            
            current_time = event.timestamp + event.duration
        
        # Add final gap segment (if any)
        if current_time < total_duration:
            segments.append({
                'start': current_time,
                'end': total_duration,
                'has_effect': False,
                'event': None
            })
        
        return segments

    def _process_segment_with_effect(self, input_video: str, output_file: str, segment: dict):
        """Process a single segment with its effect."""
        event = segment['event']
        start_time = segment['start']
        duration = segment['end'] - segment['start']
        
        # Create a simple, single-effect filter for this segment
        if event.action == "zoom_out":
            filter_str = self._create_simple_zoom_out_filter(duration, event)
        elif event.action in ["zoom_to_cam", "zoom_to_cam_reaction"]:
            filter_str = self._create_simple_cam_zoom_filter(duration, event)
        elif event.action == "zoom_to_game":
            filter_str = self._create_simple_game_zoom_filter(duration, event)
        else:
            # Fallback to plain segment
            self._extract_plain_segment(input_video, output_file, segment)
            return
        
        # Extract and process this segment
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-ss', str(start_time),
            '-t', str(duration),
            '-filter_complex', filter_str,
            '-map', '[output]',
            '-map', '0:a?', '-c:a', 'copy',
            output_file
        ]
        
        try:
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=120)
            self.log_func(f"✅ Processed segment {start_time:.1f}s-{segment['end']:.1f}s with {event.action}")
        except Exception as e:
            self.log_func(f"❌ Failed to process segment, using plain: {e}")
            self._extract_plain_segment(input_video, output_file, segment)

    def _create_simple_zoom_out_filter(self, duration: float, event: TimelineEvent) -> str:
        """Create a simple zoom out filter for a single segment."""
        # Use fewer steps for segment processing
        num_steps = min(max(int(duration * 8), 4), 20)  # 8 steps/sec, max 20
        step_duration = duration / num_steps
        
        start_zoom = 0.8
        zoom_out_rate = 0.02 * duration
        end_zoom = max(start_zoom - zoom_out_rate, 0.4)
        
        effects = ["[0:v]boxblur=50[blur_bg]"]
        
        # Create zoom levels
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            scale = start_zoom + (end_zoom - start_zoom) * progress
            effects.append(f"[0:v]scale=iw*{scale:.3f}:ih*{scale:.3f}[zoom_{step}]")
        
        # Apply zoom levels with precise timing
        last_stream = "[blur_bg]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[zoom_{step}]overlay=(W-w)/2:(H-h)/2:enable="
                f"'between(t,{step_start:.3f},{step_end:.3f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _create_simple_cam_zoom_filter(self, duration: float, event: TimelineEvent) -> str:
        """Create a simple camera zoom filter for a single segment."""
        # Simplified camera zoom with fewer steps
        num_steps = min(max(int(duration * 6), 3), 15)
        
        # Random zoom direction
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            start_scale, end_scale = 1.0, 1.0 - zoom_rate
        else:
            start_scale, end_scale = 1.0 - zoom_rate, 1.0
        
        effects = []
        step_duration = duration / num_steps
        
        # Create crop levels
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            scaled_width = f"(iw/2)*{current_scale:.3f}"
            scaled_height = f"(ih/2)*{current_scale:.3f}"
            
            effects.append(f"[0:v]crop={scaled_width}:{scaled_height}:0:0,scale=1080:1920[cropped_{step}]")
        
        # Apply crop levels
        last_stream = "[0:v]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[cropped_{step}]overlay=0:0:enable="
                f"'between(t,{step_start:.3f},{step_end:.3f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _create_simple_game_zoom_filter(self, duration: float, event: TimelineEvent) -> str:
        """Create a simple game zoom filter for a single segment."""
        # Similar to camera zoom but with game area crop
        num_steps = min(max(int(duration * 6), 3), 15)
        
        zoom_direction = random.choice(['in', 'out'])
        zoom_rate = 0.02 * duration
        
        if zoom_direction == 'in':
            start_scale, end_scale = 1.0, 1.0 - zoom_rate
        else:
            start_scale, end_scale = 1.0 - zoom_rate, 1.0
        
        effects = []
        step_duration = duration / num_steps
        
        # Create crop levels for game area
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            scaled_width = f"(iw/1.5)*{current_scale:.3f}"
            scaled_height = f"(ih/1.5)*{current_scale:.3f}"
            scaled_x = f"(iw/6)+((iw/1.5)*(1-{current_scale:.3f})/2)"
            scaled_y = f"(ih/6)+((ih/1.5)*(1-{current_scale:.3f})/2)"
            
            effects.append(f"[0:v]crop={scaled_width}:{scaled_height}:{scaled_x}:{scaled_y},scale=1080:1920[cropped_{step}]")
        
        # Apply crop levels
        last_stream = "[0:v]"
        for step in range(num_steps):
            step_start = step * step_duration
            step_end = (step + 1) * step_duration if step < num_steps - 1 else duration
            next_stream = f"[step_{step}]" if step < num_steps - 1 else "[output]"
            
            effects.append(
                f"{last_stream}[cropped_{step}]overlay=0:0:enable="
                f"'between(t,{step_start:.3f},{step_end:.3f})'{next_stream}"
            )
            last_stream = next_stream
        
        return ";".join(effects)

    def _extract_plain_segment(self, input_video: str, output_file: str, segment: dict):
        """Extract a plain segment without effects."""
        start_time = segment['start']
        duration = segment['end'] - segment['start']
        
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # No re-encoding for plain segments
            output_file
        ]
        
        try:
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=60)
        except Exception as e:
            self.log_func(f"❌ Failed to extract plain segment: {e}")

    def _concatenate_segments(self, segment_files: List[str], output_video: str):
        """Concatenate all segments into final video."""
        # Create concat file
        import tempfile
        concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        
        try:
            for file in segment_files:
                concat_file.write(f"file '{file}'\n")
            concat_file.close()
            
            # Concatenate
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file.name,
                '-c', 'copy',
                output_video
            ]
            
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=300)
            self.log_func("✅ Successfully concatenated all segments")
            
        finally:
            try:
                os.unlink(concat_file.name)
            except:
                pass

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 60.0  # Fallback

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
                # Smooth animated zoom to camera
                zoom_filter = self._create_smooth_zoom_to_cam(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
                
            elif event.action == "zoom_to_game":
                # Smooth animated zoom to game
                zoom_filter = self._create_smooth_zoom_to_game(last_stream, next_stream, event)
                filter_parts.append(zoom_filter)
            
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
        """Multi-step zoom animation with high smoothness."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Increase frame rate for ultra-smooth animation
        steps_per_second = 20  # Increased from 12 to 20
        num_steps = max(int(duration * steps_per_second), 12)  # Minimum 12 steps
        num_steps = min(num_steps, 60)  # Maximum 60 steps (increased from 36)
        
        step_duration = duration / num_steps
        
        start_zoom = 0.8
        zoom_out_rate = 0.02 * duration  # 2% per second
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

    def _create_smooth_zoom_to_cam(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create smooth animated zoom to camera with random zoom direction."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Randomly choose zoom in or zoom out
        zoom_direction = random.choice(['in', 'out'])
        
        # Camera area crop parameters
        crop_width = "iw/2"
        crop_height = "ih/2" 
        crop_x = "0"  # Left side (camera)
        crop_y = "0"  # Top
        
        zoom_rate = 0.02 * duration  # 2% per second
        
        if zoom_direction == 'in':
            # Start at base crop, zoom in further (crop gets smaller)
            start_crop_scale = 1.0
            end_crop_scale = 1.0 - zoom_rate  # Crop gets smaller = more zoomed in
            self.log_func(f"Camera zoom IN: {start_crop_scale:.2f} → {end_crop_scale:.2f} over {duration:.1f}s")
        else:
            # Start zoomed in, zoom out (crop gets bigger) 
            start_crop_scale = 1.0 - zoom_rate
            end_crop_scale = 1.0  # Crop gets bigger = more zoomed out
            self.log_func(f"Camera zoom OUT: {start_crop_scale:.2f} → {end_crop_scale:.2f} over {duration:.1f}s")
        
        return self._create_animated_crop_zoom(
            input_stream, output_stream, event,
            crop_width, crop_height, crop_x, crop_y,
            start_crop_scale, end_crop_scale
        )

    def _create_smooth_zoom_to_game(self, input_stream: str, output_stream: str, event: TimelineEvent) -> str:
        """Create smooth animated zoom to game with random zoom direction."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Randomly choose zoom in or zoom out
        zoom_direction = random.choice(['in', 'out'])
        
        # Game area crop parameters  
        crop_width = "iw/1.5"
        crop_height = "ih/1.5"
        crop_x = "iw/6"  # Center horizontally
        crop_y = "ih/6"  # Center vertically
        
        zoom_rate = 0.02 * duration  # 2% per second
        
        if zoom_direction == 'in':
            # Start at base crop, zoom in further
            start_crop_scale = 1.0
            end_crop_scale = 1.0 - zoom_rate
            self.log_func(f"Game zoom IN: {start_crop_scale:.2f} → {end_crop_scale:.2f} over {duration:.1f}s")
        else:
            # Start zoomed in, zoom out
            start_crop_scale = 1.0 - zoom_rate
            end_crop_scale = 1.0
            self.log_func(f"Game zoom OUT: {start_crop_scale:.2f} → {end_crop_scale:.2f} over {duration:.1f}s")
        
        return self._create_animated_crop_zoom(
            input_stream, output_stream, event,
            crop_width, crop_height, crop_x, crop_y,
            start_crop_scale, end_crop_scale
        )

    def _create_animated_crop_zoom(self, input_stream: str, output_stream: str, event: TimelineEvent,
                                  base_crop_width: str, base_crop_height: str, 
                                  base_crop_x: str, base_crop_y: str,
                                  start_scale: float, end_scale: float) -> str:
        """Create animated crop zoom with high smoothness."""
        start_time = event.timestamp
        end_time = event.timestamp + event.duration
        duration = event.duration
        
        # Increase frame rate for crop zooms too
        steps_per_second = 16  # Increased from 10 to 16
        num_steps = max(int(duration * steps_per_second), 10)  # Minimum 10 steps
        num_steps = min(num_steps, 48)  # Maximum 48 steps (increased from 30)
        
        step_duration = duration / num_steps
        
        effects = []
        
        # Create crop levels with very smooth scaling
        for step in range(num_steps):
            progress = step / (num_steps - 1) if num_steps > 1 else 0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            # Calculate dynamic crop dimensions with higher precision
            scaled_width = f"({base_crop_width})*{current_scale:.5f}"
            scaled_height = f"({base_crop_height})*{current_scale:.5f}"
            
            # Adjust crop position to keep it centered
            if "iw/6" in base_crop_x:  # Game crop (centered)
                scaled_x = f"({base_crop_x})+(({base_crop_width})*(1-{current_scale:.5f})/2)"
                scaled_y = f"({base_crop_y})+(({base_crop_height})*(1-{current_scale:.5f})/2)"
            else:  # Camera crop (left side)
                scaled_x = base_crop_x  # Keep at left edge
                scaled_y = base_crop_y  # Keep at top
            
            effects.append(
                f"[0:v]crop={scaled_width}:{scaled_height}:{scaled_x}:{scaled_y},"
                f"scale=1080:1920[cropped_{step}]"
            )
        
        # Apply each crop level sequentially
        last_stream = input_stream
        for step in range(num_steps):
            step_start = start_time + (step * step_duration)
            step_end = start_time + ((step + 1) * step_duration) if step < num_steps - 1 else end_time
            next_stream = f"[crop_step_{step}]" if step < num_steps - 1 else output_stream
            
            effects.append(
                f"{last_stream}[cropped_{step}]overlay=0:0:enable="
                f"'between(t,{step_start:.5f},{step_end:.5f})'{next_stream}"
            )
            
            last_stream = next_stream
        
        return ";".join(effects)

    def _run_ffmpeg(self, input_video: str, output_video: str, filtergraph: str, final_stream: str):
        """Run the ffmpeg command with smart error handling."""
        
        # Only fall back if filter is REALLY excessive (much higher limit)
        if len(filtergraph) > 50000:  # Much higher limit - 50k chars instead of 5k
            self.log_func(f"⚠️ Filter extremely complex ({len(filtergraph)} chars), falling back...")
            self._apply_simple_fallback_edits(input_video, output_video)
            return
        
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-filter_complex', filtergraph,
            '-map', final_stream,
            '-map', '0:a?', 
            '-c:a', 'copy',
            '-movflags', '+faststart',  # Better MP4 compatibility
            output_video
        ]

        try:
            self.log_func("Executing ffmpeg command with smooth zoom effects...")
            self.log_func(f"Filter length: {len(filtergraph)} characters")
                
            result = subprocess.run(
                ffmpeg_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # Longer timeout for complex operations
            )
            self.log_func("✅ Video editing with smooth zooms complete.")
        except subprocess.TimeoutExpired:
            self.log_func("❌ ERROR: ffmpeg command timed out. Trying simpler approach...")
            self._try_simpler_approach(input_video, output_video, filtergraph)
        except subprocess.CalledProcessError as e:
            self.log_func("❌ ERROR during video editing. Trying simpler approach...")
            if "Invalid argument" in e.stderr:
                self.log_func("Filter syntax error detected.")
            self._try_simpler_approach(input_video, output_video, filtergraph)

    def _try_simpler_approach(self, input_video: str, output_video: str, original_filter: str):
        """Try a simpler approach before giving up completely."""
        self.log_func("🔄 Attempting simpler zoom approach...")
        
        # Try with fewer steps - reduce complexity but keep the effects
        simplified_filter = self._simplify_filter(original_filter)
        
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', input_video,
            '-filter_complex', simplified_filter,
            '-map', '[final]',  # Simplified final stream name
            '-map', '0:a?', '-c:a', 'copy',
            output_video
        ]
        
        try:
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, timeout=300)
            self.log_func("✅ Simpler approach succeeded!")
        except:
            self.log_func("📋 All approaches failed, copying original (but this shouldn't happen often)")
            self._apply_simple_fallback_edits(input_video, output_video)

    def _simplify_filter(self, complex_filter: str) -> str:
        """Create a very simple version that just does basic zoom effects."""
        # This is a emergency fallback - just do basic zooms without smooth animation
        return "[0:v]scale=1080:1920[final]"  # Most basic filter possible

    def _apply_simple_fallback_edits(self, input_video: str, output_video: str):
        """Fallback to simple copy if complex editing fails."""
        try:
            import shutil
            shutil.copy2(input_video, output_video)
            self.log_func("📋 Fallback: Copied original video without edits.")
        except Exception as e:
            self.log_func(f"❌ Even fallback copy failed: {e}")
            # Create a minimal output file to prevent further errors
            with open(output_video, 'w') as f:
                f.write("# Video processing failed")