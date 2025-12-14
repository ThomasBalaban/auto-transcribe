# ai_director/video_editor.py
import subprocess
import os
import random
from typing import List
from ai_director.data_models import TimelineEvent

class VideoEditor:
    """Executes the editing timeline with smooth, high-FPS parametric zoom effects."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.log_func("📹 AI Director Video Editor initialized (parametric zooms).")

    # ========== PUBLIC ENTRY ==========
    def apply_edits(self, input_video: str, output_video: str, timeline: List[TimelineEvent]):
        if not timeline:
            self.log_func("No edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return

        applicable = [e for e in timeline if e.action in ("zoom_to_cam", "zoom_to_game", "zoom_out", "zoom_to_cam_reaction")]
        if not applicable:
            self.log_func("No applicable edits in timeline. Copying video without edits.")
            import shutil
            shutil.copy(input_video, output_video)
            return

        self.log_func(f"Applying {len(applicable)} edits with segment-based, per-frame animations...")
        self._apply_edits_segment_based(input_video, output_video, applicable)

    # ========== CORE: SEGMENT-BASED PIPELINE ==========
    def _apply_edits_segment_based(self, input_video: str, output_video: str, events: List[TimelineEvent]):
        """Process non-effect gaps as copy, effect spans as re-encoded parametric zooms; then concat."""
        import tempfile

        self.log_func("Using segment-based processing for performance and stability...")

        total_duration = self._get_video_duration(input_video)
        segments = self._create_segments(events, total_duration)

        temp_dir = tempfile.mkdtemp()
        segment_files: List[str] = []

        try:
            for i, seg in enumerate(segments):
                out_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                if seg["has_effect"]:
                    self._process_segment_with_effect(input_video, out_path, seg)
                else:
                    self._extract_plain_segment(input_video, out_path, seg)
                segment_files.append(out_path)

            self._concatenate_segments(segment_files, output_video)
        finally:
            # cleanup
            for f in segment_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

    def _create_segments(self, events: List[TimelineEvent], total_duration: float) -> List[dict]:
        """Break the timeline into plain and effect segments."""
        segments = []
        t = 0.0
        for e in sorted(events, key=lambda x: x.timestamp):
            if t < e.timestamp:
                segments.append({"start": t, "end": e.timestamp, "has_effect": False, "event": None})
            segments.append({"start": e.timestamp, "end": e.timestamp + e.duration, "has_effect": True, "event": e})
            t = e.timestamp + e.duration
        if t < total_duration:
            segments.append({"start": t, "end": total_duration, "has_effect": False, "event": None})
        return segments

    def _process_segment_with_effect(self, input_video: str, output_file: str, segment: dict):
        """Process a single segment with its effect (accurate seek + reset pts)."""
        event = segment['event']
        start = segment['start']
        dur   = segment['end'] - segment['start']

        if dur <= 0.0:
            with open(output_file, "wb") as f:
                pass
            return

        # Select the filter based on action
        if event.action == "zoom_out":
            filter_str = self._create_param_zoom_out_filter(dur)
        elif event.action in ("zoom_to_cam", "zoom_to_cam_reaction"):
            filter_str = self._create_param_cam_zoom_filter(dur)
        elif event.action == "zoom_to_game":
            filter_str = self._create_param_game_zoom_filter(dur)
        else:
            self._extract_plain_segment(input_video, output_file, segment)
            return

        # Uses CRF 18 (Standard)
        # Note: We use -r 60 to enforce a constant 60fps output
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-ss", f"{start:.6f}",              # <--- MOVED: Input Seeking (Resets timestamps to 0)
            "-t",  f"{dur:.6f}",
            "-i", input_video,                  
            "-filter_complex", filter_str,
            "-map", "[output]",
            "-map", "0:a?", "-c:a", "copy",
            "-c:v", "libx264", 
            "-preset", "veryfast",
            "-crf", "18", 
            "-r", "60",                         # Enforce 60fps
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_file,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            self.log_func(f"✅ Effect {event.action} @ {start:.2f}s for {dur:.2f}s")
        except subprocess.TimeoutExpired:
            self.log_func("❌ Timeout on effect segment; extracting plain segment instead.")
            self._extract_plain_segment(input_video, output_file, segment)
        except subprocess.CalledProcessError as e:
            self.log_func("❌ FFmpeg error on effect segment; falling back to plain.")
            if e.stderr:
                self.log_func(f"FFmpeg Error: {e.stderr[-500:]}")
            self._extract_plain_segment(input_video, output_file, segment)


    # ========== ANIMATION FILTERS (RECURSIVE ZOOMPAN + SCALE OVERLAY) ==========

    def _create_param_cam_zoom_filter(self, duration: float) -> str:
        """
        Recursive logic: If frame 0, force 1.5. Else, PreviousZoom + step.
        Guarantees start at 1.5 regardless of timestamps.
        
        UPDATED: Slowed down zoom rate by 50% (step 0.075 instead of 0.15)
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        total_frames = max(1, duration * 60)
        
        # 1.50 -> 1.575 (Difference 0.075 - 50% of original)
        step = 0.075 / total_frames
        
        z_expr = f"min(1.575,if(eq(on,0),1.5,pzoom+{step:.6f}))"

        return (
            "[0:v]setpts=PTS-STARTPTS,"
            f"zoompan=z='{z_expr}':x=0:y=0:d=1:s=1080x1920:fps=60,"
            "format=yuv420p[output]"
        )

    def _create_param_game_zoom_filter(self, duration: float) -> str:
        """
        Recursive logic: If frame 0, force 1.33. Else, PreviousZoom + step.
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        total_frames = max(1, duration * 60)
        
        # 1.33 -> 1.46 (Difference 0.13)
        step = 0.13 / total_frames
        
        z_expr = f"min(1.46,if(eq(on,0),1.33,pzoom+{step:.6f}))"
        
        # Center coordinates
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"

        return (
            "[0:v]setpts=PTS-STARTPTS,"
            f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1:s=1080x1920:fps=60,"
            "format=yuv420p[output]"
        )

    def _create_param_zoom_out_filter(self, duration: float) -> str:
        """
        Classic Blur + Scale Out Effect.
        Background: Blurred and darkened copy of video.
        Foreground: Scales from 0.8 (80%) down to 0.75 (75%).
        Uses 'n' (frame count) for stability.
        
        UPDATED: Slowed down zoom out rate by 50% (0.05 change instead of 0.1)
        """
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        total_frames = max(1, duration * 60)
        
        # Start Scale: 0.8 (Starts "shrunken")
        # End Scale: 0.75 (Very slow pull back)
        # Total Change: 0.05 (was 0.1)
        
        # Expression: 0.8 - (0.05 * n / total_frames)
        scale_expr = f"(0.8-(0.05*n/{int(total_frames)}))"
        
        # We wrap dimensions in 2*trunc(.../2) to ensure EVEN numbers for YUV420p
        w_expr = f"2*trunc(iw*{scale_expr}/2)"
        h_expr = f"2*trunc(ih*{scale_expr}/2)"

        return (
            # 1. Split input into Foreground (fg) and Background (bg)
            "[0:v]setpts=PTS-STARTPTS,scale=1080:1920,split[fg_in][bg_in];"
            
            # 2. Prepare Background: Blur it heavily + Darken slightly
            "[bg_in]boxblur=40:5,eq=brightness=-0.1[bg];"
            
            # 3. Scale Foreground: Use 'eval=frame' to update size every frame based on 'n'
            f"[fg_in]scale=w='{w_expr}':h='{h_expr}':eval=frame[fg_scaled];"
            
            # 4. Overlay: Center the scaled foreground on the blurred background
            "[bg][fg_scaled]overlay=x='(W-w)/2':y='(H-h)/2',"
            "fps=60,format=yuv420p[output]"
        )

    # ========== PLAIN SEGMENT, CONCAT, UTILITIES ==========
    def _extract_plain_segment(self, input_video: str, output_file: str, segment: dict):
        """Extract a non-effect span with robust re-encode."""
        start = segment["start"]
        dur   = segment["end"] - segment["start"]
        if dur <= 0:
            with open(output_file, "wb") as f:
                pass
            return

        filter_str = "scale=1080:1920:flags=bilinear,fps=60,format=yuv420p[vout]"
        
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-ss", f"{start:.6f}",              # <--- MOVED: Input Seeking
            "-t",  f"{dur:.6f}",
            "-i", input_video,
            "-filter_complex", filter_str,
            "-map", "[vout]",
            "-map", "0:a?", "-c:a", "copy",
            "-c:v", "libx264", 
            "-preset", "veryfast",
            "-crf", "18",
            "-r", "60",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_file,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            self.log_func(f"❌ Plain segment re-encode failed, stderr (tail):\n{e.stderr[-500:] if e.stderr else 'No stderr'}")
            self._apply_simple_fallback_edits(input_video, output_file)
        except subprocess.TimeoutExpired:
            self.log_func("❌ Plain segment re-encode timed out; falling back to copy.")
            self._apply_simple_fallback_edits(input_video, output_file)

    def _concatenate_segments(self, segment_files: List[str], output_video: str):
        """Concat segments and re-encode final."""
        import tempfile
        concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for f in segment_files:
                concat_list.write(f"file '{f}'\n")
            concat_list.close()

            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_list.name,
                "-fflags", "+genpts",
                "-c:v", "libx264", 
                "-preset", "veryfast",
                "-crf", "18", 
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_video,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900)
            self.log_func("✅ Successfully concatenated all segments (re-encoded)")
        except subprocess.CalledProcessError as e:
            self.log_func("❌ Concat failed.")
            if e.stderr:
                self.log_func(f"FFmpeg Error: {e.stderr[:500]}")
            self._apply_simple_fallback_edits(segment_files[0], output_video)
        finally:
            try:
                os.unlink(concat_list.name)
            except:
                pass


    def _get_video_duration(self, video_path: str) -> float:
        """Get duration via ffprobe; return 60s fallback if unknown."""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            return float(res.stdout.strip())
        except Exception:
            return 60.0

    def _apply_simple_fallback_edits(self, input_video: str, output_video: str):
        """Ultimate fallback: copy the original to output."""
        try:
            import shutil
            shutil.copy2(input_video, output_video)
            self.log_func("📋 Fallback: Copied original video without edits.")
        except Exception as e:
            self.log_func(f"❌ Even fallback copy failed: {e}")
            with open(output_video, "w") as f:
                f.write("# Video processing failed")