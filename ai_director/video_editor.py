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

        if event.action == "zoom_out":
            filter_str = self._create_param_zoom_out_filter(dur)
        elif event.action in ("zoom_to_cam", "zoom_to_cam_reaction"):
            filter_str = self._create_param_cam_zoom_filter(dur)
        elif event.action == "zoom_to_game":
            filter_str = self._create_param_game_zoom_filter(dur)
        else:
            self._extract_plain_segment(input_video, output_file, segment)
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,                  # accurate seek
            "-ss", f"{start:.6f}",
            "-t",  f"{dur:.6f}",
            "-filter_complex", filter_str,
            "-map", "[output]",
            "-map", "0:a?", "-c:a", "copy",     # keep your audio track(s) as-is
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-reset_timestamps", "1",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_file,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            self.log_func(f"✅ Effect {event.action} @ {start:.2f}s for {dur:.2f}s")
        except subprocess.TimeoutExpired:
            self.log_func("❌ Timeout on effect segment; extracting plain segment instead.")
            self._extract_plain_segment(input_video, output_file, segment)
        except subprocess.CalledProcessError as e:
            self.log_func("❌ FFmpeg error on effect segment; falling back to plain.")
            self.log_func(e.stderr[-4000:])
            self._extract_plain_segment(input_video, output_file, segment)


    # ========== PARAMETRIC FILTER BUILDERS (FAST, SMOOTH) ==========
    def _create_param_zoom_out_filter(self, duration: float) -> str:
        """Smooth zoom-out: foreground scaled by z(t) over a blurred background."""
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        start_zoom = 0.80
        end_zoom = max(0.40, start_zoom - 0.02 * duration)  # ~2% per second
        z = f"({start_zoom:.5f}+({end_zoom - start_zoom:.5f})*t/{max(duration,1e-6):.6f})"

        return (
            # Split once to create fg/bg paths
            "[0:v]scale=1080:1920,split[fg_in][bg_in];"
            # Background blur
            "[bg_in]boxblur=50[bg];"
            # Foreground scaled continuously each frame (scale supports eval=frame)
            f"[fg_in]scale=w='1080*{z}':h='1920*{z}':eval=frame[fgz];"
            # Center overlay; no eval needed here
            "[bg][fgz]overlay=x='(W-w)/2':y='(H-h)/2',"
            "fps=60,format=yuv420p[output]"
        )

    def _create_param_cam_zoom_filter(self, duration: float) -> str:
        """Smooth camera zoom by varying crop size over time (top-left ROI, stronger/eased)."""
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        zoom_in = True  # reactions feel better zooming in; set to random.choice([True, False]) if you want variety
        # cosine ease 0..1
        p = f"min(max(t/{max(duration,1e-6):.6f},0),1)"
        ease = f"(1-cos(3.14159265*{p}))/2"

        # stronger motion: from 1.00 -> 0.70 (30% tighter crop) or reverse
        s_in  = f"(1.00 - 0.30*{ease})"
        s_out = f"(0.70 + 0.30*{ease})"
        s = s_in if zoom_in else s_out

        # integer-even crop dims to keep YUV happy
        w = f"floor((in_w/2)*{s}/2)*2"
        h = f"floor((in_h/2)*{s}/2)*2"
        x = "0"
        y = "0"

        return (
            "[0:v]setpts=PTS-STARTPTS,"  # t starts at 0
            f"crop=w='{w}':h='{h}':x={x}:y={y}:exact=1,"
            "scale=1080:1920:flags=bicubic,fps=60,format=yuv420p[output]"
        )


    def _create_param_game_zoom_filter(self, duration: float) -> str:
        """Smooth game-area zoom around centered base region (stronger/eased)."""
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        zoom_in = True  # zoom in reads best; flip to taste
        p = f"min(max(t/{max(duration,1e-6):.6f},0),1)"
        ease = f"(1-cos(3.14159265*{p}))/2"

        # base region (center) and stronger range (to 0.75 of base)
        base_w, base_h = "(in_w/1.5)", "(in_h/1.5)"
        base_x, base_y = "(in_w/6)", "(in_h/6)"

        s_in  = f"(1.00 - 0.25*{ease})"   # 25% tighter at end
        s_out = f"(0.75 + 0.25*{ease})"
        s = s_in if zoom_in else s_out

        w = f"floor(({base_w})*{s}/2)*2"
        h = f"floor(({base_h})*{s}/2)*2"

        # center inside base region; clamp and floor so crop stays valid
        x_centered = f"({base_x}) + (({base_w}) - ({w}))/2"
        y_centered = f"({base_y}) + (({base_h}) - ({h}))/2"
        x = f"floor(max(min({x_centered}, in_w-({w})), 0))"
        y = f"floor(max(min({y_centered}, in_h-({h})), 0))"

        return (
            "[0:v]setpts=PTS-STARTPTS,"
            f"crop=w='{w}':h='{h}':x='{x}':y='{y}':exact=1,"
            "scale=1080:1920:flags=bicubic,fps=60,format=yuv420p[output]"
        )


    def _create_param_zoom_out_filter(self, duration: float) -> str:
        """Smooth zoom-out over blurred BG (kept, with mild clamp)."""
        if duration <= 0:
            return "[0:v]scale=1080:1920,format=yuv420p[output]"

        # cosine ease for nicer feel
        p = f"min(max(t/{max(duration,1e-6):.6f},0),1)"
        ease = f"(1-cos(3.14159265*{p}))/2"

        start_zoom = 0.80
        end_zoom   = max(0.60, start_zoom - 0.02 * duration)  # keep >= 0.60
        z = f"({start_zoom:.5f} + ({end_zoom - start_zoom:.5f})*{ease})"

        return (
            "[0:v]setpts=PTS-STARTPTS,scale=1080:1920,split[fg_in][bg_in];"
            "[bg_in]boxblur=50[bg];"
            f"[fg_in]scale=w='1080*{z}':h='1920*{z}':eval=frame[fgz];"
            "[bg][fgz]overlay=x='(W-w)/2':y='(H-h)/2',"
            "fps=60,format=yuv420p[output]"
        )


    # ========== PLAIN SEGMENT, CONCAT, UTILITIES ==========
    def _extract_plain_segment(self, input_video: str, output_file: str, segment: dict):
        """Extract a non-effect span with robust re-encode (matches effect segments)."""
        start = segment["start"]
        dur   = segment["end"] - segment["start"]
        if dur <= 0:
            with open(output_file, "wb") as f:
                pass
            return

        filter_str = "scale=1080:1920:flags=bicubic,fps=60,format=yuv420p[vout]"
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,                  # accurate seek
            "-ss", f"{start:.6f}",
            "-t",  f"{dur:.6f}",
            "-filter_complex", filter_str,
            "-map", "[vout]",
            "-map", "0:a?", "-c:a", "copy",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-reset_timestamps", "1",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output_file,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        except subprocess.CalledProcessError as e:
            self.log_func(f"❌ Plain segment re-encode failed, stderr (tail):\n{e.stderr[-2000:]}")
            self._apply_simple_fallback_edits(input_video, output_file)
        except subprocess.TimeoutExpired:
            self.log_func("❌ Plain segment re-encode timed out; falling back to copy.")
            self._apply_simple_fallback_edits(input_video, output_file)



    def _concatenate_segments(self, segment_files: List[str], output_video: str):
        """Concat segments and re-encode final to lock A/V sync and timestamps."""
        import tempfile
        concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for f in segment_files:
                concat_list.write(f"file '{f}'\n")
            concat_list.close()

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list.name,
                "-fflags", "+genpts",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",                # unify audio timing
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_video,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900)
            self.log_func("✅ Successfully concatenated all segments (re-encoded)")
        except subprocess.CalledProcessError as e:
            self.log_func("❌ Concat failed.")
            self.log_func(e.stderr[:4000])
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
