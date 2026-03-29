# clip_editor/intelligent_trimmer.py
"""
Intelligent Clip Trimming System.

Key changes vs original:
  • Two-pass Gemini strategy: FIND PUNCH POINT first, then trim around it.
  • "Minimum viable setup" framing to stop over-cutting context.
  • Snappier pacing guidance (hard jump cuts encouraged).
  • 1-second end-buffer kept to prevent cut-off reactions.
"""

from typing import List, Dict, Tuple, Optional
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from video_utils import get_video_duration
import subprocess
import os
import json
import re
import tempfile
import google.generativeai as genai  # type: ignore
import time


class IntelligentTrimmer:
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.vision_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
        self.max_clip_duration = 55.0
        self.min_clip_duration = 12.0

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_for_trim(
        self,
        video_path: str,
        title_details: Optional[Tuple[str, str, str]] = None,
        mic_transcriptions: Optional[List[str]] = None,
        desktop_transcriptions: Optional[List[str]] = None,
    ) -> List[Tuple[float, float]]:
        try:
            self.log_func("\n" + "=" * 60)
            self.log_func("🎬 INTELLIGENT CLIP TRIMMING - ANALYSIS PHASE")
            self.log_func("=" * 60)
            video_duration = get_video_duration(video_path, self.log_func)
            return self._call_gemini_for_trim_analysis(
                video_path, video_duration, title_details,
                mic_transcriptions, desktop_transcriptions,
            )
        except Exception as e:
            self.log_func(f"❌ Trim analysis failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []

    def apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]],
    ) -> bool:
        return self._apply_trim(input_video, output_video, segments_to_keep)

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _format_transcription_for_prompt(
        self,
        transcriptions: List[str],
        max_lines: int = 200,
    ) -> str:
        if not transcriptions:
            return "No dialogue detected"
        parsed = []
        for line in transcriptions:
            try:
                time_part, text = line.split(":", 1)
                start_str, _ = time_part.split("-")
                parsed.append((float(start_str), text.strip()))
            except Exception:
                continue
        if not parsed:
            return "No valid dialogue"
        if len(parsed) > max_lines:
            half = max_lines // 2
            parsed = (
                parsed[:half]
                + [(-1, "... [middle omitted] ...")]
                + parsed[-half:]
            )
        lines = []
        for ts, text in parsed:
            lines.append(text if ts == -1 else f"[{ts:.1f}s] {text}")
        return "\n".join(lines)

    # ── Gemini call ───────────────────────────────────────────────────────────

    def _call_gemini_for_trim_analysis(
        self,
        video_path: str,
        video_duration: float,
        title_details,
        mic_transcriptions,
        desktop_transcriptions,
    ) -> List[Tuple[float, float]]:
        try:
            self.log_func("\n📤 Uploading video to Gemini …")
            video_file = genai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError("Video upload failed")

            prompt = self._build_trim_prompt(
                video_duration, title_details,
                mic_transcriptions, desktop_transcriptions,
            )
            self.log_func("   Analysing for trim decisions …")
            response = self.vision_analyzer.model.generate_content(
                [prompt, video_file],
                safety_settings=self.vision_analyzer.safety_settings,
            )
            genai.delete_file(video_file.name)
            return self._parse_trim_response(response.text, video_duration)
        except Exception as e:
            self.log_func(f"❌ Gemini trim analysis failed: {e}")
            return []

    def _build_trim_prompt(
        self,
        video_duration: float,
        title_details,
        mic_transcriptions,
        desktop_transcriptions,
    ) -> str:
        title_context = ""
        if title_details:
            title, description, reasoning = title_details
            title_context = (
                f"\n**CONTEXT FROM TITLE ANALYSIS:**\n"
                f"- Title: \"{title}\"\n"
                f"- Description: {description}\n"
                f"- Why it's clip-worthy: {reasoning}\n"
            )

        mic_dialogue = self._format_transcription_for_prompt(
            mic_transcriptions or [])
        game_dialogue = self._format_transcription_for_prompt(
            desktop_transcriptions or [])

        if mic_transcriptions:
            self.log_func(
                f"   Added {len(mic_transcriptions)} mic lines to prompt")
        if desktop_transcriptions:
            self.log_func(
                f"   Added {len(desktop_transcriptions)} game lines to prompt")

        return f"""You are a viral YouTube Shorts editor. Your entire job is to make this clip hit HARDER and feel FASTER.
{title_context}

=== DIALOGUE ===
PLAYER VOICE:
{mic_dialogue}

GAME AUDIO:
{game_dialogue}

VIDEO: {video_duration:.1f}s total. Target: 12–45s final.

══════════════════════════════════════════════
STEP 1 — FIND THE PUNCH POINT (do this first)
══════════════════════════════════════════════
The "punch point" is the single moment of maximum impact: the jumpscare, the kill, the laugh, the fail, the reveal.
It is almost always in the LAST THIRD of the clip.

Identify it precisely. Write its timestamp. Everything else is built around it.

══════════════════════════════════════════════
STEP 2 — MINIMUM VIABLE SETUP
══════════════════════════════════════════════
Ask: "What is the LEAST amount of footage a stranger needs to understand why the punch point is funny/scary/impressive?"
- Context that IS needed: player entering the room, the threat appearing, the setup line being said.
- Context that is NOT needed: walking, menu navigation, dead air, repeating commentary, anything after the laugh dies down.
- A good setup is 5–15 seconds. Longer setups need extraordinary justification.

══════════════════════════════════════════════
STEP 3 — CUT AGGRESSIVELY BETWEEN THEM
══════════════════════════════════════════════
Hard jump cuts (even mid-movement) are fine and actually preferred for pacing.
- Any pause > 1.5 seconds that isn't tension-building: CUT IT.
- Trailing commentary after the punchline dies: CUT IT (end within 3s of the laugh/reaction peak).
- Repetitive phrases: CUT THEM.
- Multiple setup attempts: keep only the BEST one.

══════════════════════════════════════════════
STEP 4 — VERIFY
══════════════════════════════════════════════
Read your proposed cut list back. Ask:
1. Does every kept second directly serve the punch or the minimum setup?
2. Does anything feel slow? If yes, cut more.
3. Is the punch point fully intact (not cut short)?

══════════════════════════════════════════════
OUTPUT — JSON ONLY, no other text
══════════════════════════════════════════════
{{
  "punch_point_time": 28.5,
  "punch_point_description": "Player gets jumpscared and screams",
  "setup_rationale": "Need 6s of player walking toward door for stakes",
  "analysis": "Removed 18s of dead-air walking and post-laugh rambling",
  "segments_to_keep": [
    {{"start": 22.0, "end": 29.5, "reason": "Setup walk + jumpscare + immediate reaction"}},
    {{"start": 31.0, "end": 37.0, "reason": "Follow-up commentary that lands"}}
  ],
  "estimated_duration": 13.5,
  "cuts_made": [
    "0–22s: irrelevant exploration",
    "29.5–31s: dead air pause",
    "37s–end: rambling wind-down"
  ]
}}"""

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_trim_response(
        self,
        response_text: str,
        video_duration: float,
    ) -> List[Tuple[float, float]]:
        try:
            self.log_func("\n📋 Parsing trim decisions …")
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                self.log_func("⚠️  No JSON found in response")
                return []

            data = json.loads(json_match.group(0))

            punch = data.get("punch_point_time")
            punch_desc = data.get("punch_point_description", "")
            if punch is not None:
                self.log_func(
                    f"\n   🥊 Punch point: {punch:.1f}s — {punch_desc}")

            setup = data.get("setup_rationale", "")
            if setup:
                self.log_func(f"   📐 Setup rationale: {setup}")

            analysis = data.get("analysis", "")
            if analysis:
                self.log_func(f"   ✂️  Analysis: {analysis}")

            segments_data = data.get("segments_to_keep", [])
            if not segments_data:
                self.log_func("⚠️  No segments returned")
                return []

            segments: List[Tuple[float, float]] = []
            for i, seg in enumerate(segments_data):
                start = max(0.0, float(seg["start"]))
                end = min(video_duration, float(seg["end"]))
                if start >= end:
                    continue
                reason = seg.get("reason", "")

                # Add end-buffer to last segment only
                if i == len(segments_data) - 1:
                    buffered_end = min(end + 1.0, video_duration)
                    if buffered_end > end:
                        self.log_func(
                            f"   📏 End-buffer: {end:.1f}s → {buffered_end:.1f}s")
                        end = buffered_end

                segments.append((start, end))
                self.log_func(
                    f"   ✓ Keep {start:.1f}s–{end:.1f}s"
                    f" ({end - start:.1f}s) — {reason}"
                )

            cuts = data.get("cuts_made", [])
            if cuts:
                self.log_func("\n   Cuts made:")
                for c in cuts:
                    self.log_func(f"   ✂️  {c}")

            est = data.get(
                "estimated_duration",
                sum(e - s for s, e in segments),
            )
            self.log_func(f"\n   ⏱  Estimated final duration: {est:.1f}s")
            return segments

        except Exception as e:
            self.log_func(f"❌ Error parsing response: {e}")
            self.log_func(f"   Raw: {response_text[:500]}")
            return []

    # ── FFmpeg execution ──────────────────────────────────────────────────────

    def _apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]],
    ) -> bool:
        try:
            self.log_func("\n✂️  Applying trim …")
            temp_dir = tempfile.gettempdir()
            segment_files = []

            for i, (start, end) in enumerate(segments_to_keep):
                seg_file = os.path.join(
                    temp_dir, f"trim_segment_{i:03d}.mp4")
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", input_video,
                    "-to", str(end - start),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    seg_file,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True)
                if result.returncode == 0 and \
                        os.path.exists(seg_file) and \
                        os.path.getsize(seg_file) > 1000:
                    segment_files.append(seg_file)
                    self.log_func(
                        f"   ✓ Segment {i + 1}/{len(segments_to_keep)}"
                        f" ({end - start:.1f}s)")
                else:
                    self.log_func(
                        f"   ✗ Segment {i + 1} extraction failed")

            if not segment_files:
                self.log_func("❌ No segments extracted")
                return False

            if len(segment_files) == 1:
                import shutil
                shutil.copy2(segment_files[0], output_video)
                os.remove(segment_files[0])
                return True

            concat_list = os.path.join(
                temp_dir, "trim_concat_list.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for sf in segment_files:
                    sf_abs = os.path.abspath(sf).replace("\\", "/")
                    f.write(f"file '{sf_abs}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                "-movflags", "+faststart",
                output_video,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            for sf in segment_files:
                try:
                    os.remove(sf)
                except Exception:
                    pass
            try:
                os.remove(concat_list)
            except Exception:
                pass

            if result.returncode == 0 and os.path.exists(output_video):
                final_dur = get_video_duration(output_video, self.log_func)
                expected = sum(e - s for s, e in segments_to_keep)
                if final_dur < 1.0 or final_dur < expected * 0.4:
                    self.log_func(
                        f"❌ Output suspiciously short: {final_dur:.1f}s")
                    return False
                self.log_func("   ✅ Trim applied successfully")
                return True

            self.log_func(
                f"❌ Concat failed: {result.stderr[-500:]}")
            return False

        except Exception as e:
            self.log_func(f"❌ Trim failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return False