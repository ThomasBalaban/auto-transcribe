# clip_editor/intelligent_trimmer.py

"""
Intelligent Clip Trimming System
Uses Gemini to analyze entire video with dialogue context and decide what to cut.
"""

from typing import List, Dict, Tuple, Optional
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from video_utils import get_video_duration
import subprocess
import os
import json
import re
import google.generativeai as genai # type: ignore
import time

class IntelligentTrimmer:
    """
    Analyzes entire video with Gemini and gets trimming decisions with dialogue awareness.
    """
    
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.vision_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
        
        # Configuration
        self.max_clip_duration = 60.0
        self.min_clip_duration = 15.0

    def analyze_for_trim(
        self,
        video_path: str,
        title_details: Optional[Tuple[str, str, str]] = None,
        mic_transcriptions: Optional[List[str]] = None,
        desktop_transcriptions: Optional[List[str]] = None
    ) -> List[Tuple[float, float]]:
        """
        Analyze video and return trim segments WITHOUT actually cutting.
        Now includes dialogue context for better decisions.
        
        Returns:
            List of (start, end) time ranges to keep
        """
        try:
            self.log_func("\n" + "="*60)
            self.log_func("🎬 INTELLIGENT CLIP TRIMMING - ANALYSIS PHASE")
            self.log_func("="*60)
            
            video_duration = get_video_duration(video_path, self.log_func)
            
            # Get trim decisions from Gemini with dialogue context
            segments_to_keep = self._call_gemini_for_trim_analysis(
                video_path, video_duration, title_details,
                mic_transcriptions, desktop_transcriptions
            )
            
            return segments_to_keep
            
        except Exception as e:
            self.log_func(f"❌ Trim analysis failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []
        
  
    def apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]]
    ) -> bool:
        """
        Execute the trim plan on a video.
        This is separate so it can be called after subtitle processing.
        
        Returns:
            Success status
        """
        return self._apply_trim(input_video, output_video, segments_to_keep)

    def _format_transcription_for_prompt(
        self, 
        transcriptions: List[str],
        max_lines: int = 200
    ) -> str:
        """Format transcriptions with timestamps for Gemini."""
        if not transcriptions:
            return "No dialogue detected"
        
        # Parse and format
        parsed = []
        for line in transcriptions:
            try:
                time_part, text = line.split(':', 1)
                start_str, end_str = time_part.split('-')
                start_time = float(start_str)
                text = text.strip()
                parsed.append((start_time, text))
            except:
                continue
        
        if not parsed:
            return "No valid dialogue"
        
        # Limit to max_lines
        if len(parsed) > max_lines:
            # Take beginning and end
            begin = parsed[:max_lines//2]
            end = parsed[-(max_lines//2):]
            parsed = begin + [(-1, "... [middle section omitted] ...")] + end
        
        # Format with timestamps
        formatted_lines = []
        for timestamp, text in parsed:
            if timestamp == -1:
                formatted_lines.append(text)
            else:
                formatted_lines.append(f"[{timestamp:.1f}s] {text}")
        
        return "\n".join(formatted_lines)

    def _call_gemini_for_trim_analysis(
        self,
        video_path: str,
        video_duration: float,
        title_details: Optional[Tuple[str, str, str]],
        mic_transcriptions: Optional[List[str]],
        desktop_transcriptions: Optional[List[str]]
    ) -> List[Tuple[float, float]]:
        """
        Upload video to Gemini and get trimming decisions with dialogue context.
        """
        try:
            self.log_func("\n📤 Uploading video to Gemini for analysis...")
            
            # Upload the video
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            self.log_func("   Waiting for video processing...")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise ValueError("Video upload failed")
            
            # Build the prompt with dialogue context
            prompt = self._build_trim_prompt(
                video_duration, title_details,
                mic_transcriptions, desktop_transcriptions
            )
            
            # Call Gemini
            self.log_func("   Analyzing video for trim decisions...")
            response = self.vision_analyzer.model.generate_content(
                [prompt, video_file],
                safety_settings=self.vision_analyzer.safety_settings
            )
            
            # Cleanup
            genai.delete_file(video_file.name)
            
            # Parse response
            segments = self._parse_trim_response(response.text, video_duration)
            
            return segments
            
        except Exception as e:
            self.log_func(f"❌ Gemini trim analysis failed: {e}")
            return []
    
    
    def _build_trim_prompt(
        self,
        video_duration: float,
        title_details: Optional[Tuple[str, str, str]],
        mic_transcriptions: Optional[List[str]],
        desktop_transcriptions: Optional[List[str]]
    ) -> str:
        """
        Build the prompt for Gemini trim analysis with dialogue context.
        """
        title_context = ""
        if title_details:
            title, description, reasoning = title_details
            title_context = f"""
**CONTEXT FROM TITLE ANALYSIS:**
- Title: "{title}"
- Description: {description}
- Why it's clip-worthy: {reasoning}
"""
        
        # Format dialogue for prompt
        mic_dialogue = "Not available"
        game_dialogue = "Not available"
        
        if mic_transcriptions:
            mic_dialogue = self._format_transcription_for_prompt(mic_transcriptions)
            self.log_func(f"   Added {len(mic_transcriptions)} mic dialogue lines to prompt")
        
        if desktop_transcriptions:
            game_dialogue = self._format_transcription_for_prompt(desktop_transcriptions)
            self.log_func(f"   Added {len(desktop_transcriptions)} game dialogue lines to prompt")
        
        prompt = f"""You are an expert video editor for TikTok and YouTube Shorts. Your goal is to make this clip highly ENGAGING and DENSE by ruthlessly removing dead air, fluff, and unnecessary context.

{title_context}

=== DIALOGUE CONTEXT (CRITICAL!) ===
**PLAYER COMMENTARY (MY VOICE):**
{mic_dialogue}

**GAME AUDIO (NPCs/Events/Music):**
{game_dialogue}

**VIDEO INFO:**
- Total Duration: {video_duration:.1f} seconds
- Target Duration: 15-45 seconds (Prioritize extremely fast pacing. Cut the fat.)
- Content Type: Gaming clips (focus on funny moments, reactions, or jumpscares)

**YOUR TASK - THREE STEPS:**

**STEP 1: FIND THE CORE ARC**
- What is the absolute minimum setup required for the payoff (the reaction/jumpscare/joke) to make sense?

**STEP 2: RUTHLESS CUTTING (REMOVE THE FLUFF)**
Look at the transcriptions and determine what to CUT. BE RUTHLESS:
- Cut ALL "dead air" where nothing is happening visually or audibly (e.g., long walks, menu navigation).
- Cut repetitive or trailing commentary that drags on after the climax has finished. Stop the clip soon after the punchline or peak reaction.
- If there is a pause of more than 2-3 seconds where nothing tension-building or funny happens, CUT IT. Jump cuts are highly encouraged to skip boring parts.

**STEP 3: FINALIZE THE PLAN**
**GUIDELINES:**
1. Do not cut mid-sentence.
2. It is highly encouraged to make multiple cuts (e.g., Keep 5s of setup -> Cut 15s of dead air -> Keep 10s of payoff and reaction).
3. Density is key. Every single second you keep MUST be entertaining or absolutely necessary for context.

**OUTPUT FORMAT (JSON ONLY):**
{{
  "analysis": "Brief explanation of what fluff was removed and why the clip is now dense and engaging",
  "segments_to_keep": [
    {{"start": 0.0, "end": 8.5, "reason": "Essential setup dialogue"}},
    {{"start": 28.2, "end": 40.0, "reason": "The climax and the immediate reaction"}}
  ],
  "estimated_duration": 20.3,
  "cuts_made": [
    "Removed 8.5-28.2s: Long silence and walking",
    "Removed 40.0-end: Boring trailing commentary"
  ],
  "dialogue_preserved": [
    "[2s] 'Let's check this room'",
    "[30s] 'OH MY GOD RUN'"
  ]
}}

**IMPORTANT:**
- Timestamps should be in seconds (decimals OK)
- Segments should be in chronological order
- Return ONLY the JSON, no extra text
"""
        return prompt


    def _parse_trim_response(
        self,
        response_text: str,
        video_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Parse Gemini's response to extract segments to keep.
        """
        try:
            self.log_func("\n📋 Parsing trim decisions...")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                self.log_func("⚠️ Could not find JSON in response")
                return []
            
            data = json.loads(json_match.group(0))
            
            # Log the analysis
            analysis = data.get('analysis', 'No analysis provided')
            self.log_func(f"\n   Analysis: {analysis}")
            
            # Log preserved dialogue
            preserved_dialogue = data.get('dialogue_preserved', [])
            if preserved_dialogue:
                self.log_func("\n   Key dialogue preserved:")
                for dialogue in preserved_dialogue:
                    self.log_func(f"   💬 {dialogue}")
            
            # Extract segments
            segments_data = data.get('segments_to_keep', [])
            
            if not segments_data:
                self.log_func("⚠️ No segments returned")
                return []
            
            segments = []
            for i, seg in enumerate(segments_data):
                start = float(seg['start'])
                end = float(seg['end'])
                reason = seg.get('reason', 'No reason given')
                
                # Validate timestamps
                if start < 0:
                    start = 0
                if end > video_duration:
                    end = video_duration
                if start >= end:
                    continue
                
                # ✅ NEW: Add 1.5 second buffer to the LAST segment to prevent cutting off endings
                is_last_segment = (i == len(segments_data) - 1)
                if is_last_segment:
                    original_end = end
                    end = min(end + 1, video_duration)  # Add 1.5s buffer, capped at video duration
                    if end > original_end:
                        self.log_func(f"   📏 Added 1.5s buffer to final segment: {original_end:.1f}s → {end:.1f}s")
                
                segments.append((start, end))
                self.log_func(f"   ✓ Keep: {start:.1f}s - {end:.1f}s ({end-start:.1f}s) - {reason}")
            
            # Log cuts made
            cuts = data.get('cuts_made', [])
            if cuts:
                self.log_func("\n   Cuts made:")
                for cut in cuts:
                    self.log_func(f"   ✂️  {cut}")
            
            # Log final duration
            estimated_duration = data.get('estimated_duration', sum(e - s for s, e in segments))
            self.log_func(f"\n   Estimated final duration: {estimated_duration:.1f}s")
            
            return segments
            
        except Exception as e:
            self.log_func(f"❌ Error parsing response: {e}")
            self.log_func(f"   Raw response: {response_text[:500]}")
            return []
    
    def _apply_trim(
        self,
        input_video: str,
        output_video: str,
        segments_to_keep: List[Tuple[float, float]]
    ) -> bool:
        """
        Apply the actual video trim using FFmpeg.
        """
        try:
            self.log_func("\n✂️ Applying trim to video...")
            
            import tempfile
            temp_dir = tempfile.gettempdir()
            segment_files = []
            
            # Extract each segment with re-encoding
            for i, (start, end) in enumerate(segments_to_keep):
                segment_file = os.path.join(temp_dir, f"trim_segment_{i:03d}.mp4")
                
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-i', input_video,
                    '-to', str(end - start),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    segment_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.log_func(f"⚠️ Warning: Segment {i+1} extraction failed")
                    self.log_func(f"   Error: {result.stderr[-500:]}")
                    continue
                
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 1000:
                    segment_files.append(segment_file)
                    segment_duration = end - start
                    self.log_func(f"   ✓ Extracted segment {i+1}/{len(segments_to_keep)} ({segment_duration:.1f}s)")
                else:
                    self.log_func(f"   ✗ Failed to extract segment {i+1}")
            
            if not segment_files:
                self.log_func("❌ No segments extracted successfully")
                return False
            
            # If only one segment, just copy it
            if len(segment_files) == 1:
                import shutil
                shutil.copy2(segment_files[0], output_video)
                os.remove(segment_files[0])
                self.log_func("   ✅ Single segment - copied directly")
                return True
            
            # Concatenate segments
            concat_list_file = os.path.join(temp_dir, "trim_concat_list.txt")
            with open(concat_list_file, 'w', encoding='utf-8') as f:
                for seg_file in segment_files:
                    seg_file_abs = os.path.abspath(seg_file)
                    seg_file_normalized = seg_file_abs.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{seg_file_normalized}'\n")
            
            self.log_func(f"   Concatenating {len(segment_files)} segments...")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_file,
                '-c', 'copy',
                '-movflags', '+faststart',
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup
            for seg_file in segment_files:
                try:
                    os.remove(seg_file)
                except:
                    pass
            try:
                os.remove(concat_list_file)
            except:
                pass
            
            if result.returncode == 0 and os.path.exists(output_video):
                final_duration = get_video_duration(output_video, self.log_func)
                expected_duration = sum(end - start for start, end in segments_to_keep)
                
                if final_duration < 1.0 or final_duration < expected_duration * 0.5:
                    self.log_func(f"❌ Output suspiciously short: {final_duration:.1f}s (expected ~{expected_duration:.1f}s)")
                    return False
                
                self.log_func("   ✅ Video trim applied successfully")
                return True
            else:
                self.log_func(f"❌ Concatenation failed")
                self.log_func(f"   FFmpeg stderr: {result.stderr[-1000:]}")
                return False
            
        except Exception as e:
            self.log_func(f"❌ Failed to apply trim: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return False