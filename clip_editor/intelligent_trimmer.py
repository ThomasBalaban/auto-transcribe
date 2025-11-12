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
        
        prompt = f"""You are an expert video editor for gaming content. Your goal is to make this clip PUNCHY and ENGAGING by removing boring/unnecessary parts while preserving the story.

{title_context}

=== DIALOGUE CONTEXT (CRITICAL!) ===
**PLAYER COMMENTARY (MY VOICE):**
{mic_dialogue}

**GAME AUDIO (NPCs/Events/Music):**
{game_dialogue}

**VIDEO INFO:**
- Total Duration: {video_duration:.1f} seconds
- Target Duration: 15-60 seconds (shorter is better if it maintains impact)
- Content Type: Gaming clips (often horror games with jumpscares or funny moments)

**YOUR TASK - THREE STEPS:**

**STEP 1: UNDERSTAND THE NARRATIVE**
Using the dialogue above and watching the video:
- What is the story arc? (tension building → payoff, setup → punchline, etc.)
- Where exactly is the climax/payoff? (when the "thing" happens)
- What dialogue is essential for understanding the payoff?
- Are there call-and-response moments between player and game?

**STEP 2: IDENTIFY ESSENTIAL DIALOGUE**
Look at the transcriptions and determine what the viewer NEEDS to hear:
- Setup dialogue that provides context
- Tension-building commentary
- The punchline or reaction
- Important NPC dialogue or game events

**CRITICAL DIALOGUE RULES:**
- If player says something ironic before an event (e.g., "I'm not scared"), KEEP IT
- If there's a conversation between player and game, keep both sides
- Reactions are often more important than the event itself
- Don't cut mid-sentence on EITHER track

**STEP 3: DECIDE WHAT TO CUT**
Only remove segments that are:
- Pre-setup where nothing story-relevant is said or shown
- Post-climax footage with no good follow-up dialogue
- Repetitive commentary that doesn't advance the story
- Dead time that doesn't build tension

**GUIDELINES:**
1. ALWAYS keep the climax/payoff moment
2. Preserve dialogue that sets up the payoff (from EITHER track)
3. Keep player reactions - they're often the best part
4. Segments should flow naturally - don't cut mid-sentence
5. Aim for 15-60 seconds, but prioritize telling a complete story
6. When in doubt, keep more dialogue rather than less

**WHAT NOT TO DO:**
- Don't cut silence if it's building tension (especially in horror)
- Don't remove setup dialogue just because it seems boring
- Don't cut between a question and its answer (either track)
- Don't split up call-and-response moments

**OUTPUT FORMAT (JSON ONLY):**
{{
  "analysis": "Brief explanation referencing specific dialogue that makes this clip good",
  "segments_to_keep": [
    {{"start": 0.0, "end": 15.5, "reason": "Player says 'I'm not scared' - essential setup"}},
    {{"start": 42.3, "end": 68.0, "reason": "The climax, scream, and follow-up commentary"}}
  ],
  "estimated_duration": 41.2,
  "cuts_made": [
    "Removed 15.5-42.3s: Walking with no dialogue or tension",
    "Removed 68.0-end: Quiet aftermath with no interesting commentary"
  ],
  "dialogue_preserved": [
    "[45s] 'I'm not scared'",
    "[48s] 'OH MY GOD!'",
    "[50s] 'That was terrifying'"
  ]
}}

**IMPORTANT:**
- Timestamps should be in seconds (decimals OK)
- Segments should be in chronological order
- Don't overlap segments
- Return ONLY the JSON, no extra text
- Reference specific dialogue timestamps in your reasoning

Now analyze this video with its dialogue and give me the trim decisions:"""
        
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
                    end = min(end + 1.5, video_duration)  # Add 1.5s buffer, capped at video duration
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