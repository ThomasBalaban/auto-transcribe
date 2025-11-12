# clip_editor/intelligent_trimmer.py

"""
Intelligent Clip Trimming System
Uses Gemini to analyze entire video and decide what to cut in one pass.
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
    Analyzes entire video with Gemini and gets trimming decisions in one call.
    Much simpler and more scalable than multi-step analysis.
    """
    
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.vision_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
        
        # Configuration
        self.max_clip_duration = 60.0  # Target maximum duration
        self.min_clip_duration = 15.0  # Minimum to keep it worthwhile

    def analyze_for_trim(
        self,
        video_path: str,
        title_details: Optional[Tuple[str, str, str]] = None
    ) -> List[Tuple[float, float]]:
        """
        Analyze video and return trim segments WITHOUT actually cutting.
        
        Returns:
            List of (start, end) time ranges to keep
        """
        try:
            self.log_func("\n" + "="*60)
            self.log_func("🎬 INTELLIGENT CLIP TRIMMING - ANALYSIS PHASE")
            self.log_func("="*60)
            
            video_duration = get_video_duration(video_path, self.log_func)
            
            # Get trim decisions from Gemini
            segments_to_keep = self._call_gemini_for_trim_analysis(
                video_path, video_duration, title_details
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
        This is now separate so it can be called after subtitle processing.
        
        Returns:
            Success status
        """
        return self._apply_trim(input_video, output_video, segments_to_keep)

    def _call_gemini_for_trim_analysis(
        self,
        video_path: str,
        video_duration: float,
        title_details: Optional[Tuple[str, str, str]]
    ) -> List[Tuple[float, float]]:
        """
        Upload video to Gemini and get trimming decisions.
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
            
            # Build the prompt
            prompt = self._build_trim_prompt(video_duration, title_details)
            
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
        title_details: Optional[Tuple[str, str, str]]
    ) -> str:
        """
        Build the prompt for Gemini trim analysis.
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
        
        prompt = f"""You are an expert video editor for gaming content. Your goal is to make this clip PUNCHY and ENGAGING by removing boring/unnecessary parts.

{title_context}

**VIDEO INFO:**
- Total Duration: {video_duration:.1f} seconds
- Target Duration: 15-60 seconds (shorter is better if it maintains impact)
- Content Type: Gaming clips (often horror games with jumpscares or funny moments)

**YOUR TASK - THREE STEPS:**

**STEP 1: UNDERSTAND THE VIDEO**
Watch the entire video and identify:
- What type of moment is this? (jumpscare, funny reaction, unexpected event, clutch play, glitch, etc.)
- What is the narrative structure? (tension building → payoff, setup → punchline, escalating chaos, etc.)
- Where exactly is the climax/payoff? (timestamp when the "thing" happens)
- What makes this clip-worthy? (refer to the title analysis context if provided)

**STEP 2: IDENTIFY ESSENTIAL ELEMENTS**
Determine what the viewer NEEDS to see/hear:
- What creates necessary context? (e.g., "why is the player entering this room?")
- What builds tension or sets up the payoff? (silence before a jumpscare, dialogue before a punchline)
- What dialogue or reactions are important? (funny commentary, scared reactions)
- Are there natural "beats" or chapters in the clip?

For horror clips: Remember that silence, slow pacing, or low activity might be ESSENTIAL for building tension before a jumpscare.

For funny clips: The setup might seem mundane but could be necessary for the punchline to land.

**STEP 3: DECIDE WHAT TO CUT**
Only remove segments that are:
- Truly redundant (the same point is made multiple times)
- Post-climax footage that adds nothing (unless there's a good reaction/follow-up)
- Pre-setup footage where nothing relevant to the story is happening yet
- Dead time that doesn't serve the pacing (long pauses that aren't building tension)

**GUIDELINES:**
1. ALWAYS include the climax/payoff moment (this is non-negotiable)
2. Preserve pacing that serves the narrative (tension-building, comedic timing, etc.)
3. Keep dialogue that provides context or is inherently funny/interesting
4. Segments should flow naturally - avoid cutting mid-sentence or mid-action
5. Aim for 15-60 seconds total, but prioritize telling a complete story over hitting a specific duration
6. When in doubt, keep more rather than less - it's better to be slightly long than to lose essential context

**WHAT NOT TO DO:**
- Don't cut silence or slow moments that are building tension (especially in horror)
- Don't remove setup just because it seems boring - ask if it's necessary for the payoff
- Don't cut to exactly 15 seconds if it would ruin the narrative flow
- Don't remove reactions or follow-up moments if they add to the comedy/horror

**OUTPUT FORMAT (JSON ONLY):**
{{
  "analysis": "Brief explanation of what makes this clip good and what you're cutting",
  "segments_to_keep": [
    {{"start": 0.0, "end": 15.5, "reason": "Essential setup showing player entering area"}},
    {{"start": 42.3, "end": 68.0, "reason": "The entire climax and punchline"}}
  ],
  "estimated_duration": 41.2,
  "cuts_made": ["Removed 5-15s: repetitive walking", "Removed 15-42s: silent gameplay"]
}}

**IMPORTANT:**
- Timestamps should be in seconds (decimals OK)
- Segments should be in chronological order
- Don't overlap segments
- Return ONLY the JSON, no extra text

Now analyze this video and give me the trim decisions:"""
        
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
            
            # Extract segments
            segments_data = data.get('segments_to_keep', [])
            
            if not segments_data:
                self.log_func("⚠️ No segments returned")
                return []
            
            segments = []
            for seg in segments_data:
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
            
            # Extract each segment with re-encoding to ensure compatibility
            for i, (start, end) in enumerate(segments_to_keep):
                segment_file = os.path.join(temp_dir, f"trim_segment_{i:03d}.mp4")
                
                # Use re-encoding instead of copy to ensure clean segments
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),           # Seek BEFORE input for speed
                    '-i', input_video,
                    '-to', str(end - start),     # Duration instead of end time
                    '-c:v', 'libx264',           # Re-encode video
                    '-preset', 'fast',
                    '-crf', '18',                # High quality
                    '-c:a', 'aac',               # Re-encode audio
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
                    self.log_func(f"   ✗ Failed to extract segment {i+1} - file too small or missing")
            
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
            
            # Concatenate segments using concat demuxer
            concat_list_file = os.path.join(temp_dir, "trim_concat_list.txt")
            with open(concat_list_file, 'w', encoding='utf-8') as f:
                for seg_file in segment_files:
                    # Escape single quotes and use absolute paths
                    seg_file_abs = os.path.abspath(seg_file)
                    seg_file_normalized = seg_file_abs.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{seg_file_normalized}'\n")
            
            self.log_func(f"   Concatenating {len(segment_files)} segments...")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_file,
                '-c', 'copy',            # Now we can copy since all segments are same format
                '-movflags', '+faststart',
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup segment files
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
                # Verify the output isn't corrupted
                final_duration = get_video_duration(output_video, self.log_func)
                expected_duration = sum(end - start for start, end in segments_to_keep)
                
                if final_duration < 1.0 or final_duration < expected_duration * 0.5:
                    self.log_func(f"❌ Output video suspiciously short: {final_duration:.1f}s (expected ~{expected_duration:.1f}s)")
                    self.log_func(f"   FFmpeg stderr: {result.stderr[-1000:]}")
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