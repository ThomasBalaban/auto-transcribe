# title_gen/title_generator.py
"""
AI-powered title generation for gaming clips using Gemini.
Analyzes the video directly with dialogue context for better titles.
"""

import json
from typing import Optional, Tuple, List
import os
import time
import google.generativeai as genai # type: ignore
import re
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from video_utils import get_video_duration
from PIL import Image


class TitleGenerator:
    """
    Generates YouTube-style titles with dialogue context.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print

    def _extract_climax_frames(self, video_path: str) -> List[Image.Image]:
        """Extracts keyframes from the last 20 seconds of the video."""
        try:
            self.log_func("... extracting climax frames from last 20 seconds.")
            video_duration = get_video_duration(video_path, self.log_func)
            climax_start_time = max(0, video_duration - 20)
            
            vision_analyzer = GeminiVisionAnalyzer(log_func=self.log_func)
            frames = vision_analyzer.extract_frames_from_video(
                video_path,
                start_time=climax_start_time,
                duration=video_duration - climax_start_time
            )
            self.log_func(f"... found {len(frames)} climax frames.")
            return frames
        except Exception as e:
            self.log_func(f"⚠️  Could not extract climax frames: {e}")
            return []

    def generate_title(
        self,
        video_path: str,
        shorts_analysis_path: str = "shorts_analysis.json",
        mic_transcriptions: Optional[List[str]] = None,      # ✅ NEW
        desktop_transcriptions: Optional[List[str]] = None   # ✅ NEW
    ) -> Optional[Tuple[str, str, str]]:
        """
        Main entry point for title generation with dialogue context.
        Returns a tuple containing (title, description, reasoning) or None.
        """
        try:
            self.log_func("\n🎬 Starting dialogue-aware title generation...")

            if not os.path.exists(shorts_analysis_path):
                self.log_func(f"⚠️ {shorts_analysis_path} not found. Cannot generate title.")
                return None

            with open(shorts_analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            climax_frames = self._extract_climax_frames(video_path)

            # ✅ NEW: Pass transcriptions to prompt builder
            prompt = self._build_gemini_prompt(
                analysis_data,
                mic_transcriptions,
                desktop_transcriptions
            )
            
            title_details = self._call_gemini_api(prompt, video_path, climax_frames)

            if title_details and title_details[0]:
                title, description, reasoning = title_details
                self.log_func("\n" + "="*60)
                self.log_func("✅ TITLE GENERATION ANALYSIS COMPLETE")
                self.log_func(f"   - VIDEO DESCRIPTION: {description}")
                self.log_func(f"   - CHOSEN TITLE: {title}")
                self.log_func(f"   - REASONING: {reasoning}")
                self.log_func("="*60 + "\n")
                return title, description, reasoning
            else:
                self.log_func("⚠️  Title generation returned an empty result.")
                return None

        except Exception as e:
            self.log_func(f"❌ Title generation failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return None

    def _format_transcription_for_prompt(
        self, 
        transcriptions: List[str], 
        focus_end: bool = True,
        max_words: int = 150
    ) -> str:
        """
        Format transcriptions for Gemini prompt.
        If focus_end=True, prioritizes last 30 seconds.
        """
        if not transcriptions:
            return "No dialogue detected"
        
        # Parse timestamps
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
        
        # If focusing on end, get last 30 seconds
        if focus_end and len(parsed) > 0:
            max_time = parsed[-1][0]
            cutoff_time = max(0, max_time - 30)
            parsed = [(t, txt) for t, txt in parsed if t >= cutoff_time]
        
        # Limit total words
        if len(parsed) > max_words:
            parsed = parsed[-max_words:]
        
        # Format nicely with timestamps
        formatted_lines = []
        for timestamp, text in parsed:
            formatted_lines.append(f"[{timestamp:.1f}s] {text}")
        
        return "\n".join(formatted_lines)

    def _build_gemini_prompt(
        self, 
        analysis_data: dict,
        mic_transcriptions: Optional[List[str]] = None,      # ✅ NEW
        desktop_transcriptions: Optional[List[str]] = None   # ✅ NEW
    ) -> str:
        """Builds the comprehensive prompt for Gemini with dialogue context."""

        # ✅ NEW: Format transcriptions for better readability
        mic_dialogue = "Not available"
        game_dialogue = "Not available"
        
        if mic_transcriptions:
            mic_dialogue = self._format_transcription_for_prompt(mic_transcriptions, focus_end=True)
            self.log_func(f"   Formatted {len(mic_transcriptions)} mic words for prompt (last 30s focus)")
        
        if desktop_transcriptions:
            game_dialogue = self._format_transcription_for_prompt(desktop_transcriptions, focus_end=True)
            self.log_func(f"   Formatted {len(desktop_transcriptions)} game words for prompt (last 30s focus)")

        successful_titles = [
            f"- \"{item['title']}\" (Views: {item['views']:,}): {item['gemini_analysis']['title_effectiveness_analysis']}"
            for item in analysis_data.get('shorts', [])[:15]
        ]
        
        prompt = f"""You are an expert YouTube content strategist specializing in viral gaming clips.

=== STYLE ANALYSIS (My Most Successful Videos) ===
Here is an analysis of my top-performing YouTube Shorts. Your title must match this voice.
{chr(10).join(successful_titles)}

=== DIALOGUE CONTEXT (CRITICAL!) ===
**PLAYER COMMENTARY (MY VOICE - MOST IMPORTANT):**
{mic_dialogue}

**GAME AUDIO (NPCs/Events/Music):**
{game_dialogue}

=== CRITICAL INSTRUCTIONS ===
1. **PRIORITIZE WHAT WAS SAID**: The player's actual words are your PRIMARY source for titles.
   - Look for ironic statements before outcomes ("I'm not scared" → gets scared)
   - Funny reactions or commentary
   - Quotable moments that capture the essence
   
2. **DIALOGUE PATTERNS TO EXPLOIT:**
   - "I'm not scared" → [gets scared] = "Famous Last Words"
   - Funny one-liners or exclamations = Direct quote or inspired titles
   - Call-and-response between player and game = Highlight the exchange
   - Player confidence → immediate failure = Irony-based titles
   
3. **FOCUS ON THE CLIMAX**: The last 15-20 seconds of dialogue + video are most important.

4. **EXAMPLES OF DIALOGUE-DRIVEN TITLES:**
   - Player: "This is easy" → Dies immediately → Title: "Spoke Too Soon"
   - Player: "WHAT THE—" → Title: Use the authentic reaction
   - NPC: "Don't go in there" → Player goes in → Title: "I Should Have Listened"
   - Player: "I got this" → Fails → Title: "He Did Not, In Fact, Got This"

5. **AVOID UNNECESSARY CONTEXT**: Don't mention friends' usernames or stream-specific details.

6. **DO NOT GUESS THE GAME TITLE**: Only include game name in parentheses (Game Name) if you are 100% certain. If unsure, OMIT the game title.

7. **GET THE CLICK**: Humor, irony, intrigue, or curiosity gaps are more important than perfect grammar. Make people want to click.

8. **AVOID PROFANITY**: Do not include any direct curse words or allusions to them (e.g., "f---", "s**t"). The word "hell" is acceptable. Innuendo and suggestive humor are fine, but the final title must remain advertiser-friendly and avoid explicit language.

=== YOUR TASK ===
Analyze the DIALOGUE FIRST (especially last 30 seconds), then the video frames. Create a title that captures the best moment - preferably derived from what was actually said.

The video frames show the visual climax - combine this with dialogue for maximum impact.

Respond with a single JSON object:
{{
  "video_description": "What happened (focus on what was SAID and what happened visually)",
  "title": "Your title (preferably derived from or inspired by dialogue)",
  "reasoning": "Why this title works (reference specific dialogue quotes if used)"
}}

Example Response:
{{
  "video_description": "Player confidently states 'I'm not scared of this game' while exploring, then immediately encounters a jumpscare and screams 'OH MY GOD'",
  "title": "Famous Last Words",
  "reasoning": "The title captures the ironic contrast between the player's confident statement and immediate fear. This matches your successful video pattern of ironic outcomes, and the phrase 'famous last words' is instantly recognizable."
}}
"""
        
        return prompt

    def _call_gemini_api(self, prompt: str, video_path: str, climax_frames: List[Image.Image]) -> Optional[Tuple[str, str, str]]:
        """Calls the Gemini API with the video, climax frames, and prompt."""
        try:
            analyzer = GeminiVisionAnalyzer(log_func=self.log_func)

            self.log_func(f"Uploading video for title analysis: {video_path}")
            video_file = genai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError("Video upload for title generation failed.")

            api_content = [prompt, video_file]
            if climax_frames:
                self.log_func("... adding climax frames to the prompt.")
                api_content.append("\nHere are the critical climax frames to focus on:")
                api_content.extend(climax_frames)

            self.log_func("Generating title analysis with Gemini...")
            response = analyzer.model.generate_content(api_content, safety_settings=analyzer.safety_settings)
            genai.delete_file(video_file.name)

            response_text = response.text.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                self.log_func(f"⚠️  Could not find valid JSON in Gemini response: {response_text}")
                return None
            
            parsed_json = json.loads(json_match.group(0))
            
            title = self._clean_title(parsed_json.get("title", ""))
            description = parsed_json.get("video_description", "N/A")
            reasoning = parsed_json.get("reasoning", "N/A")

            return title, description, reasoning

        except Exception as e:
            self.log_func(f"💥 Gemini API call for title generation failed: {e}")
            return None

    def _clean_title(self, title: str) -> str:
        title = title.replace('**', '').replace('*', '')
        title = title.split('\n')[0]
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        if len(title) > 70:
            self.log_func(f"⚠️  Title long ({len(title)} chars), but allowing for creative style.")
        return title

    def title_to_filename(self, title: str) -> str:
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in " _-()"]).rstrip()
        safe_title = safe_title.replace(" ", "_")
        return safe_title[:200]