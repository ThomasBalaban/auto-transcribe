# title_gen/title_generator.py
"""
AI-powered title generation for gaming clips using Gemini.
Analyzes the video directly with a strong focus on the final 20 seconds.
"""

import json
from typing import Optional, Tuple, List
import os
import time
import google.generativeai as genai # type: ignore
import re
from llm.gemini_vision_analyzer import GeminiVisionAnalyzer
from video_utils import get_video_duration # For getting video duration
from PIL import Image


class TitleGenerator:
    """
    Generates YouTube-style titles with a heavy emphasis on the clip's climax.
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
    ) -> Optional[Tuple[str, str, str]]:
        """
        Main entry point for title generation.
        Returns a tuple containing (title, description, reasoning) or None.
        """
        try:
            self.log_func("\n🎬 Starting climax-focused title generation...")

            if not os.path.exists(shorts_analysis_path):
                self.log_func(f"⚠️ {shorts_analysis_path} not found. Cannot generate title.")
                return None

            with open(shorts_analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            climax_frames = self._extract_climax_frames(video_path)

            prompt = self._build_gemini_prompt(analysis_data)
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

    def _build_gemini_prompt(self, analysis_data: dict) -> str:
        """Builds the comprehensive prompt for Gemini with new, refined rules."""

        successful_titles = [
            f"- \"{item['title']}\" (Views: {item['views']:,}): {item['gemini_analysis']['title_effectiveness_analysis']}"
            for item in analysis_data.get('shorts', [])[:15]
        ]
        prompt = f"""You are an expert YouTube content strategist specializing in viral gaming clips.

        === STYLE ANALYSIS (My Most Successful Videos) ===
        Here is an analysis of my top-performing YouTube Shorts. Your title must match this voice.
        {chr(10).join(successful_titles)}

        === CRITICAL INSTRUCTIONS ===
        1.  **GOAL: GET THE CLICK.** The best title is one that is clickable. This is more important than perfect grammar or a literal description. Use humor, irony, intrigue, or create a curiosity gap to make people want to see the clip.
        2.  **PRIORITIZE PLAYER REACTIONS:** My and my friends' dialogue and reactions are the most important source for a title. On-screen text from the game is less important and should only be used if it's the absolute funniest part of the clip. Player voices > Game text.
        3.  **FOCUS ON THE PUNCHLINE:** The most important part of the clip is the final 15-20 seconds. Your analysis and title MUST focus on what happens in these final moments.
        4.  **DO NOT GUESS THE GAME TITLE:** Only include a game name in parentheses `(Game Name)` if you are 100% certain. If unsure, OMIT the game title.
        5.  **AVOID UNNECESSARY CONTEXT:** Do not mention friends' usernames or other stream-specific context that a general audience wouldn't understand.

        === YOUR TASK ===
        Analyze the provided video clip AND the specific climax frames. Then, respond with a single JSON object with three keys: "video_description", "title", and "reasoning".

        Example Response:
        {{
          "video_description": "The player confidently states they won't get scared, then immediately gets jump-scared by an animatronic and screams.",
          "title": "Famous Last Words",
          "reasoning": "The title uses irony by referencing the player's verbal reaction right before the climax, which is a common pattern in your successful videos. It's short, punchy, and intriguing, which should get clicks. I did not add a game title because it wasn't clearly visible."
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
            response = analyzer.model.generate_content(api_content)
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