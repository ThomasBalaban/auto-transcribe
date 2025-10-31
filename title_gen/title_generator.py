# core/title_generator.py
"""
AI-powered title generation for gaming clips using Gemini.
Focuses on end-weighted analysis to capture the climax moment.
"""

from typing import Dict, List, Optional
import os


class TitleGenerator:
    """
    Generates YouTube-style titles for gaming clips using end-weighted analysis.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print

    def generate_title(
        self,
        video_duration: float,
        mic_transcriptions: List[str],
        onomatopoeia_events: List[Dict],
        timeline_events: List,
        video_analysis_map: Dict[float, Dict],
    ) -> Optional[str]:
        """
        Main entry point for title generation.
        
        Args:
            video_duration: Total video duration in seconds
            mic_transcriptions: List of transcription lines with timestamps
            onomatopoeia_events: List of detected onomatopoeia events
            timeline_events: AI Director timeline decisions
            video_analysis_map: Map of timestamps to video analysis data
            
        Returns:
            Generated title string or None if generation fails
        """
        try:
            self.log_func("\n🎬 Starting title generation...")
            
            # Step 1: Prepare all data
            title_data = self._prepare_title_data(
                video_duration,
                mic_transcriptions,
                onomatopoeia_events,
                timeline_events,
                video_analysis_map
            )
            
            # Step 2: Build the Gemini prompt
            prompt = self._build_gemini_prompt(title_data)
            
            # Step 3: Call Gemini API
            title = self._call_gemini_api(prompt)
            
            if title:
                self.log_func(f"✨ Generated Title: {title}")
                self.log_func(f"   Length: {len(title)} characters")
                return title
            else:
                self.log_func("⚠️  Title generation returned empty result")
                return None
                
        except Exception as e:
            self.log_func(f"❌ Title generation failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return None

    def _prepare_title_data(
        self,
        duration: float,
        mic_transcriptions: List[str],
        onomatopoeia_events: List[Dict],
        timeline_events: List,
        video_analysis_map: Dict[float, Dict]
    ) -> Dict:
        """Extract and package all relevant data for title generation."""
        
        from title_gen.title_data_extractor import TitleDataExtractor
        
        extractor = TitleDataExtractor(log_func=self.log_func)
        
        data = extractor.extract_title_components(
            duration=duration,
            mic_transcriptions=mic_transcriptions,
            onomatopoeia_events=onomatopoeia_events,
            timeline_events=timeline_events,
            video_analysis_map=video_analysis_map
        )
        
        return data

    def _build_gemini_prompt(self, data: Dict) -> str:
        """Build the comprehensive prompt for Gemini."""
        
        from title_gen.title_prompt_builder import TitlePromptBuilder
        
        builder = TitlePromptBuilder()
        prompt = builder.build_prompt(data)
        
        # Log prompt for debugging
        self.log_func("\n" + "="*60)
        self.log_func("📝 TITLE GENERATION PROMPT")
        self.log_func("="*60)
        self.log_func(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        self.log_func("="*60 + "\n")
        
        return prompt

    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API with the constructed prompt."""
        
        try:
            from llm.gemini_text_generator import GeminiTextGenerator
            
            gemini = GeminiTextGenerator(log_func=self.log_func)
            
            # Use the model directly for title generation
            response = gemini.model.generate_content(
                prompt,
                safety_settings=gemini.safety_settings
            )
            
            # Extract and clean the title
            title = response.text.strip()
            title = self._clean_title(title)
            
            return title
            
        except Exception as e:
            self.log_func(f"💥 Gemini API call failed: {e}")
            return None

    def _clean_title(self, title: str) -> str:
        """Clean up Gemini's output to ensure clean title."""
        
        # Remove markdown formatting
        title = title.replace('**', '').replace('*', '')
        
        # Take first line only (in case Gemini adds explanation)
        title = title.split('\n')[0]
        
        # Remove common prefixes Gemini might add
        prefixes_to_remove = [
            "Title: ",
            "title: ",
            "Title:",
            "title:",
        ]
        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Ensure reasonable length (YouTube recommends 60 chars max)
        if len(title) > 70:
            self.log_func(f"⚠️  Title too long ({len(title)} chars), truncating...")
            title = title[:67] + "..."
        
        return title

    def title_to_filename(self, title: str) -> str:
        """
        Convert a title to a filesystem-safe filename.
        Replaces special characters with words (Mac-compatible).
        
        Examples:
            "WHAT THE HELL?!" -> "WHAT_THE_HELL_question_mark_exclamation"
            "I got jumpscared :(" -> "I_got_jumpscared_sad_face"
        """
        
        # Special character to word mapping
        char_map = {
            '?': '_question_mark',
            '!': '_exclamation',
            ':': '_colon',
            ';': '_semicolon',
            '@': '_at',
            '#': '_hashtag',
            '$': '_dollar',
            '%': '_percent',
            '&': '_and',
            '*': '_asterisk',
            '+': '_plus',
            '=': '_equals',
            '/': '_slash',
            '\\': '_backslash',
            '|': '_pipe',
            '<': '_less_than',
            '>': '_greater_than',
            '"': '_quote',
            "'": '_apostrophe',
            '(': '_open_paren',
            ')': '_close_paren',
            '[': '_open_bracket',
            ']': '_close_bracket',
            '{': '_open_brace',
            '}': '_close_brace',
            '~': '_tilde',
            '`': '_backtick',
            '^': '_caret',
        }
        
        # Emoticon replacements (common in your titles)
        emoticon_map = {
            ':)': '_happy_face',
            ':(': '_sad_face',
            ':D': '_big_smile',
            ';)': '_wink',
            ':P': '_tongue_out',
            ':/': '_unsure',
            ':O': '_shocked',
            'xD': '_laughing',
            '^^': '_happy',
        }
        
        filename = title
        
        # Replace emoticons first (before individual chars)
        for emoticon, replacement in emoticon_map.items():
            filename = filename.replace(emoticon, replacement)
        
        # Replace special characters with words
        for char, replacement in char_map.items():
            filename = filename.replace(char, replacement)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Replace multiple underscores with single
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        # Mac/Unix filesystems have 255 char limit
        # Leave room for .mp4 extension (4 chars)
        if len(filename) > 251:
            self.log_func(f"⚠️  Filename too long ({len(filename)} chars), truncating to 251...")
            filename = filename[:251]
            # Remove trailing underscore if truncation created one
            filename = filename.rstrip('_')
        
        self.log_func(f"📝 Converted title to filename: {filename}.mp4")
        return filename