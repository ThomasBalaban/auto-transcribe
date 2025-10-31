# core/title_prompt_builder.py
"""
Builds comprehensive prompts for Gemini title generation.
Includes style examples and guidelines based on successful titles.
"""

from typing import Dict, List


class TitlePromptBuilder:
    """Constructs prompts for Gemini API with style guidelines and examples."""

    # Real successful titles from the creator (used as training examples)
    STYLE_EXAMPLES = {
        'reaction_based': [
            "I got jumpscared by that @AngelXGirl97 :(",
            "I was not expecting this (FNaF Jumpscare)",
            "I still have no idea what this sound was (slight jumpscare)",
            "ABSOLUTELY HAMMERED BY BIRD",
            "MAIL CALL (jumpscare)",
        ],
        'declarative': [
            "THATS A WAR CRIME",
            "CHAT IS WILD (Phasmophobia TTS)",
            "THEY PUT ADS IN THIS GAME",
            "WE NEED TO CHECK HIS PITS",
            "THAT'S NOT A TOWER (I deserved that)",
        ],
        'regret_mistake': [
            "I REGRET MY CHOICES (Gross)",
            "Probably shouldn't have covered in a corner",
            "That was... Unfortunate",
            "MOST BRAINROT THING I HAVE EVER DONE",
        ],
        'confusion_wtf': [
            "WHAT IS THE SHELF DOING",
            "MARIO HANGS OUT IN THE GIRLS BATHROOM?!?!",
            "CAN @YTFGS NOT MULTITASK?!?!",
        ],
        'social': [
            "THIS IS WHY WE ARE GETTING A DIVORCE @YTFGS",
            "Chat never disappoints",
            "I AM SO PROUD OF HER",
        ]
    }

    def build_prompt(self, data: Dict) -> str:
        """
        Build the complete Gemini prompt with all context and guidelines.
        
        Args:
            data: Extracted title components from TitleDataExtractor
            
        Returns:
            Complete prompt string for Gemini API
        """
        
        prompt = f"""You are creating a YouTube title for a gaming clip. This is a raw, casual streaming moment.

=== VIDEO INFORMATION ===
Duration: {data['duration']:.1f} seconds
Game/Context: {data['game_context']}
Overall Tone: {data['overall_tone']}
Action Intensity: {data['total_events_count']} major events detected

=== THE CLIMAX (Last 30% of video - THE PUNCHLINE) ===
The clip was cut to END on this moment. This is what the title should focus on.

{self._format_final_quotes_section(data['final_quotes'])}

{self._format_final_visuals_section(data['final_visuals'])}

{self._format_final_events_section(data['final_events'])}

=== TITLE STYLE GUIDELINES ===
Study these REAL examples from this creator to match their voice exactly:

REACTION-BASED (most common - ~40%):
{self._format_examples('reaction_based')}

DECLARATIVE/CAPS (high energy - ~25%):
{self._format_examples('declarative')}

REGRET/MISTAKE (~10%):
{self._format_examples('regret_mistake')}

CONFUSION/WTF (~10%):
{self._format_examples('confusion_wtf')}

KEY PATTERNS YOU MUST FOLLOW:
1. Use parentheses for context: "(game name)" or "(descriptor)" - VERY COMMON
2. Start with "I/We/Chat/They" about 60% of the time (personal voice)
3. Strategic ALL CAPS for emotional peaks, casual lowercase otherwise
4. Be SPECIFIC - reference exact moments from the data above, not generic descriptions
5. 30-60 characters ideal (STRICT LIMIT)
6. Focus on the ENDING/CLIMAX - that's the punchline of the clip
7. Use emotional punctuation (!!!, ?!?!, :( ) when it fits naturally
8. Raw and honest - not polished or overly clickbaity
9. Match the creator's casual, reactive streaming voice

CRITICAL: The title must feel like the creator naturally named it after clipping the moment.

=== YOUR TASK ===
Generate ONE title that:
- Captures the final moment (the punchline that happens at the END)
- Uses the exact quotes, actions, or visuals from the climax data above
- Matches the casual, reactive voice from the examples
- Feels authentic and natural
- Is 30-60 characters (STRICT)

Think: "The best part happened at the end of this clip. What would I naturally call this moment?"

OUTPUT ONLY THE TITLE - NO EXPLANATIONS, ALTERNATIVES, OR MARKDOWN.
Title:"""

        return prompt

    def _format_final_quotes_section(self, quotes: List[Dict]) -> str:
        """Format the final quotes section of the prompt."""
        
        if not quotes:
            return "Final Player Reactions:\n  [No clear dialogue detected in final moments]"
        
        lines = ["Final Player Reactions (exact quotes from last 10 seconds):"]
        
        for quote in quotes:
            timestamp = quote['timestamp']
            text = quote['text']
            markers = quote.get('markers', [])
            
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            lines.append(f"  {timestamp:.1f}s: \"{text}\"{marker_str}")
        
        return "\n".join(lines)

    def _format_final_visuals_section(self, visuals: List[Dict]) -> str:
        """Format the visual descriptions section."""
        
        if not visuals:
            return "What Happened Visually:\n  [No visual analysis available for climax]"
        
        lines = ["What Happened Visually (from video analysis):"]
        
        for visual in visuals[-3:]:  # Last 3 visual analyses
            timestamp = visual['timestamp']
            caption = visual['caption']
            confidence = visual['confidence']
            
            lines.append(f"  {timestamp:.1f}s: {caption} (confidence: {confidence:.0%})")
        
        return "\n".join(lines)

    def _format_final_events_section(self, events: List[Dict]) -> str:
        """Format the audio/timeline events section."""
        
        if not events:
            return "Audio/Action Events:\n  [No major events in climax window]"
        
        lines = ["Audio/Action Events (impacts, sounds, decisions):"]
        
        for event in events[:5]:  # Top 5 events
            timestamp = event['timestamp']
            score = event['title_score']
            
            if event['type'] == 'onomatopoeia':
                word = event['word']
                lines.append(f"  {timestamp:.1f}s: {word} (relevance: {score:.2f})")
            else:
                action = event.get('action', 'action')
                reason = event.get('reason', 'event')
                lines.append(f"  {timestamp:.1f}s: {action} - {reason} (relevance: {score:.2f})")
        
        return "\n".join(lines)

    def _format_examples(self, category: str) -> str:
        """Format example titles for a category."""
        
        examples = self.STYLE_EXAMPLES.get(category, [])
        return "\n".join([f'- "{ex}"' for ex in examples])
