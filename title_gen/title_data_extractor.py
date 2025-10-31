# core/title_data_extractor.py
"""
Extracts relevant components from video data for title generation.
Focuses on end-weighted analysis (last 30% of video).
"""

from typing import Dict, List, Tuple


class TitleDataExtractor:
    """Extracts and scores components from video data for title generation."""

    def __init__(self, log_func=None):
        self.log_func = log_func or print

    def extract_title_components(
        self,
        duration: float,
        mic_transcriptions: List[str],
        timeline_events: List,
        video_analysis_map: Dict[float, Dict]
    ) -> Dict:
        """
        Extract all relevant components for title generation.
        
        Returns:
            Dictionary with all extracted data organized for prompt building
        """
        
        # Define climax window (last 30% of video)
        climax_start = duration * 0.70
        
        self.log_func(f"🎯 Analyzing climax window: {climax_start:.1f}s - {duration:.1f}s")
        
        # Extract components (no onomatopoeia)
        final_quotes = self._extract_final_quotes(mic_transcriptions, climax_start)
        final_events = self._extract_final_events(timeline_events, climax_start, duration)
        final_visuals = self._extract_final_visuals(video_analysis_map, climax_start)
        game_context = self._detect_game_context(video_analysis_map)
        overall_tone = self._analyze_overall_tone(timeline_events, mic_transcriptions)
        
        return {
            'duration': duration,
            'climax_start': climax_start,
            'final_quotes': final_quotes,
            'final_events': final_events,
            'final_visuals': final_visuals,
            'game_context': game_context,
            'overall_tone': overall_tone,
        }

    def _extract_final_quotes(
        self, 
        transcriptions: List[str], 
        climax_start: float
    ) -> List[Dict]:
        """Extract quotes from the climax window with emotional markers."""
        
        final_quotes = []
        
        for line in transcriptions:
            try:
                # Parse timestamp format: "start-end: text"
                time_part, text = line.split(':', 1)
                start_str, _ = time_part.split('-')
                start_time = float(start_str)
                text = text.strip()
                
                # Only include quotes from climax window
                if start_time >= climax_start:
                    # Analyze emotional markers
                    markers = []
                    
                    if text.isupper():
                        markers.append("YELLING")
                    
                    emotion_words = ['oh', 'what', 'no', 'god', 'hell', 'damn', 'shit']
                    if any(word in text.lower() for word in emotion_words):
                        markers.append("EMOTIONAL")
                    
                    if text.endswith(('!', '?', '!?', '?!')):
                        markers.append("PUNCTUATED")
                    
                    final_quotes.append({
                        'timestamp': start_time,
                        'text': text,
                        'markers': markers
                    })
                    
            except (ValueError, IndexError):
                continue
        
        # Sort by timestamp and return last 10
        final_quotes.sort(key=lambda x: x['timestamp'])
        return final_quotes[-10:]

    def _extract_final_events(
        self,
        timeline_events: List,
        climax_start: float,
        duration: float
    ) -> List[Dict]:
        """Extract and score timeline events from climax window."""
        
        scored_events = []
        
        # Score timeline events only
        for event in timeline_events:
            event_time = event.timestamp
            
            if event_time >= climax_start:
                position_ratio = event_time / duration
                position_weight = position_ratio ** 2
                
                if position_ratio > 0.90:
                    position_weight *= 1.5
                
                # Higher weight for dramatic moments
                confidence = event.confidence
                score = confidence * position_weight
                
                scored_events.append({
                    'type': 'timeline',
                    'action': event.action,
                    'reason': event.reason,
                    'timestamp': event_time,
                    'confidence': confidence,
                    'title_score': score
                })
        
        # Sort by score and return top 5
        scored_events.sort(key=lambda x: x['title_score'], reverse=True)
        
        self.log_func(f"📊 Climax events: {len(scored_events)} timeline decisions")
        
        return scored_events[:5]

    def _extract_final_visuals(
        self,
        video_analysis_map: Dict[float, Dict],
        climax_start: float
    ) -> List[Dict]:
        """Extract visual descriptions from climax window."""
        
        final_visuals = []
        
        for timestamp, analysis in video_analysis_map.items():
            if timestamp >= climax_start:
                final_visuals.append({
                    'timestamp': timestamp,
                    'caption': analysis.get('video_caption', 'N/A'),
                    'confidence': analysis.get('confidence', 0),
                    'scene_context': analysis.get('scene_context', set())
                })
        
        # Sort by timestamp
        final_visuals.sort(key=lambda x: x['timestamp'])
        return final_visuals

    def _detect_game_context(self, video_analysis_map: Dict[float, Dict]) -> str:
        """Try to detect the game being played from visual analysis."""
        
        # Collect all captions
        all_captions = ' '.join([
            analysis.get('video_caption', '').lower()
            for analysis in video_analysis_map.values()
        ])
        
        # Common game keywords
        game_keywords = {
            'fnaf': 'FNAF',
            'five nights': 'FNAF',
            'freddy': 'FNAF',
            'lethal company': 'Lethal Company',
            'phasmophobia': 'Phasmophobia',
            'ghost hunting': 'Phasmophobia',
            'metal gear': 'Metal Gear Solid',
            'it takes two': 'It Takes Two',
            'minecraft': 'Minecraft',
            'roblox': 'Roblox',
            'fortnite': 'Fortnite',
            'valorant': 'Valorant',
            'apex': 'Apex Legends',
        }
        
        for keyword, game_name in game_keywords.items():
            if keyword in all_captions:
                self.log_func(f"🎮 Detected game: {game_name}")
                return game_name
        
        return "Unknown Game"

    def _analyze_overall_tone(
        self,
        timeline_events: List,
        mic_transcriptions: List[str]
    ) -> str:
        """Determine overall video tone (funny/intense/chaotic)."""
        
        # Count different event types
        wild_content_count = 0
        dramatic_count = 0
        
        for event in timeline_events:
            reason = event.reason
            if 'wild_content' in reason or 'awkward' in reason:
                wild_content_count += 1
            elif 'dramatic' in reason:
                dramatic_count += 1
        
        # Count profanity/strong reactions in transcriptions
        strong_reaction_count = 0
        for line in mic_transcriptions:
            text_lower = line.lower()
            if any(word in text_lower for word in ['shit', 'fuck', 'damn', 'hell', 'oh my god', 'what the']):
                strong_reaction_count += 1
        
        # Determine tone
        if wild_content_count > dramatic_count and strong_reaction_count > 5:
            tone = "funny/chaotic"
        elif dramatic_count > 5:
            tone = "intense/action-heavy"
        elif wild_content_count > 0:
            tone = "mixed/comedic"
        else:
            tone = "casual/standard"
        
        self.log_func(f"🎭 Detected tone: {tone}")
        return tone