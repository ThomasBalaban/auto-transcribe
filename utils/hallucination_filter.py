"""
Hallucination detection and filtering for Whisper transcriptions.
Removes common hallucinated phrases that occur during quiet audio segments.
"""

import os
import numpy as np # type: ignore
import re

# Try importing librosa for audio analysis
try:
    import librosa # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Common Whisper hallucination patterns
HALLUCINATION_BLACKLIST = {
    # YouTube/Creator content
    "thanks for watching",
    "like and subscribe",
    "don't forget to subscribe", 
    "please like and subscribe",
    "see you next time",
    "make sure to hit that notification bell",
    "check out the description",
    "link in the description",
    "smash that like button",
    "hit the subscribe button",
    
    # Music related
    "music",
    "♪ music ♪",
    "background music",
    "♪",
    "♫",
    "♬",
    
    # Generic filler that's commonly hallucinated
    "subtítulos por la comunidad de amara.org",
    "subtitle by",
    "subtitles by",
    
    # Technical artifacts
    "audio",
    "video",
}

# Patterns for partial matching (regex)
HALLUCINATION_PATTERNS = [
    r"^music\s*$",  # Just "music" alone
    r"^♪.*♪$",      # Anything wrapped in music notes
    r"^[a-z]\s+[a-z]\s+[a-z]$",  # Single letter repetitions like "a a a"
    r"thanks.*watching",  # Variations of "thanks for watching"
    r"like.*subscribe",   # Variations of "like and subscribe"
    r"subtítulos.*amara", # Subtitle watermarks
]

QUIET_SEGMENT_BLACKLIST = {
    # Words that are ONLY suspicious in very quiet audio
    # These are common Whisper hallucinations during silence
    "thank",
    "thanks", 
    "you",
    "watching",
    "subscribe",
    "like",
    "please",
    "smash",
    "hit",
    "bell",
    "notification",
    "description",
    "link",
    "see",
    "next",
    "time",
    "make",
    "sure",
    "check",
    "out",
}

def is_quiet_segment(audio_path, start_time, end_time, quiet_threshold=0.02):
    """
    Check if audio segment is very quiet (likely hallucination territory).
    
    Args:
        audio_path (str): Path to the audio file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        quiet_threshold (float): RMS energy threshold for "quiet"
        
    Returns:
        bool: True if segment is quiet, False otherwise
    """
    if not LIBROSA_AVAILABLE:
        return False  # Can't analyze without librosa
    
    try:
        # Load only the specific segment we're interested in
        duration = end_time - start_time
        if duration <= 0:
            return True  # Invalid duration is suspicious
            
        # Load audio segment
        audio, sr = librosa.load(
            audio_path, 
            sr=16000, 
            offset=start_time,
            duration=duration
        )
        
        if len(audio) == 0:
            return True  # Empty audio is quiet
            
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio**2))
        
        return rms_energy < quiet_threshold
        
    except Exception:
        # If we can't analyze the audio, err on the side of caution
        return False

def has_low_confidence(word_data, confidence_threshold=0.5):
    """
    Check if word has low confidence score (WhisperX only).
    
    Args:
        word_data (dict): Word data from WhisperX with potential 'score' field
        confidence_threshold (float): Threshold below which confidence is "low"
        
    Returns:
        bool: True if confidence is low, False otherwise or if no confidence data
    """
    if isinstance(word_data, dict) and "score" in word_data:
        return word_data["score"] < confidence_threshold
    return False

def matches_hallucination_pattern(text):
    """
    Check if text matches known hallucination patterns.
    
    Args:
        text (str): Text to check
        
    Returns:
        tuple: (exact_match, partial_match) - booleans indicating match types
    """
    text_lower = text.lower().strip()
    
    # Check exact matches
    exact_match = text_lower in HALLUCINATION_BLACKLIST
    
    # Check pattern matches
    partial_match = False
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, text_lower):
            partial_match = True
            break
    
    return exact_match, partial_match

def should_filter_word(word_text, word_data, audio_path, start_time, end_time, log_func=None):
    """
    Determine if a word should be filtered as a hallucination.
    Uses multiple criteria to make conservative filtering decisions.
    
    Args:
        word_text (str): The transcribed word/phrase
        word_data (dict): Word data from WhisperX (may contain confidence)
        audio_path (str): Path to audio file for amplitude analysis
        start_time (float): Word start time
        end_time (float): Word end time
        log_func: Logging function
        
    Returns:
        tuple: (should_filter, reason) - bool and string explaining decision
    """
    # Get all our detection criteria
    is_quiet = is_quiet_segment(audio_path, start_time, end_time)
    low_confidence = has_low_confidence(word_data)
    exact_match, partial_match = matches_hallucination_pattern(word_text)
    
    # High confidence filtering (remove immediately)
    if exact_match and is_quiet:
        return True, f"Exact hallucination match in quiet segment: '{word_text}'"
    
    if exact_match and low_confidence:
        return True, f"Exact hallucination match with low confidence: '{word_text}'"
    
    # Check for common hallucination words in very quiet segments
    word_lower = word_text.lower().strip()
    if word_lower in QUIET_SEGMENT_BLACKLIST:
        if is_quiet:
            return True, f"Suspicious word in quiet segment: '{word_text}'"
        # Even if not super quiet, check confidence
        if low_confidence:
            return True, f"Suspicious word with low confidence: '{word_text}'"
    
    # Medium confidence filtering (multiple red flags needed)
    red_flags = sum([is_quiet, low_confidence, partial_match])
    if red_flags >= 2:
        flags = []
        if is_quiet:
            flags.append("quiet")
        if low_confidence:
            flags.append("low_confidence")
        if partial_match:
            flags.append("pattern_match")
        return True, f"Multiple red flags ({', '.join(flags)}): '{word_text}'"
    
    # Don't filter - not enough evidence
    return False, "Insufficient evidence for hallucination"


def filter_hallucinations(transcriptions, audio_path, track_name="", log_func=None):
    """
    Filter hallucinations from a list of transcriptions.
    
    Args:
        transcriptions (list): List of transcription strings in format "start-end: text"
        audio_path (str): Path to audio file for analysis
        track_name (str): Track name for logging
        log_func: Logging function
        
    Returns:
        list: Filtered transcriptions with hallucinations removed
    """
    if not transcriptions or not log_func:
        return transcriptions
    
    if not LIBROSA_AVAILABLE:
        log_func(f"Warning: librosa not available. Hallucination filtering limited for {track_name}")
    
    log_func(f"Checking for hallucinations in {track_name}...")
    
    filtered_transcriptions = []
    filtered_count = 0
    
    for line in transcriptions:
        try:
            # Parse the timestamp format "start-end: text"
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            start_time = float(start_str)
            end_time = float(end_str)
            word_text = text.strip()
            
            # For now, we don't have access to word_data from WhisperX here
            # We'll use a simplified approach
            word_data = {}  # Could be enhanced to pass actual WhisperX data
            
            should_filter, reason = should_filter_word(
                word_text, word_data, audio_path, start_time, end_time, log_func
            )
            
            if should_filter:
                filtered_count += 1
                log_func(f"  Filtered: {reason}")
            else:
                filtered_transcriptions.append(line)
                
        except (ValueError, IndexError) as e:
            # If parsing fails, keep the original line
            log_func(f"  WARNING: Could not parse line for hallucination check: {line}")
            filtered_transcriptions.append(line)
    
    log_func(f"Hallucination filtering complete for {track_name}: {filtered_count} items removed, {len(filtered_transcriptions)} remaining")
    return filtered_transcriptions

def filter_hallucinations_with_whisperx_data(segments, audio_path, track_name="", log_func=None):
    """
    Filter hallucinations directly from WhisperX segments (has confidence data).
    
    Args:
        segments (list): WhisperX aligned segments with word-level data
        audio_path (str): Path to audio file
        track_name (str): Track name for logging
        log_func: Logging function
        
    Returns:
        list: Filtered segments with hallucinated words removed
    """
    if not segments or not log_func:
        return segments
    
    log_func(f"Checking WhisperX segments for hallucinations in {track_name}...")
    
    filtered_segments = []
    total_words_removed = 0
    
    for segment in segments:
        if 'words' not in segment:
            filtered_segments.append(segment)
            continue
            
        filtered_words = []
        
        for word in segment['words']:
            word_text = word.get('word', '').strip()
            start_time = word.get('start', 0)
            end_time = word.get('end', 0)
            
            should_filter, reason = should_filter_word(
                word_text, word, audio_path, start_time, end_time, log_func
            )
            
            if should_filter:
                total_words_removed += 1
                log_func(f"  Filtered: {reason}")
            else:
                filtered_words.append(word)
        
        # Only keep segment if it has remaining words
        if filtered_words:
            segment_copy = segment.copy()
            segment_copy['words'] = filtered_words
            filtered_segments.append(segment_copy)
    
    log_func(f"WhisperX hallucination filtering complete for {track_name}: {total_words_removed} words removed")
    return filtered_segments