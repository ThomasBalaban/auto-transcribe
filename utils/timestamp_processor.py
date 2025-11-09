# utils/timestamp_processor.py

"""
Timestamp processing utilities for subtitle generation.
Handles duration adjustments, overlap fixing, and timestamp formatting.
"""

from typing import List, Tuple
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def detect_continuous_vocalization(
    audio_path: str,
    check_time: float,
    lookforward_duration: float = 1.5,
    energy_threshold: float = 0.025,
    log_func = None
) -> Tuple[bool, float]:
    """
    Detects if there's sustained vocalization (screaming, talking) continuing
    past a given timestamp on the mic track.
    
    Args:
        audio_path: Path to the mic track audio file
        check_time: Time (in seconds) to start checking from
        lookforward_duration: How far ahead to check (default 1.5s)
        energy_threshold: RMS energy threshold for "active vocalization"
        log_func: Logging function
        
    Returns:
        Tuple of (is_vocalizing, actual_end_time)
        - is_vocalizing: True if sustained energy detected
        - actual_end_time: When the energy actually drops below threshold
    """
    if not LIBROSA_AVAILABLE:
        if log_func:
            log_func("   ⚠️ librosa not available, skipping vocalization detection")
        return False, check_time
    
    if not audio_path:
        # No audio path provided (e.g., track didn't exist)
        return False, check_time
    
    try:
        # Load audio segment starting from check_time
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            offset=check_time,
            duration=lookforward_duration
        )
        
        if len(audio) == 0:
            if log_func:
                log_func(f"   No audio data at {check_time:.2f}s")
            return False, check_time
        
        # Analyze in small windows (50ms each) to find where energy drops
        window_size = int(0.05 * sr)  # 50ms windows
        hop_size = int(0.01 * sr)     # 10ms hop for precision
        
        sustained_until = check_time
        is_vocalizing = False
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms_energy = np.sqrt(np.mean(window**2))
            
            current_time = check_time + (i / sr)
            
            if rms_energy > energy_threshold:
                sustained_until = current_time + (window_size / sr)
                is_vocalizing = True
                if log_func:
                    log_func(f"   🔊 Energy at {current_time:.2f}s: {rms_energy:.4f} (above threshold {energy_threshold})")
            else:
                # Energy dropped - this is where vocalization actually ends
                if is_vocalizing:
                    if log_func:
                        log_func(f"   🔇 Energy dropped at {current_time:.2f}s: {rms_energy:.4f}")
                    break
        
        if is_vocalizing and log_func:
            duration = sustained_until - check_time
            log_func(f"   🎤 CONTINUOUS VOCALIZATION DETECTED!")
            log_func(f"      Energy sustained from {check_time:.2f}s to {sustained_until:.2f}s")
            log_func(f"      Duration beyond boundary: {duration:.2f}s")
        
        return is_vocalizing, sustained_until
        
    except Exception as e:
        if log_func:
            log_func(f"   ⚠️ Vocalization detection failed: {e}")
        return False, check_time


def adjust_word_duration(start_time, end_time, min_duration=0.3, max_duration=1.5):
    """
    Adjust word timestamps based on duration constraints:
    - If duration > max_duration: trim from beginning (keep end, adjust start)
    - If duration < min_duration: extend duration (keep start, adjust end)
    - Otherwise: keep unchanged
    
    Args:
        start_time (float): Original start time in seconds
        end_time (float): Original end time in seconds
        min_duration (float): Minimum allowed duration (default 0.3s)
        max_duration (float): Maximum allowed duration (default 2.0s)
        
    Returns:
        tuple: (new_start_time, new_end_time)
    """
    current_duration = end_time - start_time
    
    if current_duration > max_duration:
        new_start = end_time - max_duration
        new_end = end_time
    elif current_duration < min_duration:
        new_start = start_time
        new_end = start_time + min_duration
    else:
        new_start = start_time
        new_end = end_time
    
    return new_start, new_end


def apply_duration_adjustments(transcriptions, track_name="", log_func=None):
    """
    Apply duration adjustments to a list of transcriptions with timestamps.
    
    Args:
        transcriptions (list): List of transcription strings in format "start-end: text"
        track_name (str): Name of the track for logging purposes
        log_func: Logging function
        
    Returns:
        list: Adjusted transcriptions
    """
    if not transcriptions or not log_func:
        return transcriptions
    
    log_func(f"Applying duration adjustments for {track_name}...")
    
    adjusted_transcriptions = []
    adjustments_made = 0
    
    for line in transcriptions:
        try:
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            original_start = float(start_str)
            original_end = float(end_str)
            original_duration = original_end - original_start
            
            new_start, new_end = adjust_word_duration(original_start, original_end)
            new_duration = new_end - new_start
            
            if abs(new_start - original_start) > 0.01 or abs(new_end - original_end) > 0.01:
                adjustments_made += 1
                log_func(f"  Adjusted word '{text.strip()}': {original_duration:.2f}s → {new_duration:.2f}s")
            
            adjusted_line = f"{new_start:.2f}-{new_end:.2f}:{text}"
            adjusted_transcriptions.append(adjusted_line)
            
        except (ValueError, IndexError) as e:
            log_func(f"  WARNING: Could not parse timestamp in line: {line}")
            adjusted_transcriptions.append(line)
    
    log_func(f"Duration adjustments complete for {track_name}: {adjustments_made} words adjusted out of {len(transcriptions)}")
    return adjusted_transcriptions


def fix_overlapping_timestamps(transcriptions, min_duration=0.1):
    """
    Fix overlapping timestamps in transcriptions to ensure smooth subtitle display.
    Each entry in transcriptions should be in format: "start-end: text"
    
    Args:
        transcriptions: List of transcription lines with timestamps
        min_duration: Minimum duration for each word in seconds
        
    Returns:
        List of transcriptions with fixed timestamps
    """
    if not transcriptions:
        return []
    
    # Parse the timestamps and text
    parsed = []
    for line in transcriptions:
        try:
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            start = float(start_str)
            end = float(end_str)
            parsed.append((start, end, text.strip()))
        except ValueError:
            continue
    
    parsed.sort(key=lambda x: x[0])
    
    # Fix overlapping timestamps
    fixed = []
    if parsed:
        fixed.append(parsed[0])
        
        for i in range(1, len(parsed)):
            prev_start, prev_end, prev_text = fixed[-1]
            curr_start, curr_end, curr_text = parsed[i]
            
            if curr_end - curr_start < min_duration:
                curr_end = curr_start + min_duration
            
            # Fix overlap by setting current start after previous end
            if curr_start < prev_end:
                curr_start = prev_end + 0.01
                
                if curr_end < curr_start + min_duration:
                    curr_end = curr_start + min_duration
            
            fixed.append((curr_start, curr_end, curr_text))
    
    # Convert back to original format
    result = []
    for start, end, text in fixed:
        result.append(f"{start:.2f}-{end:.2f}: {text}")
    
    return result


def parse_timestamp_line(line):
    """
    Parse a timestamp line in format "start-end: text"
    
    Args:
        line (str): Timestamp line to parse
        
    Returns:
        tuple: (start_time, end_time, text) or None if parsing fails
    """
    try:
        time_part, text = line.split(':', 1)
        start_str, end_str = time_part.split('-')
        start_time = float(start_str)
        end_time = float(end_str)
        return start_time, end_time, text.strip()
    except (ValueError, IndexError):
        return None
    

def format_timestamp_line(start_time, end_time, text):
    """
    Format a timestamp line in the standard format "start-end: text"
    
    Args:
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        text (str): Text content
        
    Returns:
        str: Formatted timestamp line
    """
    return f"{start_time:.2f}-{end_time:.2f}: {text}"


def get_duration_stats(transcriptions):
    """
    Get statistics about word durations in a list of transcriptions.
    
    Args:
        transcriptions (list): List of transcription strings with timestamps
        
    Returns:
        dict: Statistics including min, max, average duration, and count
    """
    durations = []
    
    for line in transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed:
            start_time, end_time, _ = parsed
            duration = end_time - start_time
            durations.append(duration)
    
    if not durations:
        return {"count": 0, "min": 0, "max": 0, "average": 0}
    
    return {
        "count": len(durations),
        "min": min(durations),
        "max": max(durations),
        "average": sum(durations) / len(durations)
    }


def extend_segments_for_dialogue(
    segments_to_keep: List[Tuple[float, float]],
    raw_mic_transcriptions: List[str],
    raw_desktop_transcriptions: List[str], # ✅ NEW
    log_func,
    max_extension_seconds: float = 3.0,
    buffer_seconds: float = 0.5,
    mic_audio_path: str = None,
    desktop_audio_path: str = None          # ✅ NEW
) -> List[Tuple[float, float]]:
    """
    Extends AI-generated trim segments to protect dialogue and continuous vocalizations
    from BOTH mic and desktop tracks.
    
    KEY FEATURE: Scans GAPS between segments for untranscribed screams/yells by analyzing energy on BOTH tracks.
    
    Args:
        segments_to_keep: List of (start, end) tuples from Gemini
        raw_mic_transcriptions: Raw transcript for mic track protection
        raw_desktop_transcriptions: Raw transcript for desktop track protection (friends, NPCs)
        log_func: Logging function
        max_extension_seconds: Maximum seconds to extend for any single protection
        buffer_seconds: Safety buffer added to all boundaries
        mic_audio_path: Path to mic audio file for energy analysis
        desktop_audio_path: Path to desktop audio file for energy analysis
        
    Returns:
        Extended segments that protect all vocal content
    """
    
    if not segments_to_keep:
        log_func("   No segments to protect")
        return []
    
    # ✅ NEW: Combine word timestamps from BOTH tracks for word-level protection
    all_parsed_words = [] 
    for line in raw_mic_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed:
            all_parsed_words.append(parsed + ("mic",)) # Add track identifier
    for line in raw_desktop_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed:
            all_parsed_words.append(parsed + ("desktop",)) # Add track identifier
    
    all_parsed_words.sort(key=lambda x: x[0]) # Sort combined list by start time
    
    segments_to_keep.sort(key=lambda x: x[0])
    
    extended_segments = []
    last_end_time = 0.0

    for segment_idx, (ai_start, ai_end) in enumerate(segments_to_keep):
        
        log_func(f"\n   📍 Processing segment {segment_idx + 1}: {ai_start:.2f}s - {ai_end:.2f}s")
        
        # Apply buffer to start
        new_start = max(0.0, ai_start - buffer_seconds)
        new_end = ai_end
        
        # Prevent overlap with previous segment
        if new_start < last_end_time:
            log_func(f"      ⚠️ Start overlap: {new_start:.2f}s < {last_end_time:.2f}s, adjusting to {last_end_time:.2f}s")
            new_start = last_end_time
        
        # ===== STEP 1: SCAN THE GAP BETWEEN THIS SEGMENT AND THE NEXT =====
        # This catches screams that START in the cut zone
        
        # ✅ NEW: Check both mic and desktop audio for gap vocalizations
        gap_vocalization_end = 0.0
        
        if LIBROSA_AVAILABLE:
            if segment_idx + 1 < len(segments_to_keep):
                next_segment_start = segments_to_keep[segment_idx + 1][0]
                gap_duration = next_segment_start - ai_end
                log_func(f"      🔍 Gap to next segment: {ai_end:.2f}s to {next_segment_start:.2f}s ({gap_duration:.2f}s)")
                
                if gap_duration > 0.1: # Only scan if there's a meaningful gap
                    
                    for track_name, audio_path in [("Mic", mic_audio_path), ("Desktop", desktop_audio_path)]:
                        if not audio_path:
                            continue
                            
                        log_func(f"      🎤 Scanning gap on {track_name} track for untranscribed vocalizations...")
                        try:
                            audio, sr = librosa.load(
                                audio_path,
                                sr=16000,
                                offset=ai_end,
                                duration=min(gap_duration, 10.0)
                            )
                            
                            if len(audio) == 0:
                                continue

                            window_size, hop_size = int(0.05 * sr), int(0.01 * sr)
                            energy_threshold = 0.025
                            vocalization_start_in_gap = None
                            vocalization_end_in_gap = None
                            
                            for i in range(0, len(audio) - window_size, hop_size):
                                window = audio[i:i + window_size]
                                rms_energy = np.sqrt(np.mean(window**2))
                                current_time_in_gap = ai_end + (i / sr)
                                
                                if rms_energy > energy_threshold:
                                    if vocalization_start_in_gap is None:
                                        vocalization_start_in_gap = current_time_in_gap
                                        log_func(f"         🔊 [{track_name}] High energy detected at {current_time_in_gap:.2f}s")
                                    vocalization_end_in_gap = current_time_in_gap + (window_size / sr)
                                elif vocalization_start_in_gap is not None:
                                    break # Energy dropped
                            
                            if vocalization_end_in_gap is not None:
                                log_func(f"      🚨 [{track_name}] UNTRANSCRIBED VOCALIZATION IN GAP!")
                                gap_vocalization_end = max(gap_vocalization_end, vocalization_end_in_gap)
                                
                        except Exception as e:
                            log_func(f"      ⚠️ [{track_name}] Gap scanning failed: {e}")
                    
                    if gap_vocalization_end > 0:
                        extension_needed = gap_vocalization_end - ai_end
                        if extension_needed <= max_extension_seconds:
                            new_end = gap_vocalization_end + buffer_seconds
                            log_func(f"         ✅ Extending segment to {new_end:.2f}s to protect vocalization in gap")
                        else:
                            new_end = ai_end + max_extension_seconds + buffer_seconds
                            log_func(f"         ⚠️ Vocalization too long, capping at {new_end:.2f}s")
                    else:
                        log_func(f"      ✓ No high-energy content found in gap on either track")
            else:
                log_func(f"      ℹ️ Last segment - no gap to scan")
        
        # ===== STEP 2: CHECK FOR CONTINUOUS VOCALIZATION AT SEGMENT END =====
        # This catches when transcribed speech extends beyond its timestamp
        
        # Only run if we didn't already extend from the gap scan
        if new_end == ai_end: 
            if LIBROSA_AVAILABLE:
                log_func(f"      🎤 Checking for continuous vocalization at segment end ({ai_end:.2f}s) on both tracks...")
                
                # Check Mic Track
                mic_is_vocalizing, mic_vocal_end = detect_continuous_vocalization(
                    mic_audio_path, ai_end, 2.0, 0.025, log_func
                )
                
                # Check Desktop Track
                desktop_is_vocalizing, desktop_vocal_end = detect_continuous_vocalization(
                    desktop_audio_path, ai_end, 2.0, 0.025, log_func
                )
                
                # Combine results
                is_vocalizing = mic_is_vocalizing or desktop_is_vocalizing
                actual_vocal_end = max(mic_vocal_end, desktop_vocal_end)
                
                if is_vocalizing:
                    extension_needed = actual_vocal_end - ai_end
                    log_func(f"      🚨 VOCALIZATION EXTENDS BEYOND SEGMENT!")
                    log_func(f"         Gemini wanted to cut at: {ai_end:.2f}s")
                    log_func(f"         Actual vocal end: {actual_vocal_end:.2f}s")
                    
                    if extension_needed <= max_extension_seconds:
                        new_end = actual_vocal_end + buffer_seconds
                        log_func(f"         ✅ Extending to {new_end:.2f}s (vocal_end + buffer)")
                    else:
                        new_end = (ai_end + max_extension_seconds) + buffer_seconds
                        log_func(f"         ⚠️ Extension too long, capping at {new_end:.2f}s")
                else:
                    log_func(f"      ✓ No continuous vocalization at boundary")
                    
                    # ===== STEP 3: CHECK FOR WORD-LEVEL OVERLAPS (FALLBACK) =====
                    # This is the final check if no continuous energy was found
                    if all_parsed_words:
                        found_word_overlap = False
                        for (word_start, word_end, text, track) in all_parsed_words:
                            if word_start <= ai_end < word_end:
                                log_func(f"      📝 [{track}] Word overlap: '{text}' ends at {word_end:.2f}s, cut at {ai_end:.2f}s")
                                
                                extension_needed = word_end - ai_end
                                if extension_needed <= max_extension_seconds:
                                    new_end = word_end + buffer_seconds
                                    log_func(f"         ✅ Extending to {new_end:.2f}s (word_end + buffer)")
                                else:
                                    new_end = (ai_end + max_extension_seconds) + buffer_seconds
                                    log_func(f"         ⚠️ Extension capped at {new_end:.2f}s")
                                
                                found_word_overlap = True
                                break
                        
                        if not found_word_overlap:
                            new_end = ai_end + buffer_seconds
                            log_func(f"      ✓ No word overlaps, adding buffer: {new_end:.2f}s")
                    else:
                        new_end = ai_end + buffer_seconds
                        log_func(f"      ✓ No transcription data, adding buffer: {new_end:.2f}s")
            
            # Fallback if librosa is NOT available (Step 2 and 3 combined)
            else:
                log_func(f"      ⚠️ librosa not available, using word-based protection only")
                if all_parsed_words:
                    found_overlap = False
                    for (word_start, word_end, text, track) in all_parsed_words:
                        if word_start <= ai_end < word_end:
                            extension_needed = word_end - ai_end
                            if extension_needed <= max_extension_seconds:
                                new_end = word_end + buffer_seconds
                            else:
                                new_end = (ai_end + max_extension_seconds) + buffer_seconds
                            found_overlap = True
                            log_func(f"      📝 [{track}] Protected word '{text}': extended to {new_end:.2f}s")
                            break
                    
                    if not found_overlap:
                        new_end = ai_end + buffer_seconds
                else:
                    new_end = ai_end + buffer_seconds

        # Validate segment
        if new_start < new_end:
            extended_segments.append((new_start, new_end))
            last_end_time = new_end
            log_func(f"      ✅ Final segment: {new_start:.2f}s - {new_end:.2f}s (duration: {new_end - new_start:.2f}s)")
        else:
            log_func(f"      ❌ Invalid segment rejected: {new_start:.2f}s - {new_end:.2f}s")

    return extended_segments