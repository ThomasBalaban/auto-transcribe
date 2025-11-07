"""
Timestamp processing utilities for subtitle generation.
Handles duration adjustments, overlap fixing, and timestamp formatting.
"""

from typing import List, Tuple

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

def snap_segments_to_dialogue(
    segments_to_keep: List[Tuple[float, float]],
    mic_transcriptions: List[str],
    log_func,
    min_gap: float = 0.25,
    max_snap_distance: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Adjusts AI-generated trim segments to snap to silent gaps in the dialogue.
    
    Args:
        segments_to_keep: The list of (start, end) tuples from IntelligentTrimmer.
        mic_transcriptions: The list of "start-end: text" strings from the transcriber.
        log_func: Logger.
        min_gap: The minimum silence (in seconds) to be considered a "safe" cut point.
        max_snap_distance: The max seconds to search backward/forward for a gap.
        
    Returns:
        A new list of (start, end) tuples, snapped to dialogue.
    """
    if not mic_transcriptions or not segments_to_keep:
        log_func("   Dialogue snapping skipped: No transcriptions or segments available.")
        return segments_to_keep

    # Parse all word timestamps
    parsed_words = []
    for line in mic_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed:
            parsed_words.append(parsed)
    
    if not parsed_words:
        log_func("   Dialogue snapping skipped: Could not parse any words from transcription.")
        return segments_to_keep

    # Find all "safe" silent gaps
    safe_gaps = []
    
    # Add gap before first word if it doesn't start at 0
    if parsed_words[0][0] > min_gap:
        safe_gaps.append((0.0, parsed_words[0][0]))

    # Find gaps between words
    for i in range(1, len(parsed_words)):
        prev_end = parsed_words[i-1][1]
        curr_start = parsed_words[i][0]
        gap_duration = curr_start - prev_end
        
        if gap_duration >= min_gap:
            safe_gaps.append((prev_end, curr_start))
    
    # Add gap after last word (extends to infinity)
    last_word_end = parsed_words[-1][1]
    safe_gaps.append((last_word_end, last_word_end + 1000.0))

    if not safe_gaps:
        log_func("   Dialogue snapping warning: No safe gaps found. Using original cuts.")
        return segments_to_keep

    # Snap the AI segments to these gaps
    snapped_segments = []
    for (ai_start, ai_end) in segments_to_keep:
        new_start = ai_start
        new_end = ai_end

        # Snap Start Time: Find latest safe cut point before AI start
        possible_starts = [
            g[0] for g in safe_gaps 
            if g[0] <= ai_start and ai_start - g[0] < max_snap_distance
        ]
        if possible_starts:
            new_start = max(possible_starts)
            log_func(f"   Snapped segment start {ai_start:.2f}s -> {new_start:.2f}s")
        else:
            log_func(f"   No preceding gap found for start time {ai_start:.2f}s.")

        # Snap End Time: Find earliest safe cut point after AI end
        possible_ends = [
            g[1] for g in safe_gaps
            if g[1] >= ai_end and g[1] - ai_end < max_snap_distance
        ]
        if possible_ends:
            new_end = min(possible_ends)
            log_func(f"   Snapped segment end {ai_end:.2f}s -> {new_end:.2f}s")
        else:
            log_func(f"   No following gap found for end time {ai_end:.2f}s.")

        # Ensure valid segment
        if new_end > new_start:
            snapped_segments.append((new_start, new_end))
        else:
            log_func(f"   Warning: Snapping created invalid segment. Reverting to original {ai_start:.2f}s - {ai_end:.2f}s")
            snapped_segments.append((ai_start, ai_end))

    return snapped_segments

def extend_segments_for_dialogue(
    segments_to_keep: List[Tuple[float, float]],
    raw_mic_transcriptions: List[str],
    log_func,
    max_extension_seconds: float = 4.0,
    buffer_seconds: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Extends AI-generated trim segments to protect dialogue, with added buffer
    AND prevents segment overlap.
    """
    
    # Parse all word timestamps
    parsed_words = [] 
    for line in raw_mic_transcriptions:
        parsed = parse_timestamp_line(line)
        if parsed:
            parsed_words.append(parsed)
        
    if not parsed_words:
        log_func("   Dialogue extension skipped: No words parsed.")
        # Apply buffer to AI cuts and check for overlaps
        buffered_segments = []
        last_end_time = 0.0
        for (ai_start, ai_end) in segments_to_keep:
            new_start = max(0.0, ai_start - buffer_seconds)
            new_end = ai_end + buffer_seconds
            
            # Check for overlap
            if new_start < last_end_time:
                new_start = last_end_time
            
            buffered_segments.append((new_start, new_end))
            last_end_time = new_end
        return buffered_segments
    
    segments_to_keep.sort(key=lambda x: x[0])
    
    extended_segments = []
    last_end_time = 0.0  # Track the end of the last segment

    for (ai_start, ai_end) in segments_to_keep:
        
        # Apply buffer to AI's start time
        new_start = max(0.0, ai_start - buffer_seconds)
        new_end = ai_end
        
        # Overlap prevention: check if buffered start overlaps with last segment's end
        if new_start < last_end_time:
            log_func(f"   Overlap detected: Start {new_start:.2f}s is before last end {last_end_time:.2f}s. Adjusting.")
            new_start = last_end_time
        
        # Check for dialogue overlaps at the end
        found_overlap = False
        for (word_start, word_end, text) in parsed_words:
            # Check if AI's end cuts off this word
            if word_start <= ai_end < word_end:
                log_func(f"   Dialogue overlap detected! AI end {ai_end:.2f}s cuts off '{text}' (ends at {word_end:.2f}s)")
                
                # Apply extension gate
                extension_needed = word_end - ai_end
                if extension_needed <= max_extension_seconds:
                    new_end = word_end + buffer_seconds
                    log_func(f"   Extending cut to {new_end:.2f}s (word_end + buffer).")
                else:
                    new_end = (ai_end + max_extension_seconds) + buffer_seconds
                    log_func(f"   Extension of {extension_needed:.2f}s is too long. Capping at {new_end:.2f}s (gated + buffer).")
                
                found_overlap = True
                break 
        
        if not found_overlap:
            new_end = ai_end + buffer_seconds
            log_func(f"   AI end {ai_end:.2f}s is in a safe gap. Buffering to {new_end:.2f}s.")

        # Ensure valid segment
        if new_start < new_end:
            extended_segments.append((new_start, new_end))
            last_end_time = new_end
        else:
            log_func(f"   Skipping invalid segment after processing: ({new_start:.2f}s, {new_end:.2f}s)")

    return extended_segments