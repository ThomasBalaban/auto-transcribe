"""
Timestamp processing utilities for subtitle generation.
Handles duration adjustments, overlap fixing, and timestamp formatting.
"""

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
        # Trim from beginning - keep end time, adjust start time
        new_start = end_time - max_duration
        new_end = end_time
    elif current_duration < min_duration:
        # Extend duration - keep start time, adjust end time
        new_start = start_time
        new_end = start_time + min_duration
    else:
        # Duration is within acceptable range
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
            # Parse the timestamp format "start-end: text"
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            original_start = float(start_str)
            original_end = float(end_str)
            original_duration = original_end - original_start
            
            # Apply duration adjustments
            new_start, new_end = adjust_word_duration(original_start, original_end)
            new_duration = new_end - new_start
            
            # Track if adjustment was made
            if abs(new_start - original_start) > 0.01 or abs(new_end - original_end) > 0.01:
                adjustments_made += 1
                log_func(f"  Adjusted word '{text.strip()}': {original_duration:.2f}s → {new_duration:.2f}s")
            
            # Format back to original string format
            adjusted_line = f"{new_start:.2f}-{new_end:.2f}:{text}"
            adjusted_transcriptions.append(adjusted_line)
            
        except (ValueError, IndexError) as e:
            # If parsing fails, keep the original line
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
            # Skip lines that don't have the expected format
            continue
    
    # Sort by start time
    parsed.sort(key=lambda x: x[0])
    
    # Fix overlapping timestamps
    fixed = []
    if parsed:
        fixed.append(parsed[0])  # Add the first item
        
        for i in range(1, len(parsed)):
            prev_start, prev_end, prev_text = fixed[-1]
            curr_start, curr_end, curr_text = parsed[i]
            
            # Ensure minimum duration
            if curr_end - curr_start < min_duration:
                curr_end = curr_start + min_duration
            
            # Fix overlap
            if curr_start < prev_end:
                # Set current start time to previous end time plus a small gap
                curr_start = prev_end + 0.01
                
                # Make sure duration is still reasonable
                if curr_end < curr_start + min_duration:
                    curr_end = curr_start + min_duration
            
            fixed.append((curr_start, curr_end, curr_text))
    
    # Convert back to the original format
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