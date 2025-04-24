import os
import re

from video_utils import get_video_duration

def format_time(seconds):
    """Format time in seconds to SRT format: HH:MM:SS,mmm"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def convert_to_srt(input_text, output_file, video_file, log, is_mic_track=False):
    """Convert text to SRT format with intelligent phrase segmentation while preserving timestamps"""
    log(f"Converting transcription to SRT format with improved segmentation: {output_file}")
    video_duration = get_video_duration(video_file, log)
    log(f"Input text for SRT conversion: {input_text[:200]}...")

    # Set maximum duration for a subtitle to appear on screen
    MAX_SUBTITLE_DURATION = 5.0  # seconds
    
    # Set minimum duration for subtitle to remain visible
    MIN_SUBTITLE_DURATION = 0.5  # seconds (reduced from 1.0)
    
    # Buffer between subtitles
    BUFFER = 0.05  # 50ms buffer

    # Extract all timestamps and text
    lines = input_text.strip().split('\n')
    subtitle_segments = []

    timestamp_pattern = re.compile(r'(\d+\.\d+)-(\d+\.\d+):\s*(.*)')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        timestamp_match = timestamp_pattern.match(line)
        if timestamp_match:
            start_time = float(timestamp_match.group(1))
            end_time = float(timestamp_match.group(2))
            text = timestamp_match.group(3).strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Enforce maximum duration
            if end_time - start_time > MAX_SUBTITLE_DURATION:
                end_time = start_time + MAX_SUBTITLE_DURATION
            
            # Convert text to uppercase if it's the microphone track
            if is_mic_track:
                text = text.upper()
                
            subtitle_segments.append((start_time, end_time, text))
        else:
            # Non-timestamp lines - will handle later
            log(f"WARNING: Line without timestamp format: {line}")

    # Sort segments by start time
    subtitle_segments.sort(key=lambda x: x[0])
    
    # Fix overlapping timestamps by adjusting end times
    fixed_segments = []
    if subtitle_segments:
        # Process first segment
        fixed_segments.append(subtitle_segments[0])
        
        # Process subsequent segments
        for i in range(1, len(subtitle_segments)):
            current = subtitle_segments[i]
            prev = fixed_segments[-1]
            
            current_start, current_end, current_text = current
            prev_start, prev_end, prev_text = prev
            
            # STRICT RULE: If current starts before previous ends, adjust previous end
            if current_start <= prev_end:
                # Cut the previous segment's end time 
                new_prev_end = current_start - BUFFER
                
                # Make sure the adjusted end is at least at the minimum duration from its start
                if new_prev_end < prev_start + MIN_SUBTITLE_DURATION:
                    # We have a conflict: can't satisfy both minimum duration and no-overlap
                    # PRIORITY: No overlap takes precedence, so we skip minimum duration check
                    # but ensure end time is at least slightly after start time
                    new_prev_end = max(prev_start + 0.1, current_start - BUFFER)
                    log(f"Warning: Segment at {prev_start:.2f} shortened below minimum duration to avoid overlap")
                
                # Update the previous segment
                fixed_segments[-1] = (prev_start, new_prev_end, prev_text)
                log(f"Fixed overlap: Adjusted segment end from {prev_end:.2f} to {new_prev_end:.2f}")
            
            # Add current segment (with possible adjustment if it's too short now)
            if current_end - current_start < MIN_SUBTITLE_DURATION:
                # Only extend if it won't create a new overlap with the next segment
                if i + 1 < len(subtitle_segments):
                    next_start = subtitle_segments[i+1][0]
                    potential_end = current_start + MIN_SUBTITLE_DURATION
                    # If extending would create overlap with next segment, don't extend
                    if potential_end >= next_start:
                        # Keep as is, even if shorter than preferred
                        fixed_segments.append(current)
                        log(f"Warning: Keeping short segment at {current_start:.2f} to avoid creating new overlap")
                    else:
                        # Safe to extend
                        fixed_segments.append((current_start, potential_end, current_text))
                else:
                    # This is the last segment, safe to extend
                    fixed_segments.append((current_start, current_start + MIN_SUBTITLE_DURATION, current_text))
            else:
                # No adjustment needed
                fixed_segments.append(current)

    # Write SRT file
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        for i, (start_time, end_time, text) in enumerate(fixed_segments, 1):
            start_formatted = format_time(start_time)
            end_formatted = format_time(end_time)
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_formatted} --> {end_formatted}\n")
            srt_file.write(f"{text}\n\n")

    log(f"Conversion to {output_file} completed with {len(fixed_segments)} subtitle segments")
    fixed_count = len(subtitle_segments) - len(fixed_segments) + sum(1 for i in range(1, len(subtitle_segments)) 
                           if subtitle_segments[i][0] <= subtitle_segments[i-1][1])
    log(f"Fixed {fixed_count} timing issues")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read(500)
            log(f"Generated SRT content (first 500 chars):\n{content}")
    except Exception as e:
        log(f"WARNING: Could not read back SRT file: {e}")

    return True