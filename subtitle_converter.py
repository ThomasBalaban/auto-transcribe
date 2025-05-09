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
    """Convert text to SRT format preserving original timestamps with fixes for overlaps"""
    log(f"Converting transcription to SRT format with original timing: {output_file}")
    video_duration = get_video_duration(video_file, log)
    log(f"Input text for SRT conversion: {input_text[:200]}...")

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
                
            # Convert text to uppercase if it's the microphone track
            if is_mic_track:
                text = text.upper()
                
            subtitle_segments.append((start_time, end_time, text))
        else:
            # Non-timestamp lines
            log(f"WARNING: Line without timestamp format: {line}")

    # Sort segments by start time
    subtitle_segments.sort(key=lambda x: x[0])
    
    # Fix overlapping segments
    fixed_segments = []
    if subtitle_segments:
        fixed_segments.append(subtitle_segments[0])  # Add the first segment
        
        for i in range(1, len(subtitle_segments)):
            prev_start, prev_end, prev_text = fixed_segments[-1]
            curr_start, curr_end, curr_text = subtitle_segments[i]
            
            # Ensure minimum duration (100ms)
            min_duration = 0.1
            if curr_end - curr_start < min_duration:
                curr_end = curr_start + min_duration
            
            # Fix overlap
            if curr_start < prev_end:
                # Set current start time to previous end time plus small buffer
                curr_start = prev_end + 0.01  # Add 10ms buffer
                
                # Make sure duration is still reasonable
                if curr_end < curr_start + min_duration:
                    curr_end = curr_start + min_duration
            
            fixed_segments.append((curr_start, curr_end, curr_text))
    
    # Write SRT file without any timing adjustments
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        for i, (start_time, end_time, text) in enumerate(fixed_segments, 1):
            start_formatted = format_time(start_time)
            end_formatted = format_time(end_time)
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_formatted} --> {end_formatted}\n")
            srt_file.write(f"{text}\n\n")

    log(f"Conversion to {output_file} completed with {len(fixed_segments)} subtitle segments")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read(500)
            log(f"Generated SRT content (first 500 chars):\n{content}")
    except Exception as e:
        log(f"WARNING: Could not read back SRT file: {e}")

    return True
