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


def intelligent_phrase_segmentation(text, max_chars_per_line=42):
    """Break text into natural phrases with a maximum length"""
    # If text is already short enough, return as is
    if len(text) <= max_chars_per_line:
        return [text]
    
    # Split by natural breaks first
    natural_breaks = []
    for phrase in re.split(r'([,;:.?!])', text):
        if phrase in ',;:.?!':
            # Append punctuation to the previous phrase
            if natural_breaks:
                natural_breaks[-1] += phrase
        elif phrase.strip():
            natural_breaks.append(phrase.strip())
    
    # Now ensure each phrase is within the maximum length
    result = []
    current_line = ""
    
    for phrase in natural_breaks:
        # If adding this phrase would exceed max length and we already have content
        if len(current_line) + len(phrase) + 1 > max_chars_per_line and current_line:
            result.append(current_line)
            current_line = phrase
        # If this single phrase is too long, break it by words
        elif len(phrase) > max_chars_per_line:
            # Add any existing content first
            if current_line:
                result.append(current_line)
                current_line = ""
            
            # Break long phrase by words
            words = phrase.split()
            for word in words:
                if len(current_line) + len(word) + 1 > max_chars_per_line:
                    result.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
        else:
            # Add the phrase with a space if needed
            if current_line:
                current_line += " " + phrase
            else:
                current_line = phrase
    
    # Don't forget the last line
    if current_line:
        result.append(current_line)
    
    return result


def convert_to_srt(input_text, output_file, video_file, log):
    """Convert text to SRT format with intelligent phrase segmentation while preserving timestamps"""
    log(f"Converting transcription to SRT format with improved segmentation: {output_file}")
    video_duration = get_video_duration(video_file, log)
    log(f"Input text for SRT conversion: {input_text[:200]}...")

    with open(output_file, 'w', encoding='utf-8') as srt_file:
        lines = input_text.strip().split('\n')
        subtitle_count = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            timestamp_match = re.match(r'(\d+\.\d+)-(\d+\.\d+):\s*(.*)', line)
            if timestamp_match:
                start_time = float(timestamp_match.group(1))
                end_time = float(timestamp_match.group(2))
                text = timestamp_match.group(3).strip()
                
                # Skip empty segments
                if not text:
                    continue
                
                # Use the exact timestamps provided from transcription
                start_formatted = format_time(start_time)
                end_formatted = format_time(end_time)
                
                # Create the subtitle entry with the original text
                srt_file.write(f"{subtitle_count}\n")
                srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                srt_file.write(f"{text}\n\n")
                subtitle_count += 1
            else:
                # For lines without timestamps (fallback)
                # This is a safety measure for lines that don't have proper timestamps
                text = line.strip()
                if not text:
                    continue
                    
                # Split text into chunks of max 3 words
                words = text.split()
                chunks = []
                for i in range(0, len(words), 3):
                    chunks.append(" ".join(words[i:i+3]))
                
                # Distribute chunks evenly throughout the video duration
                chunk_duration = video_duration / max(len(chunks), 1)
                
                for i, chunk in enumerate(chunks):
                    chunk_start = i * chunk_duration
                    chunk_end = chunk_start + chunk_duration
                    
                    start_formatted = format_time(chunk_start)
                    end_formatted = format_time(chunk_end)
                    
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                    srt_file.write(f"{chunk}\n\n")
                    subtitle_count += 1

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read(500)
            log(f"Generated SRT content (first 500 chars):\n{content}")
    except Exception as e:
        log(f"WARNING: Could not read back SRT file: {e}")

    log(f"Conversion to {output_file} completed with {subtitle_count-1} subtitle segments\n")