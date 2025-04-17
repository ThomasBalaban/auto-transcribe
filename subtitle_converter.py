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


def convert_to_srt(input_text, output_file, video_file, log):
    """Convert text to SRT format with short phrases (max 3 words) per subtitle"""
    log(f"Converting transcription to SRT format with short phrases: {output_file}")
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
                words = text.split()

                total_segment_duration = end_time - start_time
                effective_speaking_time = total_segment_duration * 0.7
                time_per_word = effective_speaking_time / max(len(words), 1)

                for i in range(0, len(words), 3):
                    chunk = words[i:i+3]
                    chunk_size = len(chunk)
                    chunk_duration = max(0.5, chunk_size * time_per_word)
                    word_position = i / max(len(words), 1)
                    chunk_start = start_time + (total_segment_duration * word_position)
                    chunk_end = min(end_time, chunk_start + chunk_duration)

                    start_formatted = format_time(chunk_start)
                    end_formatted = format_time(chunk_end)

                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                    srt_file.write(f"{' '.join(chunk)}\n\n")
                    subtitle_count += 1
            else:
                words = line.split()
                num_chunks = (len(words) + 2) // 3

                for i in range(0, len(words), 3):
                    chunk = words[i:i+3]
                    chunk_index = i // 3
                    chunk_duration = video_duration / max(num_chunks, 1)
                    chunk_start = chunk_index * chunk_duration
                    chunk_end = chunk_start + chunk_duration

                    start_formatted = format_time(chunk_start)
                    end_formatted = format_time(chunk_end)

                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                    srt_file.write(f"{' '.join(chunk)}\n\n")
                    subtitle_count += 1

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read(500)
            log(f"Generated SRT content (first 500 chars):\n{content}")
    except Exception as e:
        log(f"WARNING: Could not read back SRT file: {e}")

    log(f"Conversion to {output_file} completed with {subtitle_count-1} subtitle segments\n")