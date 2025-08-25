import os
import re
from video_utils import get_video_duration
from animations.renderer import ASSRenderer
from animations.position_calculator import PositionCalculator
from animations.animation_types import AnimationType
from animations.utils import AnimationConstants
from subtitle_styles import MicrophoneStyle, AnimatedMicrophoneStyle

def format_time(seconds):
    """Format time in seconds to SRT format: HH:MM:SS,mmm"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def convert_to_srt(input_text, output_file, video_file, log, is_mic_track=False):
    """
    Converts transcription to either an animated ASS file (for mic track)
    or a standard SRT file (for other tracks).
    """
    if is_mic_track:
        # --- Microphone Track: Generate Animated ASS ---
        output_file_ass = output_file.replace(".srt", ".ass")
        log(f"Converting microphone transcription to animated ASS: {output_file_ass}")
        lines = input_text.strip().split('\n')
        renderer = ASSRenderer()
        position_calculator = PositionCalculator()
        dialogue_lines = []

        timestamp_pattern = re.compile(r'(\d+\.\d+)-(\d+\.\d+):\s*(.*)')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            timestamp_match = timestamp_pattern.match(line)
            if timestamp_match:
                start_time = float(timestamp_match.group(1))
                end_time = float(timestamp_match.group(2))
                text = timestamp_match.group(3).strip().upper()
                duration = end_time - start_time
                
                # The animation calculator still needs dummy x/y values, but they won't be used for positioning.
                dummy_x, dummy_y = 0, 0

                positions = position_calculator.calculate_animation_positions(
                    AnimationType.MIC_POP_SHRINK,
                    dummy_x,
                    dummy_y,
                    AnimatedMicrophoneStyle.FONT_SIZE,
                    duration=duration
                )

                num_frames = len(positions)
                frame_duration = duration / num_frames if num_frames > 0 else AnimationConstants.FRAME_DURATION

                for frame, position_data in enumerate(positions):
                    # We only need the font_size from the animation data now
                    _, _, _, frame_font_size, _ = position_data
                    
                    frame_start = start_time + (frame * frame_duration)
                    frame_end = frame_start + frame_duration
                    if frame_end > end_time:
                        frame_end = end_time
                    if frame_start >= end_time:
                        break

                    # Call the new renderer function that doesn't use \pos
                    dialogue_line = renderer.create_mic_dialogue_line(
                        frame_start, frame_end, text, frame_font_size, style=AnimatedMicrophoneStyle.STYLE_NAME
                    )
                    dialogue_lines.append(dialogue_line)

        ass_header = renderer.create_ass_header()
        # Add the animated mic style to the header
        ass_header = ass_header.replace(
            "[V4+ Styles]",
            f"[V4+ Styles]\n{AnimatedMicrophoneStyle.get_ass_style_string()}"
        )

        ass_content = [ass_header]
        ass_content.extend(dialogue_lines)

        with open(output_file_ass, 'w', encoding='utf-8') as f:
            f.write("\n".join(ass_content))
        log(f"Animated microphone ASS file created: {output_file_ass}")
        return True
    else:
        # --- Other Tracks: Generate SRT ---
        log(f"Converting transcription to SRT format with original timing: {output_file}")
        video_duration = get_video_duration(video_file, log)
        log(f"Input text for SRT conversion: {input_text[:200]}...")

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
                if not text:
                    continue
                subtitle_segments.append((start_time, end_time, text))
            else:
                log(f"WARNING: Line without timestamp format: {line}")

        subtitle_segments.sort(key=lambda x: x[0])
        fixed_segments = []
        if subtitle_segments:
            fixed_segments.append(subtitle_segments[0])
            for i in range(1, len(subtitle_segments)):
                prev_start, prev_end, prev_text = fixed_segments[-1]
                curr_start, curr_end, curr_text = subtitle_segments[i]
                min_duration = 0.1
                if curr_end - curr_start < min_duration:
                    curr_end = curr_start + min_duration
                if curr_start < prev_end:
                    curr_start = prev_end + 0.01
                    if curr_end < curr_start + min_duration:
                        curr_end = curr_start + min_duration
                fixed_segments.append((curr_start, curr_end, curr_text))

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