import os
import re
import subprocess
import json

def get_video_duration(video_file, log):
    """Get the duration of a video file using ffprobe"""
    try:
        command = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            video_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        log(f"Video duration: {duration} seconds")
        return duration
    except Exception as e:
        log(f"Error getting video duration: {e}")
        return 60.0  # Default to 60 seconds if cannot determine

def convert_to_srt(input_text, output_file, video_file, log):
    """Convert text to SRT format with short phrases (max 3 words) per subtitle"""
    log(f"Converting transcription to SRT format with short phrases: {output_file}")
    
    # Get video duration for proper subtitle timing
    video_duration = get_video_duration(video_file, log)
    
    # Debug the input text
    log(f"Input text for SRT conversion: {input_text[:200]}...")  # Show first 200 chars
    
    with open(output_file, 'w', encoding='utf-8') as srt_file:
        lines = input_text.strip().split('\n')
        
        # Counter for subtitle number
        subtitle_count = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if there are timestamps in the line (format: "start-end: text")
            timestamp_match = re.match(r'(\d+\.\d+)-(\d+\.\d+):\s*(.*)', line)
            
            if timestamp_match:
                # Extract timestamps and text
                start_time = float(timestamp_match.group(1))
                end_time = float(timestamp_match.group(2))
                text = timestamp_match.group(3).strip()
                
                # For very short segments (like "ooh"), just display the whole thing
                if len(text.split()) <= 3:
                    # Write SRT entry
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                    srt_file.write(f"{text}\n\n")
                    subtitle_count += 1
                    continue
                
                # For longer segments, split into chunks of max 3 words
                words = text.split()
                
                # Calculate average time per word
                # Speech is typically 2-3 words per second
                total_segment_duration = end_time - start_time
                # Assume actual speaking time is about 70% of the segment duration
                # (accounts for pauses, breaths, etc.)
                effective_speaking_time = total_segment_duration * 0.7
                time_per_word = effective_speaking_time / len(words)
                
                # Process in chunks of maximum 3 words
                for i in range(0, len(words), 3):
                    chunk = words[i:i+3]
                    if not chunk:
                        continue
                    
                    # Calculate estimated time for this chunk
                    chunk_size = len(chunk)
                    chunk_duration = chunk_size * time_per_word
                    # Ensure minimum duration of 0.5 seconds
                    chunk_duration = max(0.5, chunk_duration)
                    
                    # Calculate start time as a proportion of the whole segment
                    # based on word position
                    word_position = i / len(words)
                    chunk_start = start_time + (total_segment_duration * word_position)
                    chunk_end = min(end_time, chunk_start + chunk_duration)
                    
                    # Handle the case where chunk_end might exceed end_time
                    if chunk_end > end_time:
                        chunk_end = end_time
                        
                    # Format times for SRT
                    start_formatted = format_time(chunk_start)
                    end_formatted = format_time(chunk_end)
                    
                    # Write SRT entry
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                    srt_file.write(f"{' '.join(chunk)}\n\n")
                    
                    subtitle_count += 1
            else:
                # No timestamps found, distribute evenly (fallback)
                # This is for user-edited text without timestamps
                words = line.split()
                
                if not words:
                    continue
                
                # Process in chunks of maximum 3 words
                for i in range(0, len(words), 3):
                    chunk = words[i:i+3]
                    if not chunk:
                        continue
                    
                    # Calculate position in video (evenly distributed)
                    # Number of chunks in this line
                    num_chunks = (len(words) + 2) // 3  # Ceiling division by 3
                    
                    # This chunk's index
                    chunk_index = i // 3
                    
                    # Duration for each chunk
                    chunk_duration = video_duration / max(num_chunks, 1)  # Avoid division by zero
                    
                    # Calculate start and end times
                    chunk_start = (chunk_index * chunk_duration)
                    chunk_end = chunk_start + chunk_duration
                    
                    # Format times for SRT
                    start_formatted = format_time(chunk_start)
                    end_formatted = format_time(chunk_end)
                    
                    # Write SRT entry
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_formatted} --> {end_formatted}\n")
                    srt_file.write(f"{' '.join(chunk)}\n\n")
                    
                    subtitle_count += 1
    
    # Debug: read back the file content
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read(500)  # First 500 chars
            log(f"Generated SRT content (first 500 chars):\n{content}")
    except Exception as e:
        log(f"WARNING: Could not read back SRT file: {e}")
            
    log(f"Conversion to {output_file} completed with {subtitle_count-1} subtitle segments\n")

def format_time(seconds):
    """Format time in seconds to SRT format: HH:MM:SS,mmm"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def embed_subtitles(input_video, output_video, subtitles_file, log):
    """Embed subtitles into a video file"""
    log(f"Embedding subtitles from {subtitles_file} into {output_video}\n")
    
    # First verify the subtitle file exists and has content
    if not os.path.exists(subtitles_file):
        log(f"ERROR: Subtitle file does not exist: {subtitles_file}")
        raise FileNotFoundError(f"Subtitle file not found: {subtitles_file}")
    
    file_size = os.path.getsize(subtitles_file)
    if file_size == 0:
        log(f"ERROR: Subtitle file is empty: {subtitles_file}")
        raise ValueError(f"Subtitle file is empty: {subtitles_file}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log(f"Created output directory: {output_dir}")
    
    subtitle_style = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=1,Italic=0,BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=20"
    )
    
    # Adjust the subtitles file path format for ffmpeg
    subtitles_file = subtitles_file.replace('\\', '/')
    if os.name == 'nt':
        subtitles_file = subtitles_file.replace(':', r'\:')
    
    # Special handling for MKV files
    temp_mp4 = None
    try:
        if input_video.lower().endswith('.mkv'):
            log("Input is MKV format. Converting to MP4 first...")
            temp_mp4 = f"{os.path.splitext(output_video)[0]}_temp.mp4"
            convert_cmd = [
                'ffmpeg', 
                '-i', input_video, 
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Convert audio to AAC for better compatibility
                '-strict', 'experimental',
                temp_mp4
            ]
            
            log(f"Running conversion command: {' '.join(convert_cmd)}")
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            log("MKV to MP4 conversion successful")
            
            # Now use the temp MP4 as input
            input_video = temp_mp4
    
        # Now embed the subtitles
        command = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f"subtitles='{subtitles_file}':force_style='{subtitle_style}'",
            '-c:a', 'copy',
            output_video
        ]
        
        log(f"Running embedding command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        log(f"Subtitles embedded into {output_video} successfully\n")
        
    except subprocess.CalledProcessError as e:
        log(f"ERROR: FFmpeg process failed with exit code {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            log(f"Standard output: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            log(f"Standard error: {e.stderr}")
        
        # More detailed error handling
        error_msg = str(e)
        if "No such file or directory" in error_msg:
            log("ERROR: One of the files could not be found")
        elif "Invalid data found when processing input" in error_msg:
            log("ERROR: The input file format is not properly recognized")
        elif "Unable to open" in error_msg and ".srt" in error_msg:
            log("ERROR: Could not open the SRT file - check permissions and path")
            
        log(f"Error embedding subtitles: {e}\n")
        raise RuntimeError(f"Error embedding subtitles: {e}")
    
    finally:
        # Clean up temporary MP4 if it was created
        if temp_mp4 and os.path.exists(temp_mp4):
            try:
                os.remove(temp_mp4)
                log(f"Removed temporary MP4 file: {temp_mp4}")
            except Exception as e:
                log(f"Warning: Could not remove temporary MP4 file: {e}")