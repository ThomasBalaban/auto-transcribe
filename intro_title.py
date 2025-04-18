import os
import subprocess
from subtitle_styles import IntroTitleStyle

def add_intro_title(input_video, output_video, title_text, title_duration=None, log=None):
    """
    Add an intro title overlay to the beginning of a video
    
    Args:
        input_video: Path to input video
        output_video: Path to save the output video
        title_text: Text to display as the intro title
        title_duration: Duration in seconds to show the title (default from style)
        log: Optional logging function
    """
    if log:
        log(f"Adding intro title overlay to video: '{title_text}'")
    
    # Escape single quotes for ffmpeg
    escaped_text = title_text.replace("'", "'\\''")
    
    # Get the filter string from the style
    filter_string = IntroTitleStyle.get_ffmpeg_filter(escaped_text, title_duration)
    
    # Create the ffmpeg command to add the title overlay
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', filter_string,
        '-c:a', 'copy',  # Copy audio
        output_video
    ]
    
    if log:
        log(f"Running ffmpeg command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        if log:
            log(f"Successfully added intro title overlay to video: {output_video}")
        return True
    except Exception as e:
        if log:
            log(f"ERROR adding intro title: {str(e)}")
        return False