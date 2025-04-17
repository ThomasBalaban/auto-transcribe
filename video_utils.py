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