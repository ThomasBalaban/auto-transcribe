"""
File Processing Module for Onomatopoeia Detection.
Handles audio extraction and file type detection.
"""

import os
import random
import tempfile
import subprocess
from typing import Optional


class FileProcessor:
    """
    Handles file processing operations for video and audio files.
    """

    def __init__(self, log_func=None):
        self.log_func = log_func or print

    def detect_file_type(self, input_path: str) -> str:
        """
        Detect if file is video, audio, or unsupported.

        Returns:
            'video', 'audio', or 'unsupported'
        """
        file_ext = os.path.splitext(input_path)[1].lower()

        if file_ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            return 'video'
        elif file_ext in ['.wav', '.mp3', '.flac', '.m4a']:
            return 'audio'
        else:
            return 'unsupported'

    def extract_audio_from_video(self, video_path: str, track_index: str = "a:1") -> str:
        """
        Extract a specific audio track from a video file for analysis.

        Args:
            video_path (str): Path to the input video file.
            track_index (str): The FFmpeg stream specifier for the audio track (e.g., "a:1" for the second audio track).

        Returns:
            str: Path to the extracted temporary audio file.
        """
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"extracted_audio_{random.randint(1000,9999)}.wav")

        try:
            self.log_func(f"Extracting audio track '{track_index}' from video...")
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-map', f"0:{track_index}",  # Select the specified audio track
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '48000',
                '-ac', '1',  # Mono
                audio_path
            ]

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log_func(f"✅ Audio extracted to: {audio_path}")
            return audio_path

        except subprocess.CalledProcessError as e:
            self.log_func(f"Failed to extract audio track '{track_index}'. It may not exist.")
            self.log_func(f"FFmpeg stderr: {e.stderr}")
            # As a fallback, try extracting the first audio track
            self.log_func("Attempting to extract the first audio track (a:0) as a fallback...")
            return self.extract_audio_from_video(video_path, "a:0")
        except Exception as e:
            self.log_func(f"An unexpected error occurred during audio extraction: {e}")
            raise


    def cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary file safely"""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.log_func(f"🗑️ Cleaned up temp file: {file_path}")
            except Exception as e:
                self.log_func(f"Warning: Could not clean up {file_path}: {e}")