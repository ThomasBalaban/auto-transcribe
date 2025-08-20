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
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video for analysis"""
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"extracted_audio_{random.randint(1000,9999)}.wav")
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '48000',
                '-ac', '1',  # Mono
                audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.log_func(f"✅ Audio extracted: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.log_func(f"Failed to extract audio: {e}")
            raise
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary file safely"""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.log_func(f"🗑️  Cleaned up temp file: {file_path}")
            except Exception as e:
                self.log_func(f"Warning: Could not clean up {file_path}: {e}")