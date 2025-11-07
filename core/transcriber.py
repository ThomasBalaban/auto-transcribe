# core/transcriber.py

import os
import time
import json
import subprocess
import sys
import numpy as np  # type: ignore
import whisperx # type: ignore
from faster_whisper import WhisperModel  # type: ignore
from utils.timestamp_processor import apply_duration_adjustments, fix_overlapping_timestamps
from utils.hallucination_filter import filter_hallucinations, filter_hallucinations_with_whisperx_data
from typing import List, Tuple

# Try importing WhisperX components with explicit error handling
try:
    from whisperx import load_align_model, align # type: ignore
    WHISPERX_AVAILABLE = True
except ImportError as e:
    print(f"WhisperX import error: {e}")
    WHISPERX_AVAILABLE = False

log_box = None

def set_log_box(log_widget):
    global log_box
    log_box = log_widget

def log(message):
    if log_box:
        log_box.insert("end", message + "\n")
        log_box.see("end")
    else:
        print(message.encode('utf-8', errors='replace').decode('utf-8'))

def load_model(model_path, device, log_func):
    log_func(f"Loading Whisper model {model_path} on {device}...")
    try:
        compute_type = "float32"
        if device == "cuda":
            compute_type = "float16"

        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        log_func(f"Error loading model: {e}")
        raise

def convert_to_audio(input_file, output_file, track_index, log_func):
    try:
        # Find ffmpeg path
        ffmpeg_path = "ffmpeg"
        try:
            # Use subprocess.POST for stdout/stderr to hide output unless error
            subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            mac_paths = ["/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/opt/local/bin/ffmpeg"]
            for path in mac_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            else:
                raise FileNotFoundError("ffmpeg not found.")

        log_func(f"Extracting audio track {track_index}...")

        command = [
            ffmpeg_path,
            "-y",
            "-i", input_file,
            "-map", f"0:{track_index}",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_file
        ]

        log_func(f"Running extraction command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            log_func(f"Extraction failed for track {track_index}. FFmpeg stderr:")
            for line in result.stderr.splitlines():
                log_func(f"  {line}")
            return False

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            log_func(f"Audio extraction failed to create a valid file for track {track_index}: {output_file}")
            return False

        log_func(f"Audio extracted to {output_file} (size: {os.path.getsize(output_file)} bytes)")
        return True

    except Exception as e:
        log_func(f"Error converting video to audio: {e}")
        import traceback
        log_func(f"Traceback: {traceback.format_exc()}")
        return False
    
def align_with_whisperx(audio_path, segments_list, device, language_code, log_func):
    if not WHISPERX_AVAILABLE:
        log_func("WhisperX not available. Skipping alignment.")
        return None
    try:
        log_func(f"Loading WhisperX alignment model for language '{language_code}' on {device}...")
        model_a, metadata = load_align_model(language_code=language_code, device=device)
        whisperx_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_list]
        log_func(f"Running WhisperX alignment on {len(whisperx_segments)} segments...")
        alignment_result = align(whisperx_segments, model_a, metadata, audio_path, device, return_char_alignments=False)
        return alignment_result["segments"]
    except Exception as e:
        log_func(f"WhisperX alignment failed: {e}")
        return None

def transcribe_audio(model_path, device, audio_path, include_timecodes, log_func, language, track_name="") -> Tuple[List[str], List[str]]:
    try:
        start_time = time.time()
        log_func(f"Starting transcription for {audio_path} ({track_name})")

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            log_func(f"ERROR: Audio file is invalid: {audio_path}")
            return [], []

        model = load_model(model_path=model_path, device=device, log_func=log_func)
        is_mic_track = "mic" in track_name.lower() or "track 2" in track_name.lower()

        log_func("Transcribing with Whisper...")
        segments, info = model.transcribe(
            audio_path,
            language=language.lower() if language != "English" else "en",
            word_timestamps=True,
            beam_size=5
        )

        segments_list = list(segments)
        log_func(f"Found {len(segments_list)} segments.")

        use_whisperx = False
        if WHISPERX_AVAILABLE and segments_list:
            aligned_segments = align_with_whisperx(audio_path, segments_list, device, info.language, log_func)
            if aligned_segments:
                log_func(f"Using WhisperX aligned segments for {track_name}")
                aligned_segments = filter_hallucinations_with_whisperx_data(aligned_segments, audio_path, track_name, log_func)
                use_whisperx = True
            else:
                log_func(f"WhisperX alignment failed, using original timestamps for {track_name}")
        
        transcriptions = []
        if use_whisperx:
            for segment in aligned_segments:
                if 'words' not in segment: continue
                for word in segment["words"]:
                    word_text = word["word"].upper() if is_mic_track else word["word"]
                    transcriptions.append(f"{word['start']:.2f}-{word['end']:.2f}: {word_text}")
        else:
            for segment in segments_list:
                if not segment.words: continue
                for word in segment.words:
                    word_text = word.word.upper() if is_mic_track else word.word
                    transcriptions.append(f"{word.start:.2f}-{word.end:.2f}: {word_text}")

        log_func(f"Processed {len(transcriptions)} words for {track_name}.")
        
        raw_transcriptions_for_trimming = []
        adjusted_transcriptions_for_subtitles = []

        if include_timecodes and transcriptions:
             if not use_whisperx:
                  transcriptions = filter_hallucinations(transcriptions, audio_path, track_name, log_func)
             
             list_for_trimming = list(transcriptions)
             list_for_subs = list(transcriptions)

             raw_transcriptions_for_trimming = fix_overlapping_timestamps(list_for_trimming)
             
             adjusted_transcriptions_for_subtitles = apply_duration_adjustments(list_for_subs, track_name, log_func)
             adjusted_transcriptions_for_subtitles = fix_overlapping_timestamps(adjusted_transcriptions_for_subtitles)

        else:
            raw_transcriptions_for_trimming = list(transcriptions)
            adjusted_transcriptions_for_subtitles = list(transcriptions)

        return raw_transcriptions_for_trimming, adjusted_transcriptions_for_subtitles

    except Exception as e:
        log_func(f"Error in transcription process: {e}")
        import traceback
        log_func(f"Transcription traceback: {traceback.format_exc()}")
        error_list = [f"0.0-5.0: Transcription error: {str(e)}"]
        return error_list, error_list