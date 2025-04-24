import os
import time
import json
import subprocess
import sys
import numpy as np  # type: ignore
from faster_whisper import WhisperModel  # type: ignore

# Try importing WhisperX components with explicit error handling
try:
    from whisperx import load_align_model, align
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

def load_model(model_path="large", device="cpu"):
    log(f"Loading Whisper model {model_path} on {device}...")
    try:
        compute_type = "float32"
        if device == "cuda":
            compute_type = "float16"

        # Create the model - using simpler approach without batch processing
        model = WhisperModel(model_path, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        log(f"Error loading model: {e}")
        raise

def convert_to_audio(input_file, output_file, track_index):
    try:
        try:
            subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ffmpeg_path = "ffmpeg"
        except subprocess.CalledProcessError:
            mac_paths = ["/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/opt/local/bin/ffmpeg"]
            for path in mac_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            else:
                raise FileNotFoundError("ffmpeg not found.")

        log(f"Extracting audio track {track_index}...")
        
        # First, let's try a simple extraction without filters to check if the track exists
        check_cmd = [
            ffmpeg_path,
            "-i", input_file,
            "-map", f"0:{track_index}",
            "-f", "null",
            "-"
        ]
        
        try:
            # Just check if this track exists
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            if check_result.returncode != 0:
                log(f"WARNING: Track {track_index} might not exist: {check_result.stderr}")
                
            # Try to get more info about the track
            info_cmd = [ffmpeg_path, "-i", input_file]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            log(f"File info: {info_result.stderr}")
        except Exception as e:
            log(f"Track check failed: {e}")
        
        # Proceed with regular extraction
        command = [
            ffmpeg_path,
            "-i", input_file,
            "-map", f"0:{track_index}",
            "-ar", "16000",  # Remove noise filtering for now
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_file
        ]
        
        log(f"Running extraction command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"Extraction failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise FileNotFoundError(f"Audio extraction failed: {output_file}")

        # Add debug info about the extracted audio
        audio_info_cmd = [ffmpeg_path, "-i", output_file]
        audio_info = subprocess.run(audio_info_cmd, capture_output=True, text=True)
        log(f"Extracted audio info: {audio_info.stderr}")
        
        log(f"Audio extracted to {output_file} (size: {os.path.getsize(output_file)} bytes)")
        return True
    except Exception as e:
        log(f"Error converting video to audio: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return False

def should_filter_word(word):
    if len(word.strip()) <= 1 and word.strip().lower() not in ['i', 'a']:
        return True
    if word.strip().lower() in ['um', 'uh', 'er', 'ah']:
        return True
    return False

def align_with_whisperx(audio_path, whisperx_segments, device="cpu"):
    """Align segments with WhisperX for improved word-level timestamps"""
    if not WHISPERX_AVAILABLE:
        log("WhisperX not available. Skipping alignment.")
        return None
        
    try:
        log(f"Loading WhisperX alignment model on {device}...")
        model_a = load_align_model("en", device)
        
        log(f"Running WhisperX alignment on {len(whisperx_segments)} segments...")
        result = align(whisperx_segments, model_a, audio_path, device)
        
        # Check if the alignment returned valid results
        if not result or "segments" not in result:
            log("WhisperX alignment returned invalid result")
            return None
            
        log(f"WhisperX alignment successful: {len(result['segments'])} segments")
        return result["segments"]
    except Exception as e:
        log(f"WhisperX alignment failed: {e}")
        import traceback
        log(f"WhisperX traceback: {traceback.format_exc()}")
        return None

def transcribe_audio(model_path, device, audio_path, include_timecodes, log_func, language, track_name=""):
    try:
        start_time = time.time()
        log_func(f"Starting transcription for {audio_path} ({track_name})")
        
        # Check if the audio file exists and has size
        if not os.path.exists(audio_path):
            log_func(f"ERROR: Audio file not found: {audio_path}")
            return [f"0.0-5.0: Audio file not found: {audio_path}"] if include_timecodes else ["Audio file not found"]
            
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            log_func(f"ERROR: Audio file is empty: {audio_path}")
            return [f"0.0-5.0: Audio file is empty"] if include_timecodes else ["Audio file is empty"]
            
        log_func(f"Audio file size: {file_size} bytes")

        # Set the language for transcription
        lang = "en" if language == "English" else language.lower()

        # Load the Whisper model
        model = load_model(model_path=model_path, device=device)

        is_mic_track = "mic" in track_name.lower() or "track 2" in track_name.lower()

        log_func("Transcribing with Whisper...")
        try:
            # Using very relaxed VAD settings to catch more speech
            segments, info = model.transcribe(
                audio_path,
                language=lang,
                vad_filter=True,
                word_timestamps=True,
                beam_size=5,
                vad_parameters={"min_silence_duration_ms": 100},  # More sensitive
                condition_on_previous_text=False
            )
            
            log_func(f"Whisper detected language: {info.language} ({info.language_probability:.2f})")
        except Exception as e:
            log_func(f"Error during transcription: {e}")
            import traceback
            log_func(f"Transcription traceback: {traceback.format_exc()}")
            return [f"0.0-5.0: Transcription error: {str(e)}"] if include_timecodes else [f"Transcription error: {str(e)}"]

        # Convert segments iterator to list
        segments_list = list(segments)
        log_func(f"Found {len(segments_list)} segments")
        
        # Debug information about each segment
        for i, segment in enumerate(segments_list):
            log_func(f"Segment {i+1}: [{segment.start:.2f}s -> {segment.end:.2f}s] '{segment.text}'")
            if segment.words:
                log_func(f"  Words: {len(segment.words)}")
                for j, word in enumerate(segment.words[:5]):  # Show first 5 words only
                    log_func(f"    Word {j+1}: [{word.start:.2f}s -> {word.end:.2f}s] '{word.word}'")
                if len(segment.words) > 5:
                    log_func(f"    ... and {len(segment.words)-5} more words")
            else:
                log_func("  No word timestamps in this segment")

        # If we have segments with words, try to run WhisperX alignment
        if WHISPERX_AVAILABLE and segments_list and any(segment.words for segment in segments_list):
            try:
                # Prepare segments for WhisperX
                whisperx_segments = []
                for segment in segments_list:
                    whisperx_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                
                log_func("Running WhisperX alignment for improved timing...")
                aligned_segments = align_with_whisperx(audio_path, whisperx_segments, device)
                
                if aligned_segments:
                    log_func("Using WhisperX aligned segments for better timing")
                    use_whisperx = True
                else:
                    log_func("WhisperX alignment failed, using original Whisper timestamps")
                    use_whisperx = False
            except Exception as e:
                log_func(f"WhisperX processing error: {e}")
                use_whisperx = False
        else:
            use_whisperx = False

        # Handle transcription output
        transcriptions = []
        word_count = 0

        if use_whisperx:
            log_func("Processing WhisperX aligned words...")
            for segment in aligned_segments:
                for word in segment["words"]:
                    word_count += 1
                    word_text = word["word"]
                    word_start = word["start"]
                    word_end = word["end"]

                    if should_filter_word(word_text):
                        continue

                    if word_end - word_start < 0.3:  # Avoid words that are too short
                        word_end = word_start + 0.3  # Ensure minimum duration

                    if is_mic_track:
                        word_text = word_text.upper()  # Optional: Capitalize for mic tracks

                    if include_timecodes:
                        transcriptions.append(f"{word_start:.2f}-{word_end:.2f}: {word_text}")
                    else:
                        transcriptions.append(word_text)
        else:
            log_func("Using Whisper's original word timestamps...")
            for segment in segments_list:
                for word in segment.words:
                    word_count += 1
                    word_text = word.word
                    word_start = word.start
                    word_end = word.end

                    if should_filter_word(word_text):
                        continue

                    if word_end - word_start < 0.3:  # Avoid words that are too short
                        word_end = word_start + 0.3  # Ensure minimum duration

                    if is_mic_track:
                        word_text = word_text.upper()  # Optional: Capitalize for mic tracks

                    if include_timecodes:
                        transcriptions.append(f"{word_start:.2f}-{word_end:.2f}: {word_text}")
                    else:
                        transcriptions.append(word_text)

        log_func(f"Processed {word_count} words.")
        log_func(f"Transcription took {time.time() - start_time:.2f} seconds.")
        
        return transcriptions

    except Exception as e:
        log_func(f"Error in transcription process: {e}")
        return [f"0.0-5.0: Transcription error: {str(e)}"]

def write_transcriptions_to_file(transcriptions, output_file):
    """Writes transcriptions to a file"""
    try:
        with open(output_file, 'w') as file:
            for transcription in transcriptions:
                file.write(transcription + "\n")
        log(f"Transcriptions written to {output_file}")
    except Exception as e:
        log(f"Error writing transcriptions to file: {e}")
