import os
import time
import json
import subprocess
import sys
import numpy as np  # type: ignore
import whisperx # type: ignore
from faster_whisper import WhisperModel  # type: ignore

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

        # Add debug info about the extracted audio without causing warnings
        audio_info_cmd = [ffmpeg_path, "-i", output_file, "-f", "null", "-"]
        audio_info = subprocess.run(audio_info_cmd, capture_output=True, text=True)
        # Extract useful info from stderr while ignoring the "At least one output file" message
        audio_info_text = "\n".join([line for line in audio_info.stderr.split("\n") 
                                    if "At least one output file" not in line])
        log(f"Extracted audio info: {audio_info_text}")
        
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









def align_with_whisperx(audio_path, segments_list, device="cpu", language_code="en"):
    """Align segments with WhisperX for improved word-level timestamps"""
    if not WHISPERX_AVAILABLE:
        log("WhisperX not available. Skipping alignment.")
        return None
        
    try:
        log(f"Loading WhisperX alignment model for language '{language_code}' on {device}...")
        model_a, metadata = load_align_model(language_code, device)
        
        # Create proper segment format for WhisperX
        # The align function expects a direct list of segment dictionaries
        whisperx_segments = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for segment in segments_list
        ]
        
        log(f"Running WhisperX alignment on {len(whisperx_segments)} segments...")
        
        # Pass the segments list directly to align()
        alignment_result = align(
            whisperx_segments,  # Direct list of segment dictionaries
            model_a,
            metadata,
            audio_path,
            device,
            return_char_alignments=False
        )
        
        # Check if the alignment returned valid results
        if not alignment_result or "segments" not in alignment_result:
            log("WhisperX alignment returned invalid result")
            return None
            
        log(f"WhisperX alignment successful: {len(alignment_result['segments'])} segments")
        
        # Detailed logging of WhisperX results
        for i, segment in enumerate(alignment_result['segments'][:3]):  # Show first 3 segments
            log(f"WhisperX segment {i}: {segment['start']:.2f}s - {segment['end']:.2f}s: '{segment['text']}'")
            if 'words' in segment:
                log(f"  Contains {len(segment['words'])} aligned words")
                for j, word in enumerate(segment['words'][:3]):  # Show first 3 words
                    log(f"    Word {j}: {word['start']:.2f}s - {word['end']:.2f}s: '{word['word']}'")
            else:
                log("  No word-level alignment in this segment")
        
        return alignment_result["segments"]
    except Exception as e:
        log(f"WhisperX alignment failed: {e}")
        import traceback
        log(f"WhisperX traceback: {traceback.format_exc()}")
        return None
    



def fix_overlapping_timestamps(transcriptions, min_duration=0.1):
    """
    Fix overlapping timestamps in transcriptions to ensure smooth subtitle display.
    Each entry in transcriptions should be in format: "start-end: text"
    
    Args:
        transcriptions: List of transcription lines with timestamps
        min_duration: Minimum duration for each word in seconds
        
    Returns:
        List of transcriptions with fixed timestamps
    """
    if not transcriptions:
        return []
        
    # Parse the timestamps and text
    parsed = []
    for line in transcriptions:
        try:
            time_part, text = line.split(':', 1)
            start_str, end_str = time_part.split('-')
            start = float(start_str)
            end = float(end_str)
            parsed.append((start, end, text.strip()))
        except ValueError:
            # Skip lines that don't have the expected format
            continue
    
    # Sort by start time
    parsed.sort(key=lambda x: x[0])
    
    # Fix overlapping timestamps
    fixed = []
    if parsed:
        fixed.append(parsed[0])  # Add the first item
        
        for i in range(1, len(parsed)):
            prev_start, prev_end, prev_text = fixed[-1]
            curr_start, curr_end, curr_text = parsed[i]
            
            # Ensure minimum duration
            if curr_end - curr_start < min_duration:
                curr_end = curr_start + min_duration
            
            # Fix overlap
            if curr_start < prev_end:
                # Set current start time to previous end time
                curr_start = prev_end
                
                # Make sure duration is still reasonable
                if curr_end < curr_start + min_duration:
                    curr_end = curr_start + min_duration
            
            fixed.append((curr_start, curr_end, curr_text))
    
    # Convert back to the original format
    result = []
    for start, end, text in fixed:
        result.append(f"{start:.2f}-{end:.2f}: {text}")
    
    return result



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
            # Using word-level timestamps and no VAD filter for maximum accuracy
            segments, info = model.transcribe(
                audio_path,
                language=lang,
                vad_filter=False,  # Disable VAD filtering to get raw results
                word_timestamps=True,
                beam_size=5,
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
        for i, segment in enumerate(segments_list[:3]):  # Show just the first 3 for brevity
            log_func(f"Segment {i+1}: [{segment.start:.2f}s -> {segment.end:.2f}s] '{segment.text}'")
            if segment.words:
                log_func(f"  Words: {len(segment.words)}")
                for j, word in enumerate(segment.words[:3]):  # Show first 3 words only
                    log_func(f"    Word {j+1}: [{word.start:.2f}s -> {word.end:.2f}s] '{word.word}'")
                if len(segment.words) > 3:
                    log_func(f"    ... and {len(segment.words)-3} more words")
            else:
                log_func("  No word timestamps in this segment")

        # If WhisperX is available, prioritize its alignment
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
                # Pass the language code from Whisper's detection
                aligned_segments = align_with_whisperx(
                    audio_path, 
                    whisperx_segments, 
                    device,
                    language_code=info.language  # Use detected language
                )
                
                if aligned_segments:
                    log_func(f"Using WhisperX aligned segments for better timing for {track_name}")
                    use_whisperx = True
                else:
                    log_func(f"WhisperX alignment failed for {track_name}, using original Whisper timestamps")
                    use_whisperx = False
            except Exception as e:
                log_func(f"WhisperX processing error: {e}")
                use_whisperx = False
        else:
            use_whisperx = False
            if not WHISPERX_AVAILABLE:
                log_func("WhisperX not available. Using standard Whisper timestamps.")
            elif not segments_list:
                log_func("No segments found. Using standard Whisper timestamps.")
            elif not any(segment.words for segment in segments_list):
                log_func("No word-level timestamps found. Using standard Whisper timestamps.")

        # Handle transcription output - focusing on raw timing
        transcriptions = []
        word_count = 0

        if use_whisperx:
            log_func(f"Processing WhisperX aligned words for {track_name}...")
            for segment in aligned_segments:
                if 'words' not in segment:
                    log_func(f"No words in WhisperX segment: {segment['text']}")
                    continue
                    
                for word in segment["words"]:
                    word_count += 1
                    word_text = word["word"]
                    word_start = word["start"]
                    word_end = word["end"]

                    # Only basic filtering for completely unusable items
                    if not word_text.strip():
                        continue

                    if is_mic_track:
                        word_text = word_text.upper()  # Uppercase for mic tracks

                    if include_timecodes:
                        transcriptions.append(f"{word_start:.2f}-{word_end:.2f}: {word_text}")
                    else:
                        transcriptions.append(word_text)
        else:
            log_func(f"Using Whisper's original word timestamps for {track_name}...")
            for segment in segments_list:
                if not segment.words:
                    continue
                    
                for word in segment.words:
                    word_count += 1
                    word_text = word.word
                    word_start = word.start
                    word_end = word.end

                    # Only basic filtering for completely unusable items
                    if not word_text.strip():
                        continue

                    if is_mic_track:
                        word_text = word_text.upper()  # Uppercase for mic tracks

                    if include_timecodes:
                        transcriptions.append(f"{word_start:.2f}-{word_end:.2f}: {word_text}")
                    else:
                        transcriptions.append(word_text)

        log_func(f"Processed {word_count} words for {track_name}.")
        log_func(f"Transcription for {track_name} took {time.time() - start_time:.2f} seconds.")
        log_func(f"ALIGNMENT METHOD FOR {track_name}: {'WhisperX' if use_whisperx else 'Standard Whisper'}")
        
         # Fix overlapping timestamps to prevent subtitle display issues
        if include_timecodes and transcriptions:
            log_func(f"Fixing overlapping timestamps for {track_name}...")
            original_count = len(transcriptions)
            transcriptions = fix_overlapping_timestamps(transcriptions)
            log_func(f"Fixed timestamps: {original_count} → {len(transcriptions)} entries")
            
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
