import os
import time
import json
import wave
import subprocess
import sys

# Set environment variable FIRST
os.environ["VOSK_SILENT"] = "1"  

# Now import Vosk 
from vosk import Model, KaldiRecognizer # type: ignore

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

def load_model(model_path="local_modals/vosk-model-en-us-0.42-gigaspeech"):
    log(f"Loading Vosk model from {model_path}...")
    try:
        # On Mac, ensure path uses correct format
        model_path = os.path.expanduser(model_path)
        return Model(model_path)
    except Exception as e:
        log(f"Error loading model: {e}")
        log(f"Make sure the model path is correct and the model files exist")
        raise

def convert_to_audio(input_file, output_file, track_index):
    """Convert video to audio, extracting specific track by index"""
    try:
        # Get available ffmpeg path
        try:
            subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ffmpeg_path = "ffmpeg"
        except subprocess.CalledProcessError:
            # If ffmpeg is not in PATH, try with full path for Mac
            mac_ffmpeg_paths = [
                "/usr/local/bin/ffmpeg",
                "/opt/homebrew/bin/ffmpeg",
                "/opt/local/bin/ffmpeg"
            ]
            
            for path in mac_ffmpeg_paths:
                if os.path.exists(path):
                    ffmpeg_path = path
                    break
            else:
                raise FileNotFoundError("ffmpeg not found. Please install ffmpeg or provide the full path.")
        
        log(f"Extracting audio track {track_index} from video...")
        
        # Convert to mono 16kHz WAV which is optimal for Vosk
        # Specify the audio track using -map option
        command = [
            ffmpeg_path, 
            "-i", input_file,
            "-map", f"0:{track_index}",  # Select specific track by index
            "-ar", "16000",              # 16kHz sample rate
            "-ac", "1",                  # mono audio
            "-c:a", "pcm_s16le",         # PCM 16-bit format
            output_file
        ]
        
        log(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"Error in ffmpeg: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        
        # Verify the output file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            log(f"Successfully extracted audio track {track_index} to {output_file}")
        else:
            log(f"Failed to extract audio: output file is missing or empty")
            raise FileNotFoundError(f"Audio extraction failed: {output_file}")
            
    except subprocess.CalledProcessError as e:
        log(f"Error converting video to audio: {e}")
        if hasattr(e, 'stderr'):
            log(f"ffmpeg error: {e.stderr}")
        raise
    except FileNotFoundError as e:
        log(f"Error: {e}")
        raise

def filter_low_confidence_words(words, min_confidence=0.75):
    """Filter out words with confidence below the threshold"""
    if not words:
        return []
    
    filtered_words = [word for word in words if word.get('conf', 0) >= min_confidence]
    return filtered_words

def create_merged_text(filtered_words):
    """Create a clean text from filtered words"""
    if not filtered_words:
        return ""
    
    return " ".join(word.get('word', '') for word in filtered_words)

def transcribe_audio(model_path, device, audio_path, include_timecodes, log_func, language, track_name=""):
    try:
        start_time = time.time()
        log_func(f"Starting transcription for {audio_path} ({track_name})")
        
        # Load the model
        model = load_model(model_path)
        
        # Open the audio file
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            log_func("Audio file must be WAV format mono PCM.")
            return []
            
        # Create recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)  # Enable word timestamps
        rec.SetPartialWords(True)  # Enable partial results with word timing

        transcriptions = []
        results = []
        
        # Process audio chunks
        while True:
            data = wf.readframes(4000)  # Read audio in chunks
            if len(data) == 0:
                break
                
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "result" in result:
                    results.append(result)
                
        # Get final result
        final_result = json.loads(rec.FinalResult())
        if "result" in final_result:
            results.append(final_result)
            
        # Process results to create transcriptions with confidence filtering
        min_confidence = 0.75  # Minimum confidence threshold
        min_segment_duration = 0.5  # Minimum segment duration in seconds
        
        for result in results:
            if "result" not in result:
                continue
                
            words = result["result"]
            if not words:
                continue
                
            # Filter out low-confidence words
            filtered_words = filter_low_confidence_words(words, min_confidence)
            
            # Skip if no words remain after filtering
            if not filtered_words:
                continue
            
            # Create segments with improved timing
            segments = []
            current_segment = []
            segment_start = filtered_words[0]["start"]
            
            for i, word in enumerate(filtered_words):
                current_segment.append(word)
                
                # Decide if we should end the segment here
                end_segment = False
                
                # Check if this is the last word
                if i == len(filtered_words) - 1:
                    end_segment = True
                # Or if there's a natural pause (more than 0.5s between words)
                elif i < len(filtered_words) - 1 and filtered_words[i+1]["start"] - word["end"] > 0.5:
                    end_segment = True
                # Or if the segment is getting too long (more than 3 words)
                elif len(current_segment) >= 3:
                    end_segment = True
                
                if end_segment and current_segment:
                    segment_end = current_segment[-1]["end"]
                    segment_duration = segment_end - segment_start
                    
                    # Only include segments with minimum duration
                    if segment_duration >= min_segment_duration:
                        segment_text = create_merged_text(current_segment)
                        
                        if include_timecodes:
                            transcriptions.append(f"{segment_start:.2f}-{segment_end:.2f}: {segment_text}")
                        else:
                            transcriptions.append(segment_text)
                    
                    # Reset for next segment
                    if i < len(filtered_words) - 1:
                        current_segment = []
                        segment_start = filtered_words[i+1]["start"]
        
        transcription_time = time.time() - start_time
        log_func(f"Transcription completed in {transcription_time:.2f} seconds with confidence filtering.")
        
        # If no transcriptions were generated after filtering, add a message
        if not transcriptions:
            log_func("No high-confidence speech detected after filtering.")
            if include_timecodes:
                transcriptions = [f"0.0-5.0: No reliable speech detected in {track_name}"]
            else:
                transcriptions = [f"No reliable speech detected in {track_name}"]
        
        return transcriptions
        
    except Exception as e:
        log_func(f"An error occurred: {e}")
        return []

def write_transcriptions_to_file(transcriptions, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in transcriptions:
            file.write(line + '\n')