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

def convert_to_audio(input_file, output_file):
    """Convert video to audio, extracting Track 2"""
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
        
        # Use a fixed track index of 2 (the second audio track)
        track_index = 2
        log(f"Extracting audio track {track_index} (Track2) from video...")
        
        # Convert to mono 16kHz WAV which is optimal for Vosk
        # Specify the audio track using -map option
        command = [
            ffmpeg_path, 
            "-i", input_file,
            "-map", f"0:{track_index}",  # Select Track 2 by index
            "-ar", "16000",              # 16kHz sample rate
            "-ac", "1",                  # mono audio
            "-c:a", "pcm_s16le",         # PCM 16-bit format
            output_file
        ]
        
        log(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"Error in ffmpeg: {result.stderr}")
            # If Track 2 fails, try with Track 1 as fallback
            log("Track 2 extraction failed, trying Track 1 as fallback...")
            command[4] = "0:1"  # Change to Track 1
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                log(f"Fallback to Track 1 also failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            else:
                log("Successfully extracted Track 1 as fallback")
        
        # Verify the output file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            log(f"Successfully extracted audio to {output_file}")
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

def transcribe_audio(model_path, device, audio_path, include_timecodes, log_func, language):
    try:
        start_time = time.time()
        log_func(f"Starting transcription for {audio_path} (Track 2)")
        
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
            
        # Process results to create transcriptions
        for result in results:
            if "result" not in result:
                continue
                
            words = result["result"]
            if not words:
                continue
                
            text = result.get("text", "")
            
            if include_timecodes and words:
                start = words[0]["start"]
                end = words[-1]["end"]
                transcriptions.append(f"{start:.2f}-{end:.2f}: {text}")
            else:
                transcriptions.append(text)
        
        transcription_time = time.time() - start_time
        log_func(f"Transcription completed in {transcription_time:.2f} seconds.")
        return transcriptions
        
    except Exception as e:
        log_func(f"An error occurred: {e}")
        return []

def write_transcriptions_to_file(transcriptions, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in transcriptions:
            file.write(line + '\n')