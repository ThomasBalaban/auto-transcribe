import os
import threading
import tempfile
import time
import queue
import customtkinter as ctk # type: ignore
import traceback
from tkinter import filedialog, messagebox
from transcriber import transcribe_audio, convert_to_audio, write_transcriptions_to_file
from embedder import convert_to_srt, embed_subtitles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a message queue for thread-safe logging
message_queue = queue.Queue()

# Thread-safe log function that puts messages in the queue
def log(message):
    message_queue.put(message + "\n")

# Function to process log messages in the main thread
def process_log_messages():
    try:
        while True:
            message = message_queue.get_nowait()
            log_box.insert(ctk.END, message)
            log_box.see(ctk.END)
    except queue.Empty:
        pass
    finally:
        # Schedule to run again after 100ms
        root.after(100, process_log_messages)

# Update GUI from the main thread
def update_gui(widget, func, *args, **kwargs):
    if threading.current_thread() is threading.main_thread():
        return func(*args, **kwargs)
    else:
        root.after(0, lambda: func(*args, **kwargs))

# Browse file function
def browse_file(entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        entry.delete(0, ctk.END)
        entry.insert(0, file_path)

# Browse output directory function
def browse_output(entry):
    output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
    if output_path:
        entry.delete(0, ctk.END)
        entry.insert(0, output_path)

# Start the complete process in a separate thread
def start_complete_process_thread():
    threading.Thread(target=complete_process).start()

# Function to try deleting a file with retries
def try_delete_file(file_path, retries=5, delay=1):
    """Try to delete a file with retries."""
    for _ in range(retries):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(delay)
    log(f"Could not delete temporary SRT file after multiple attempts. It might still be in use: {file_path}")

# Combined process - transcribe and embed in one go
def complete_process():
    input_file = file_entry.get()
    model_path = "local_modals/vosk-model-en-us-0.42-gigaspeech"  # Fixed to your local Vosk model
    device = "cpu"  # Vosk doesn't use GPU acceleration like Whisper
    include_timecodes = timecodes_var.get()
    selected_language = "English"  # Fixed to English for Vosk
    
    if not input_file:
        messagebox.showerror("Error", "Please select an input file.")
        return

    output_file = output_entry.get()
    if not output_file:
        messagebox.showerror("Error", "Please select an output file.")
        return

    log("Starting complete process: transcription and subtitle embedding...")
    log("Using audio Track 2 (fixed)")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            log(f"Created output directory: {output_dir}")
        except Exception as e:
            log(f"ERROR creating output directory: {e}")
            return
    
    # Step 1: Transcribe the audio
    with tempfile.TemporaryDirectory() as temp_dir:
        log(f"Created temporary directory: {temp_dir}")
        audio_path = os.path.join(temp_dir, "temp_audio.wav")
        log(f"Will extract audio to: {audio_path}")
        
        if input_file.endswith(('.mp4', '.mkv', '.avi')):
            log("Converting video to audio (using Track 2)...")
            try:
                convert_to_audio(input_file, audio_path)
            except Exception as e:
                log(f"ERROR converting video to audio: {e}")
                messagebox.showerror("Error", "Failed to extract audio. The file may not contain the expected audio track.")
                return
        else:
            audio_path = input_file

        # Debug check if audio file was created
        if os.path.exists(audio_path):
            log(f"Audio file created successfully: {audio_path} (Size: {os.path.getsize(audio_path)} bytes)")
        else:
            log(f"ERROR: Audio file was not created: {audio_path}")
            return

        # Transcribe audio
        log("Transcribing audio...")
        transcriptions = transcribe_audio(model_path, device, audio_path, include_timecodes, log, selected_language)
        
        log(f"Transcription complete. Got {len(transcriptions)} lines.")
        # Debug: Show first few lines of transcription
        if transcriptions:
            for i, line in enumerate(transcriptions[:3]):
                log(f"Transcription line {i+1}: {line}")
        else:
            log("WARNING: No transcriptions were generated")
            # Show empty transcription box but continue with process
            transcriptions = ["0.0-5.0: No speech detected"]

        # Display transcription for editing - must use main thread
        def update_transcription_box():
            transcription_textbox.delete("1.0", ctk.END)
            transcription_textbox.insert(ctk.END, "\n".join(transcriptions))
        
        # Execute in main thread and wait for it to complete
        if threading.current_thread() is threading.main_thread():
            update_transcription_box()
        else:
            done_event = threading.Event()
            def main_thread_task():
                update_transcription_box()
                done_event.set()
            root.after(0, main_thread_task)
            done_event.wait(timeout=5)  # 5 second timeout
        
        # Step 2: Convert transcriptions to SRT and embed them
        log("Converting transcriptions to SRT format...")
        
        # Use a named temporary file with a fixed path that we control
        temp_srt_path = os.path.join(temp_dir, "subtitles.srt")
        log(f"Will create SRT file at: {temp_srt_path}")
        
        try:
            # Get the updated transcription text from the textbox (in case user edited it)
            transcription_text = ""
            
            def get_transcription_text():
                nonlocal transcription_text
                transcription_text = transcription_textbox.get("1.0", ctk.END).strip()
            
            # Execute in main thread and wait for completion
            if threading.current_thread() is threading.main_thread():
                get_transcription_text()
            else:
                # Create an event to signal completion
                done_event = threading.Event()
                def main_thread_task():
                    get_transcription_text()
                    done_event.set()
                root.after(0, main_thread_task)
                # Wait for completion
                done_event.wait(timeout=5)  # 5 second timeout
            
            log(f"Got transcription text: {len(transcription_text)} characters")
            
            # Convert to SRT with video file duration info
            convert_to_srt(transcription_text, temp_srt_path, input_file, log)
            
            # Check if the SRT file was created successfully
            if os.path.exists(temp_srt_path):
                log(f"SRT file created successfully at {temp_srt_path}")
                log(f"SRT file size: {os.path.getsize(temp_srt_path)} bytes")
                
                # Embed subtitles
                log("Embedding subtitles into video...")
                embed_subtitles(input_file, output_file, temp_srt_path, log)
                log("Complete process finished successfully!")
            else:
                log(f"ERROR: SRT file was not created at {temp_srt_path}")
                
        except Exception as e:
            log(f"ERROR during processing: {str(e)}")
            log(traceback.format_exc())

# Individual process functions (kept for backward compatibility)
def start_transcription_thread():
    threading.Thread(target=start_transcription).start()

def start_transcription():
    input_file = file_entry.get()
    model_path = "local_modals/vosk-model-en-us-0.42-gigaspeech"  # Fixed to your local Vosk model
    device = "cpu"  # Vosk doesn't use GPU acceleration like Whisper
    include_timecodes = timecodes_var.get()
    selected_language = "English"  # Fixed to English for Vosk
    
    if not input_file:
        messagebox.showerror("Error", "Please select an input file.")
        return

    output_file = output_entry.get()
    if not output_file:
        messagebox.showerror("Error", "Please select an output file.")
        return

    # Check if input file is video or audio
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "temp_audio.wav")
        if input_file.endswith(('.mp4', '.mkv', '.avi')):
            log("Converting video to audio (using Track 2)...")
            convert_to_audio(input_file, audio_path)
        else:
            audio_path = input_file

        # Transcribe audio
        transcriptions = transcribe_audio(model_path, device, audio_path, include_timecodes, log, selected_language)

        # Save transcriptions to file
        write_transcriptions_to_file(transcriptions, output_file)

    # Display transcription for editing - must use main thread
    def update_transcription_box():
        transcription_textbox.delete("1.0", ctk.END)
        transcription_textbox.insert(ctk.END, "\n".join(transcriptions))
    
    root.after(0, update_transcription_box)

def start_embedding_thread():
    threading.Thread(target=start_embedding).start()

def start_embedding():
    input_text = transcription_textbox.get("1.0", ctk.END).strip()
    input_video = file_entry.get()
    output_video = output_entry.get()

    if not input_text or not input_video or not output_video:
        messagebox.showerror("Error", "Please ensure all fields are filled.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_srt_path = os.path.join(temp_dir, "subtitles.srt")
        log(f"Created temporary SRT file: {temp_srt_path}")
        
        # Use updated convert_to_srt with video file parameter
        convert_to_srt(input_text, temp_srt_path, input_video, log)
        
        try:
            embed_subtitles(input_video, output_video, temp_srt_path, log)
        except Exception as e:
            log(f"ERROR during embedding: {str(e)}")
            log(traceback.format_exc())

# Set up GUI
ctk.set_appearance_mode("dark")  # Modes: "system" (default), "light", "dark"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

root = ctk.CTk()
root.title("SimpleAutoSubs - Track 2 Subtitler")

frame = ctk.CTkFrame(root)
frame.grid(row=0, column=0, padx=20, pady=20)

ctk.CTkLabel(frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
file_entry = ctk.CTkEntry(frame, width=400)
file_entry.grid(row=0, column=1, padx=5, pady=5)
ctk.CTkButton(frame, text="Browse", command=lambda: browse_file(file_entry)).grid(row=0, column=2, padx=5, pady=5)

ctk.CTkLabel(frame, text="Output File:").grid(row=1, column=0, sticky="w", pady=5)
output_entry = ctk.CTkEntry(frame, width=400)
output_entry.grid(row=1, column=1, padx=5, pady=5)
ctk.CTkButton(frame, text="Browse", command=lambda: browse_output(output_entry)).grid(row=1, column=2, padx=5, pady=5)

timecodes_var = ctk.BooleanVar()
timecodes_var.set(True)  # Default to True for word-by-word subtitles
ctk.CTkCheckBox(frame, text="Include Timecodes (Recommended for Word-by-Word Subtitles)", 
                variable=timecodes_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

# Main process button - one click to do everything
ctk.CTkButton(frame, text="Transcribe & Embed Subtitles", command=start_complete_process_thread).grid(row=3, column=0, columnspan=3, pady=10)

transcription_textbox = ctk.CTkTextbox(frame, height=200, width=600)
transcription_textbox.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

log_box = ctk.CTkTextbox(frame, height=100, width=600)
log_box.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

# Status label to display the current model
model_status = ctk.CTkLabel(frame, text="Using Vosk English Model: vosk-model-en-us-0.42-gigaspeech")
model_status.grid(row=6, column=0, columnspan=3, pady=5)

track_info = ctk.CTkLabel(frame, text="Using Track 2 for subtitle extraction (fixed)")
track_info.grid(row=7, column=0, columnspan=3, pady=5)

subtitle_style_info = ctk.CTkLabel(frame, text="Using word-by-word subtitle style (max 3 words per subtitle)")
subtitle_style_info.grid(row=8, column=0, columnspan=3, pady=5)

# Start the log message processing
root.after(100, process_log_messages)

root.mainloop()