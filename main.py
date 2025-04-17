import os
import threading
import tempfile
import time
import queue
import customtkinter as ctk  # type: ignore
import traceback
from tkinter import filedialog, messagebox
from transcriber import transcribe_audio, convert_to_audio, write_transcriptions_to_file
from embedder import convert_to_srt, embed_dual_subtitles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DualSubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleAutoSubs - Dual Track Subtitler")
        
        # Create a message queue for thread-safe logging
        self.message_queue = queue.Queue()
        
        # Set up GUI components
        self.setup_ui()
        
        # Start the log message processing
        self.root.after(100, self.process_log_messages)

    def log(self, message):
        """Thread-safe log function that puts messages in the queue"""
        self.message_queue.put(message + "\n")

    def process_log_messages(self):
        """Function to process log messages in the main thread"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log_box.insert(ctk.END, message)
                self.log_box.see(ctk.END)
        except queue.Empty:
            pass
        finally:
            # Schedule to run again after 100ms
            self.root.after(100, self.process_log_messages)

    def setup_ui(self):
        """Create and arrange all UI elements"""
        frame = ctk.CTkFrame(self.root)
        frame.grid(row=0, column=0, padx=20, pady=20)
        
        # Input file selection
        ctk.CTkLabel(frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
        self.file_entry = ctk.CTkEntry(frame, width=400)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file selection
        ctk.CTkLabel(frame, text="Output File:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_entry = ctk.CTkEntry(frame, width=400)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Options
        self.timecodes_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(frame, text="Include Timecodes (Recommended for Word-by-Word Subtitles)", 
                        variable=self.timecodes_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Main process button
        ctk.CTkButton(frame, text="Transcribe & Embed Subtitles", 
                     command=self.start_complete_process_thread,
                     height=40,
                     font=("Arial", 14, "bold")).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Transcription text area
        ctk.CTkLabel(frame, text="Transcription Preview (both tracks will be shown here after processing):").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.transcription_textbox = ctk.CTkTextbox(frame, height=250, width=600)
        self.transcription_textbox.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Log area
        ctk.CTkLabel(frame, text="Processing Log:").grid(
            row=6, column=0, sticky="w", pady=(10, 0))
        self.log_box = ctk.CTkTextbox(frame, height=150, width=600)
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Info labels
        self.model_status = ctk.CTkLabel(frame, text="Using Vosk English Model: vosk-model-en-us-0.42-gigaspeech")
        self.model_status.grid(row=8, column=0, columnspan=3, pady=5)
        
        self.track_info = ctk.CTkLabel(frame, text="Track 2 subtitles will appear above Track 3 subtitles")
        self.track_info.grid(row=9, column=0, columnspan=3, pady=5)
        
        self.subtitle_style_info = ctk.CTkLabel(frame, 
                                         text="Using word-by-word subtitle style (max 3 words per subtitle)")
        self.subtitle_style_info.grid(row=10, column=0, columnspan=3, pady=5)

    def browse_file(self):
        """Browse for input file"""
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entry.delete(0, ctk.END)
            self.file_entry.insert(0, file_path)

    def browse_output(self):
        """Browse for output file location"""
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if output_path:
            self.output_entry.delete(0, ctk.END)
            self.output_entry.insert(0, output_path)

    def start_complete_process_thread(self):
        """Start the complete process in a separate thread"""
        threading.Thread(target=self.complete_process).start()

    def complete_process(self):
        """Process both tracks and generate subtitled video"""
        input_file = self.file_entry.get()
        model_path = "local_modals/vosk-model-en-us-0.42-gigaspeech"
        device = "cpu"
        include_timecodes = self.timecodes_var.get()
        selected_language = "English"
        
        if not input_file:
            messagebox.showerror("Error", "Please select an input file.")
            return

        output_file = self.output_entry.get()
        if not output_file:
            messagebox.showerror("Error", "Please select an output file.")
            return

        self.log("Starting complete process: dual track transcription and subtitle embedding...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log(f"Created output directory: {output_dir}")
            except Exception as e:
                self.log(f"ERROR creating output directory: {e}")
                return
        
        # Step 1: Transcribe both audio tracks
        with tempfile.TemporaryDirectory() as temp_dir:
            self.log(f"Created temporary directory: {temp_dir}")
            
            # Paths for the temporary files
            track2_audio_path = os.path.join(temp_dir, "track2_audio.wav")
            track3_audio_path = os.path.join(temp_dir, "track3_audio.wav")
            track2_srt_path = os.path.join(temp_dir, "track2_subtitles.srt")
            track3_srt_path = os.path.join(temp_dir, "track3_subtitles.srt")
            
            if not input_file.endswith(('.mp4', '.mkv', '.avi')):
                messagebox.showerror("Error", "Input file must be a video file (MP4, MKV, or AVI).")
                return
            
            # Process Track 2 (Microphone)
            self.log("PROCESSING TRACK 2 (MICROPHONE):")
            self.log(f"Extracting audio from Track 2...")
            try:
                convert_to_audio(input_file, track2_audio_path, 2)  # Track index 2
            except Exception as e:
                self.log(f"ERROR converting Track 2 to audio: {e}")
                messagebox.showerror("Error", "Failed to extract audio from Track 2.")
                return
                
            # Debug check if audio file was created
            if os.path.exists(track2_audio_path):
                self.log(f"Track 2 audio file created successfully: {track2_audio_path} (Size: {os.path.getsize(track2_audio_path)} bytes)")
            else:
                self.log(f"ERROR: Track 2 audio file was not created: {track2_audio_path}")
                return

            # Transcribe audio for Track 2
            self.log("Transcribing Track 2 audio...")
            track2_transcriptions = transcribe_audio(model_path, device, track2_audio_path, include_timecodes, self.log, selected_language, "Track 2 (Mic)")
            
            self.log(f"Track 2 transcription complete. Got {len(track2_transcriptions)} lines.")
            # Debug: Show first few lines of transcription
            if track2_transcriptions:
                for i, line in enumerate(track2_transcriptions[:3]):
                    self.log(f"Track 2 transcription line {i+1}: {line}")
            else:
                self.log("WARNING: No transcriptions were generated for Track 2")
                track2_transcriptions = ["0.0-5.0: No speech detected in Track 2"]
                
            # Convert Track 2 transcriptions to SRT
            self.log("Converting Track 2 transcriptions to SRT format...")
            track2_text = "\n".join(track2_transcriptions)
            convert_to_srt(track2_text, track2_srt_path, input_file, self.log)
                
            # Process Track 3 (Desktop)
            self.log("\nPROCESSING TRACK 3 (DESKTOP):")
            self.log(f"Extracting audio from Track 3...")
            try:
                convert_to_audio(input_file, track3_audio_path, 3)  # Track index 3
            except Exception as e:
                self.log(f"ERROR converting Track 3 to audio: {e}")
                messagebox.showerror("Error", "Failed to extract audio from Track 3.")
                return
                
            # Debug check if audio file was created
            if os.path.exists(track3_audio_path):
                self.log(f"Track 3 audio file created successfully: {track3_audio_path} (Size: {os.path.getsize(track3_audio_path)} bytes)")
            else:
                self.log(f"ERROR: Track 3 audio file was not created: {track3_audio_path}")
                return

            # Transcribe audio for Track 3
            self.log("Transcribing Track 3 audio...")
            track3_transcriptions = transcribe_audio(model_path, device, track3_audio_path, include_timecodes, self.log, selected_language, "Track 3 (Desktop)")
            
            self.log(f"Track 3 transcription complete. Got {len(track3_transcriptions)} lines.")
            # Debug: Show first few lines of transcription
            if track3_transcriptions:
                for i, line in enumerate(track3_transcriptions[:3]):
                    self.log(f"Track 3 transcription line {i+1}: {line}")
            else:
                self.log("WARNING: No transcriptions were generated for Track 3")
                track3_transcriptions = ["0.0-5.0: No speech detected in Track 3"]
                
            # Convert Track 3 transcriptions to SRT
            self.log("Converting Track 3 transcriptions to SRT format...")
            track3_text = "\n".join(track3_transcriptions)
            convert_to_srt(track3_text, track3_srt_path, input_file, self.log)
                
            # Display combined transcription for editing - must use main thread
            def update_transcription_box():
                self.transcription_textbox.delete("1.0", ctk.END)
                self.transcription_textbox.insert(ctk.END, "=== TRACK 2 (MIC) TRANSCRIPTION ===\n")
                self.transcription_textbox.insert(ctk.END, "\n".join(track2_transcriptions))
                self.transcription_textbox.insert(ctk.END, "\n\n=== TRACK 3 (DESKTOP) TRANSCRIPTION ===\n")
                self.transcription_textbox.insert(ctk.END, "\n".join(track3_transcriptions))
            
            # Execute in main thread and wait for it to complete
            if threading.current_thread() is threading.main_thread():
                update_transcription_box()
            else:
                done_event = threading.Event()
                def main_thread_task():
                    update_transcription_box()
                    done_event.set()
                self.root.after(0, main_thread_task)
                done_event.wait(timeout=5)  # 5 second timeout
            
            # Step 2: Embed both subtitle tracks
            self.log("\nEmbedding both subtitle tracks into video...")
            
            try:
                # Embed both subtitle tracks
                embed_dual_subtitles(input_file, output_file, track2_srt_path, track3_srt_path, self.log)
                self.log("Complete process finished successfully!")
                    
            except Exception as e:
                self.log(f"ERROR during processing: {str(e)}")
                self.log(traceback.format_exc())


# Initialize the application
if __name__ == "__main__":
    # Set appearance mode and theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create root window
    root = ctk.CTk()
    
    # Create application instance
    app = DualSubtitleApp(root)
    
    # Start the main event loop
    root.mainloop()