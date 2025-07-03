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
        
        # List to store the input files and their corresponding output paths
        self.input_files = []
        self.output_files = []
        
        # Current processing index
        self.current_process_index = -1
        self.processing_active = False
        
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
        # Create a main scrollable frame that will contain everything
        main_scroll_container = ctk.CTkScrollableFrame(self.root, width=660, height=600)
        main_scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure the root window to be resizable and handle the scrollable frame
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create a frame inside the scrollable container for our UI elements
        frame = ctk.CTkFrame(main_scroll_container)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Files list frame
        files_frame = ctk.CTkFrame(frame)
        files_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Input files list
        ctk.CTkLabel(files_frame, text="Input Files:").pack(anchor="w", pady=5)
        
        # Create a frame for the textbox
        list_frame = ctk.CTkFrame(files_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a textbox instead of listbox (since CustomTkinter doesn't have CTkListbox)
        self.files_textbox = ctk.CTkTextbox(list_frame, height=120, width=600)
        self.files_textbox.pack(fill="both", expand=True)
        
        # Store indices for managing the list
        self.file_indices = []
        
        # File list buttons
        button_frame = ctk.CTkFrame(files_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # Add file button
        add_button = ctk.CTkButton(button_frame, text="Add Files", command=self.add_files)
        add_button.pack(side="left", padx=5, pady=5)
        
        # Remove file button
        remove_button = ctk.CTkButton(button_frame, text="Remove Last", command=self.remove_selected_file)
        remove_button.pack(side="left", padx=5, pady=5)
        
        # Clear all button
        clear_button = ctk.CTkButton(button_frame, text="Clear All", command=self.clear_all_files)
        clear_button.pack(side="left", padx=5, pady=5)
        
        # Output directory selection
        output_frame = ctk.CTkFrame(frame)
        output_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(output_frame, text="Output Directory:").pack(side="left", padx=5, pady=5)
        self.output_dir_entry = ctk.CTkEntry(output_frame, width=400)
        self.output_dir_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", command=self.browse_output_dir).pack(side="left", padx=5, pady=5)
        
        # Model selection and device frame
        model_frame = ctk.CTkFrame(frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        # Add model selection dropdown
        ctk.CTkLabel(model_frame, text="Whisper Model:").pack(side="left", padx=5, pady=5)
        self.model_var = ctk.StringVar(value="large")
        model_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=["tiny", "base", "small", "medium", "large"],
            variable=self.model_var
        )
        model_dropdown.pack(side="left", padx=5, pady=5)
        
        # Add device selection
        ctk.CTkLabel(model_frame, text="Device:").pack(side="left", padx=20, pady=5)
        self.device_var = ctk.StringVar(value="cpu")
        device_dropdown = ctk.CTkOptionMenu(
            model_frame,
            values=["cpu", "cuda"],
            variable=self.device_var
        )
        device_dropdown.pack(side="left", padx=5, pady=5)
        
        # Add WhisperX check button
        whisperx_check_button = ctk.CTkButton(
            model_frame,
            text="Check WhisperX",
            command=self.check_whisperx_availability,
            width=120
        )
        whisperx_check_button.pack(side="left", padx=(20, 5), pady=5)
        
        # Progress frame
        progress_frame = ctk.CTkFrame(frame)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        # Progress indicator
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.progress_label.pack(side="top", pady=2)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        self.progress_bar.pack(side="top", pady=5, fill="x")
        self.progress_bar.set(0)
        
        # Hidden but always-on timecode variable
        self.timecodes_var = ctk.BooleanVar(value=True)
        
        # Main process button
        self.process_button = ctk.CTkButton(
            frame, 
            text="Process All Videos", 
            command=self.start_batch_processing_thread,
            height=40,
            font=("Arial", 14, "bold")
        )
        self.process_button.pack(pady=10)
        
        # Log area
        ctk.CTkLabel(frame, text="Processing Log:").pack(anchor="w", pady=(10, 0))
        self.log_box = ctk.CTkTextbox(frame, height=450, width=600)
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)

    def check_whisperx_availability(self):
        """Check if WhisperX is available and display detailed status"""
        try:
            # Clear previous log messages for this check
            self.log("="*50)
            self.log("CHECKING WHISPERX AVAILABILITY...")
            self.log("="*50)
            
            # Check if WhisperX can be imported
            try:
                import whisperx # type: ignore
                self.log("✓ WhisperX module found and imported successfully")
                whisperx_available = True
            except ImportError as e:
                self.log(f"✗ WhisperX module import failed: {e}")
                whisperx_available = False
            
            # Check specific components if main import worked
            if whisperx_available:
                try:
                    from whisperx import load_align_model, align # type: ignore
                    self.log("✓ WhisperX alignment functions available")
                    components_available = True
                except ImportError as e:
                    self.log(f"✗ WhisperX alignment components missing: {e}")
                    components_available = False
            else:
                components_available = False
            
            # Check PyTorch (required for WhisperX)
            try:
                import torch # type: ignore
                self.log(f"✓ PyTorch found (version: {torch.__version__})")
                
                # Check CUDA availability if using CUDA device
                if self.device_var.get() == "cuda":
                    if torch.cuda.is_available():
                        self.log(f"✓ CUDA available (devices: {torch.cuda.device_count()})")
                        if torch.cuda.device_count() > 0:
                            cuda_info = f"Current device: {torch.cuda.get_device_name(0)}"
                            self.log(f"  {cuda_info}")
                    else:
                        self.log("⚠ CUDA not available (will fall back to CPU)")
            except ImportError:
                self.log("✗ PyTorch not found (required for WhisperX)")
            
            # Check other dependencies
            try:
                import faster_whisper # type: ignore
                self.log("✓ Faster-Whisper available")
            except ImportError:
                self.log("✗ Faster-Whisper not found")
            
            # Overall status summary
            self.log("-" * 30)
            if whisperx_available and components_available:
                self.log("🎉 WHISPERX STATUS: FULLY AVAILABLE")
                self.log("   Your transcriptions will use improved WhisperX alignment")
                messagebox.showinfo(
                    "WhisperX Status", 
                    "✓ WhisperX is fully available!\n\nYour transcriptions will use improved word-level alignment for better subtitle timing."
                )
            elif whisperx_available:
                self.log("⚠ WHISPERX STATUS: PARTIALLY AVAILABLE")
                self.log("   Some components missing - will fall back to standard Whisper")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "⚠ WhisperX is partially available.\n\nSome components are missing. The app will use standard Whisper timestamps instead."
                )
            else:
                self.log("❌ WHISPERX STATUS: NOT AVAILABLE")
                self.log("   Will use standard Whisper timestamps only")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "❌ WhisperX is not available.\n\nThe app will work fine but will use standard Whisper timestamps. For better accuracy, consider installing WhisperX:\n\npip install whisperx"
                )
            
            self.log("="*50)
            
        except Exception as e:
            error_msg = f"Error checking WhisperX availability: {e}"
            self.log(error_msg)
            messagebox.showerror("Check Error", error_msg)


    def check_whisperx_availability(self):
        """Check if WhisperX is available and display detailed status"""
        try:
            # Clear previous log messages for this check
            self.log("="*50)
            self.log("CHECKING WHISPERX AVAILABILITY...")
            self.log("="*50)
            
            # Check if WhisperX can be imported
            try:
                import whisperx # type: ignore
                self.log("✓ WhisperX module found and imported successfully")
                whisperx_available = True
            except ImportError as e:
                self.log(f"✗ WhisperX module import failed: {e}")
                whisperx_available = False
            
            # Check specific components if main import worked
            if whisperx_available:
                try:
                    from whisperx import load_align_model, align # type: ignore
                    self.log("✓ WhisperX alignment functions available")
                    components_available = True
                except ImportError as e:
                    self.log(f"✗ WhisperX alignment components missing: {e}")
                    components_available = False
            else:
                components_available = False
            
            # Check PyTorch (required for WhisperX)
            try:
                import torch # type: ignore
                self.log(f"✓ PyTorch found (version: {torch.__version__})")
                
                # Check CUDA availability if using CUDA device
                if self.device_var.get() == "cuda":
                    if torch.cuda.is_available():
                        self.log(f"✓ CUDA available (devices: {torch.cuda.device_count()})")
                        cuda_info = f"Current device: {torch.cuda.get_device_name(0)}"
                        self.log(f"  {cuda_info}")
                    else:
                        self.log("⚠ CUDA not available (will fall back to CPU)")
            except ImportError:
                self.log("✗ PyTorch not found (required for WhisperX)")
            
            # Check other dependencies
            try:
                import faster_whisper # type: ignore
                self.log("✓ Faster-Whisper available")
            except ImportError:
                self.log("✗ Faster-Whisper not found")
            
            # Overall status summary
            self.log("-" * 30)
            if whisperx_available and components_available:
                self.log("🎉 WHISPERX STATUS: FULLY AVAILABLE")
                self.log("   Your transcriptions will use improved WhisperX alignment")
                messagebox.showinfo(
                    "WhisperX Status", 
                    "✓ WhisperX is fully available!\n\nYour transcriptions will use improved word-level alignment for better subtitle timing."
                )
            elif whisperx_available:
                self.log("⚠ WHISPERX STATUS: PARTIALLY AVAILABLE")
                self.log("   Some components missing - will fall back to standard Whisper")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "⚠ WhisperX is partially available.\n\nSome components are missing. The app will use standard Whisper timestamps instead."
                )
            else:
                self.log("❌ WHISPERX STATUS: NOT AVAILABLE")
                self.log("   Will use standard Whisper timestamps only")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "❌ WhisperX is not available.\n\nThe app will work fine but will use standard Whisper timestamps. For better accuracy, consider installing WhisperX:\n\npip install whisperx"
                )
            
            self.log("="*50)
            
        except Exception as e:
            error_msg = f"Error checking WhisperX availability: {e}"
            self.log(error_msg)
            messagebox.showerror("Check Error", error_msg)

    def add_files(self):
        """Add multiple files to the list"""
        if len(self.input_files) >= 10:
            messagebox.showwarning("Maximum Files Reached", "You can only process up to 10 files at once.")
            return
            
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi")]
        )
        
        if not file_paths:
            return
            
        # Check if adding these would exceed the limit
        if len(self.input_files) + len(file_paths) > 10:
            remaining = 10 - len(self.input_files)
            messagebox.showwarning(
                "Maximum Files Reached", 
                f"You can only add {remaining} more file(s). Only the first {remaining} selected files will be added."
            )
            file_paths = file_paths[:remaining]
        
        # Add each file to the list
        for file_path in file_paths:
            if file_path in self.input_files:
                continue  # Skip duplicates
                
            self.input_files.append(file_path)
            
            # Generate default output path
            input_basename = os.path.basename(file_path)
            input_name, input_ext = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            
            # Get output directory
            output_dir = self.output_dir_entry.get()
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                # Update the output directory entry
                self.output_dir_entry.delete(0, ctk.END)
                self.output_dir_entry.insert(0, output_dir)
            
            # Create full output path
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if output file exists and create unique name if needed
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files.append(unique_output_path)
            
            # Add to text display with input -> output
            display_text = f"{os.path.basename(file_path)} → {os.path.basename(unique_output_path)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(len(self.input_files) - 1)

    def remove_selected_file(self):
        """Remove the last file from the list"""
        if not self.input_files:
            messagebox.showinfo("No Files", "There are no files to remove.")
            return
            
        # Remove the last file (since we can't easily select in a textbox)
        self.input_files.pop()
        self.output_files.pop()
        
        # Clear textbox and re-add all entries
        self.files_textbox.delete("1.0", ctk.END)
        self.file_indices = []
        
        for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
            display_text = f"{os.path.basename(input_file)} → {os.path.basename(output_file)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(i)

    def clear_all_files(self):
        """Clear all files from the list"""
        self.input_files = []
        self.output_files = []
        self.file_indices = []
        self.files_textbox.delete("1.0", ctk.END)

    def browse_output_dir(self):
        """Browse for output directory"""
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, output_dir)
            
            # Update all output paths based on the new directory
            self.update_output_paths(output_dir)

    def update_output_paths(self, output_dir):
        """Update all output paths based on a new directory"""
        if not self.input_files:
            return
            
        # Clear the textbox
        self.files_textbox.delete("1.0", ctk.END)
        self.file_indices = []
        
        # Update output paths
        for i, input_path in enumerate(self.input_files):
            input_basename = os.path.basename(input_path)
            input_name, input_ext = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            
            # Create full output path
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if output file exists and create unique name if needed
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files[i] = unique_output_path
            
            # Add to textbox with display of input -> output
            display_text = f"{os.path.basename(input_path)} → {os.path.basename(unique_output_path)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(i)

    def get_unique_output_path(self, base_path):
        """Generate a unique output filename by adding incremental counters"""
        if not os.path.exists(base_path):
            return base_path
            
        # If file already exists, add counter
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(f"{filename}-{counter}{ext}"):
            counter += 1
            
        return f"{filename}-{counter}{ext}"

    def start_batch_processing_thread(self):
        """Start the batch processing in a separate thread"""
        if not self.input_files:
            messagebox.showinfo("No Files", "Please add at least one video file to process.")
            return
            
        if self.processing_active:
            messagebox.showinfo("Processing Active", "Already processing videos. Please wait until completion.")
            return
            
        # Create output directory if it doesn't exist
        output_dir = self.output_dir_entry.get()
        if not output_dir:
            messagebox.showinfo("No Output Directory", "Please select an output directory.")
            return
            
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log(f"Created output directory: {output_dir}")
            except Exception as e:
                self.log(f"ERROR creating output directory: {e}")
                messagebox.showerror("Error", f"Could not create output directory: {e}")
                return
        
        # Disable the process button
        self.process_button.configure(state="disabled", text="Processing...")
        
        # Start the processing thread
        self.processing_active = True
        threading.Thread(target=self.process_all_videos).start()

    def process_all_videos(self):
        """Process all videos in the input list sequentially"""
        try:
            total_videos = len(self.input_files)
            self.log(f"Starting batch processing of {total_videos} videos...")
            
            for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
                self.current_process_index = i
                
                # Update progress indicator
                def update_progress():
                    self.progress_label.configure(text=f"Processing video {i+1} of {total_videos}: {os.path.basename(input_file)}")
                    self.progress_bar.set((i) / total_videos)  # Set progress before processing starts
                
                # Execute in main thread
                if threading.current_thread() is threading.main_thread():
                    update_progress()
                else:
                    self.root.after(0, update_progress)
                
                # Process the current video
                self.log(f"\n{'='*40}")
                self.log(f"PROCESSING VIDEO {i+1} OF {total_videos}:")
                self.log(f"Input: {input_file}")
                self.log(f"Output: {output_file}")
                self.log(f"{'='*40}\n")
                
                # Call the complete_process method for this video
                self.process_single_video(input_file, output_file)
                
                # Update progress after completion
                def update_after_completion():
                    # Update progress bar
                    self.progress_bar.set((i+1) / total_videos)
                    
                    # Update the displayed text to mark completion
                    self.files_textbox.delete("1.0", ctk.END)
                    for j, (in_file, out_file) in enumerate(zip(self.input_files, self.output_files)):
                        display_text = f"{os.path.basename(in_file)} → {os.path.basename(out_file)}"
                        if j <= i:  # Mark completed files
                            display_text += " ✓"
                        display_text += "\n"
                        self.files_textbox.insert(ctk.END, display_text)
                
                # Execute in main thread
                if threading.current_thread() is threading.main_thread():
                    update_after_completion()
                else:
                    self.root.after(0, update_after_completion)
            
            # All videos processed
            self.log("\n" + "="*40)
            self.log(f"BATCH PROCESSING COMPLETE! All {total_videos} videos processed successfully.")
            self.log("="*40)
            
            # Reset the UI when done
            def reset_ui():
                self.progress_label.configure(text=f"All {total_videos} videos processed successfully!")
                self.progress_bar.set(1.0)  # Full progress
                self.process_button.configure(state="normal", text="Process All Videos")
                # Show completion message from the main thread
                messagebox.showinfo("Processing Complete", f"All {total_videos} videos have been processed successfully!")
            
            # Execute reset_ui in the main thread
            if threading.current_thread() is threading.main_thread():
                reset_ui()
            else:
                self.root.after(0, reset_ui)
        
        # No messagebox display here - it's moved inside reset_ui function
        except Exception as e:
            self.log(f"ERROR during batch processing: {str(e)}")
            self.log(traceback.format_exc())
            
            # Reset the UI in case of error
            def reset_ui_error():
                self.progress_label.configure(text=f"Error processing videos. See log for details.")
                self.process_button.configure(state="normal", text="Process All Videos")
                # Show error message from the main thread
                messagebox.showerror("Processing Error", f"Error processing videos: {str(e)}")
            
            # Execute reset_ui_error in the main thread
            if threading.current_thread() is threading.main_thread():
                reset_ui_error()
            else:
                self.root.after(0, reset_ui_error)
            
            # No messagebox display here - it's moved inside reset_ui_error function
        
        finally:
            self.processing_active = False
            self.current_process_index = -1



    def process_single_video(self, input_file, output_file):
        """Process a single video file"""
        # Get selected model and device from UI
        model_path = self.model_var.get()  # Whisper model size
        device = self.device_var.get()     # CPU or CUDA
        
        include_timecodes = self.timecodes_var.get()
        selected_language = "English"
        
        # Step 1: Transcribe both audio tracks
        with tempfile.TemporaryDirectory() as temp_dir:
            self.log(f"Created temporary directory: {temp_dir}")
            
            # Paths for the temporary files
            track2_audio_path = os.path.join(temp_dir, "track2_audio.wav")
            track3_audio_path = os.path.join(temp_dir, "track3_audio.wav")
            track2_srt_path = os.path.join(temp_dir, "track2_subtitles.srt")
            track3_srt_path = os.path.join(temp_dir, "track3_subtitles.srt")
            
            # Process Track 2 (Microphone)
            self.log("PROCESSING TRACK 2 (MICROPHONE):")
            self.log(f"Extracting audio from Track 2...")
            try:
                convert_to_audio(input_file, track2_audio_path, 2)  # Track index 2
            except Exception as e:
                self.log(f"ERROR converting Track 2 to audio: {e}")
                raise Exception(f"Failed to extract audio from Track 2: {e}")
                
            # Debug check if audio file was created
            if os.path.exists(track2_audio_path):
                self.log(f"Track 2 audio file created successfully: {track2_audio_path} (Size: {os.path.getsize(track2_audio_path)} bytes)")
            else:
                self.log(f"ERROR: Track 2 audio file was not created: {track2_audio_path}")
                raise Exception("Track 2 audio file was not created")

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
            convert_to_srt(track2_text, track2_srt_path, input_file, self.log, is_mic_track=True)
                
            # Process Track 3 (Desktop)
            self.log("\nPROCESSING TRACK 3 (DESKTOP):")
            self.log(f"Extracting audio from Track 3...")
            try:
                convert_to_audio(input_file, track3_audio_path, 3)  # Track index 3
            except Exception as e:
                self.log(f"ERROR converting Track 3 to audio: {e}")
                raise Exception(f"Failed to extract audio from Track 3: {e}")
                
            # Debug check if audio file was created
            if os.path.exists(track3_audio_path):
                self.log(f"Track 3 audio file created successfully: {track3_audio_path} (Size: {os.path.getsize(track3_audio_path)} bytes)")
            else:
                self.log(f"ERROR: Track 3 audio file was not created: {track3_audio_path}")
                raise Exception("Track 3 audio file was not created")

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
            convert_to_srt(track3_text, track3_srt_path, input_file, self.log, is_mic_track=False)
            
            # Step 2: Embed both subtitle tracks
            self.log("\nEmbedding both subtitle tracks into video...")
            
            try:
                embed_dual_subtitles(input_file, output_file, track2_srt_path, track3_srt_path, self.log)
                self.log(f"Video processing completed successfully: {os.path.basename(output_file)}")
                
            except Exception as e:
                self.log(f"ERROR during embedding: {str(e)}")
                self.log(traceback.format_exc())
                raise Exception(f"Failed to embed subtitles: {e}")


# Initialize the application
if __name__ == "__main__":
    # Set appearance mode and theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create root window with reasonable dimensions
    root = ctk.CTk()
    root.title("SimpleAutoSubs - Dual Track Subtitler")
    root.geometry("700x700")  # Set initial size with reasonable height
    
    # Create application instance
    app = DualSubtitleApp(root)
    
    # Start the main event loop
    root.mainloop()