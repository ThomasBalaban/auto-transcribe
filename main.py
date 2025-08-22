"""
SimpleAutoSubs - Main Application
Dual track subtitle generator with multimodal animated onomatopoeia effects.
Simplified to use the unified detection system.
"""

import os
import threading
import queue
import customtkinter as ctk # type: ignore
import traceback
from tkinter import filedialog, messagebox
from multimodal_events import MultimodalOnomatopoeia, Config, sliding_windows

from ui_components import UISetup, TestDialogs
from video_processor import VideoProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DualSubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleAutoSubs - Multimodal Onomatopoeia Subtitler")
        
        # Threading and logging
        self.message_queue = queue.Queue()
        self.processing_active = False
        self.current_process_index = -1
        
        # File management
        self.input_files = []
        self.output_files = []
        self.file_indices = []
        
        # Set up UI
        self.setup_ui()
        
        # Start log message processing
        self.root.after(100, self.process_log_messages)

    def setup_ui(self):
        """Create and arrange all UI elements using the UI components module."""
        # Create main layout
        frame = UISetup.create_main_layout(self.root)
        
        # Create sections
        self.files_textbox = UISetup.create_file_list_section(frame, self)
        self.output_dir_entry = UISetup.create_output_directory_section(frame, self)
        
        # Create onomatopoeia section
        self.animation_var = UISetup.create_onomatopoeia_section(frame, self)
        
        self.progress_label, self.progress_bar = UISetup.create_progress_section(frame)
        
        # Process button
        self.process_button = UISetup.create_process_button(frame, self)
        self.process_button.pack(pady=10)
        
        # Log area
        self.log_box = UISetup.create_log_section(frame)
        
        # Hidden timecode variable (always enabled)
        self.timecodes_var = ctk.BooleanVar(value=True)

    def log(self, message):
        """Thread-safe log function."""
        self.message_queue.put(message + "\n")

    def process_log_messages(self):
        """Process log messages in the main thread."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log_box.insert(ctk.END, message)
                self.log_box.see(ctk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_messages)

    # Simplified system checking
    def check_system_status(self):
        """Check multimodal system status."""
        TestDialogs.check_system_status(self)

    # File management methods (unchanged)
    def add_files(self):
        """Add multiple files to the list."""
        if len(self.input_files) >= 15:
            messagebox.showwarning("Maximum Files Reached", "You can only process up to 15 files at once.")
            return
            
        file_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_paths:
            return
            
        # Check file limit
        if len(self.input_files) + len(file_paths) > 15:
            remaining = 15 - len(self.input_files)
            messagebox.showwarning(
                "Maximum Files Reached", 
                f"You can only add {remaining} more file(s). Only the first {remaining} selected files will be added."
            )
            file_paths = file_paths[:remaining]
        
        # Add files
        for file_path in file_paths:
            if file_path in self.input_files:
                continue
                
            self.input_files.append(file_path)
            
            # Generate output path
            input_basename = os.path.basename(file_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            
            # Get output directory
            output_dir = self.output_dir_entry.get()
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                self.output_dir_entry.delete(0, ctk.END)
                self.output_dir_entry.insert(0, output_dir)
            
            # Create unique output path
            output_path = os.path.join(output_dir, output_filename)
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files.append(unique_output_path)
            
            # Add to display
            display_text = f"{os.path.basename(file_path)} → {os.path.basename(unique_output_path)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(len(self.input_files) - 1)

    def remove_selected_file(self):
        """Remove the last file from the list."""
        if not self.input_files:
            messagebox.showinfo("No Files", "There are no files to remove.")
            return
            
        self.input_files.pop()
        self.output_files.pop()
        self.refresh_file_display()

    def clear_all_files(self):
        """Clear all files from the list."""
        self.input_files = []
        self.output_files = []
        self.file_indices = []
        self.files_textbox.delete("1.0", ctk.END)

    def refresh_file_display(self):
        """Refresh the file display."""
        self.files_textbox.delete("1.0", ctk.END)
        self.file_indices = []
        
        for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
            display_text = f"{os.path.basename(input_file)} → {os.path.basename(output_file)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(i)

    def browse_output_dir(self):
        """Browse for output directory."""
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.update_output_paths(output_dir)

    def update_output_paths(self, output_dir):
        """Update all output paths based on a new directory."""
        if not self.input_files:
            return
            
        for i, input_path in enumerate(self.input_files):
            input_basename = os.path.basename(input_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_path = os.path.join(output_dir, output_filename)
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files[i] = unique_output_path
            
        self.refresh_file_display()

    def get_unique_output_path(self, base_path):
        """Generate a unique output filename by adding incremental counters."""
        if not os.path.exists(base_path):
            return base_path
            
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(f"{filename}-{counter}{ext}"):
            counter += 1
            
        return f"{filename}-{counter}{ext}"

    # Processing methods (simplified)
    def start_batch_processing_thread(self):
        """Start the batch processing in a separate thread."""
        if not self.input_files:
            messagebox.showinfo("No Files", "Please add at least one video file to process.")
            return
            
        if self.processing_active:
            messagebox.showinfo("Processing Active", "Already processing videos. Please wait until completion.")
            return
            
        # Create output directory
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
        
        # Start processing
        self.process_button.configure(state="disabled", text="Processing...")
        self.processing_active = True
        threading.Thread(target=self.process_all_videos).start()

    def process_all_videos(self):
        """Process all videos in the input list sequentially."""
        try:
            total_videos = len(self.input_files)
            self.log(f"Starting batch processing of {total_videos} videos...")
            
            # Log settings
            animation_type = self.animation_var.get()
            self.log(f"Multimodal Detection Settings:")
            self.log(f"  Animation Type: {animation_type}")
            
            for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
                self.current_process_index = i
                
                # Update progress
                def update_progress():
                    self.progress_label.configure(text=f"Processing video {i+1} of {total_videos}: {os.path.basename(input_file)}")
                    self.progress_bar.set(i / total_videos)
                
                self.root.after(0, update_progress)
                
                # Process video
                self.log(f"\n{'='*40}")
                self.log(f"PROCESSING VIDEO {i+1} OF {total_videos}:")
                self.log(f"Input: {input_file}")
                self.log(f"Output: {output_file}")
                self.log(f"{'='*40}\n")
                
                # Use simplified VideoProcessor
                VideoProcessor.process_single_video(
                    input_file, 
                    output_file, 
                    animation_type,
                    self.log
                )
                
                # Update completion status
                def update_completion():
                    self.progress_bar.set((i+1) / total_videos)
                    self.update_file_display_with_completion(i)
                
                self.root.after(0, update_completion)
            
            # All videos processed
            self.log("\n" + "="*40)
            self.log(f"BATCH PROCESSING COMPLETE! All {total_videos} videos processed successfully.")
            self.log("="*40)
            
            # Reset UI
            def reset_ui():
                self.progress_label.configure(text=f"All {total_videos} videos processed successfully!")
                self.progress_bar.set(1.0)
                self.process_button.configure(state="normal", text="Process All Videos")
                messagebox.showinfo("Processing Complete", f"All {total_videos} videos have been processed successfully!")
            
            self.root.after(0, reset_ui)
        
        except Exception as e:
            self.log(f"ERROR during batch processing: {str(e)}")
            self.log(traceback.format_exc())
            
            def reset_ui_error():
                self.progress_label.configure(text="Error processing videos. See log for details.")
                self.process_button.configure(state="normal", text="Process All Videos")
                messagebox.showerror("Processing Error", f"Error processing videos: {str(e)}")
            
            self.root.after(0, reset_ui_error)
        
        finally:
            self.processing_active = False
            self.current_process_index = -1

    def update_file_display_with_completion(self, completed_index):
        """Update file display to show completion status."""
        self.files_textbox.delete("1.0", ctk.END)
        for j, (in_file, out_file) in enumerate(zip(self.input_files, self.output_files)):
            display_text = f"{os.path.basename(in_file)} → {os.path.basename(out_file)}"
            if j <= completed_index:
                display_text += " ✓"
            display_text += "\n"
            self.files_textbox.insert(ctk.END, display_text)


def main():
    """Initialize and run the application."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.title("SimpleAutoSubs - Multimodal Onomatopoeia Subtitler")
    root.geometry("700x750")
    
    app = DualSubtitleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()