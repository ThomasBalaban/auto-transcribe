# main.py
import os

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import threading
import queue
import customtkinter as ctk # type: ignore
import traceback
import json
import datetime
import sys
from tkinter import filedialog, messagebox

from ui.ui_components import UISetup, TestDialogs
from core.video_processor import VideoProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DualSubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleAutoSubs - Multimodal Onomatopoeia Subtitler")
        self.message_queue = queue.Queue()
        self.processing_active = False
        self.current_process_index = -1
        self.input_files = []
        self.output_files = []
        self.generated_titles = []
        self.file_indices = []
        
        # Logging variables
        self.session_log_path = None
        
        # Setup crash handling for GUI main loop
        self.root.report_callback_exception = self.handle_gui_exception
        
        self.setup_ui()
        self.root.after(100, self.process_log_messages)
        self.load_window_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def handle_gui_exception(self, exc_type, exc_value, exc_traceback):
        """Catches and logs errors that occur in the GUI event loop."""
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        full_msg = f"\n❌ CRITICAL APP CRASH (GUI):\n{error_msg}\n"
        print(full_msg) # Print to console
        self.log(full_msg) # Log to file/ui

    def save_window_geometry(self):
        try:
            geometry = self.root.geometry()
            with open("window_geometry.json", "w") as f:
                json.dump({"geometry": geometry}, f)
        except Exception as e:
            self.log(f"Error saving window geometry: {e}")

    def load_window_geometry(self):
        try:
            if os.path.exists("window_geometry.json"):
                with open("window_geometry.json", "r") as f:
                    config = json.load(f)
                    geometry = config.get("geometry")
                    if geometry:
                        self.root.geometry(geometry)
        except Exception as e:
            self.log(f"Error loading window geometry: {e}. Using default.")
            self.root.geometry("700x750")

    def on_closing(self):
        self.save_window_geometry()
        self.root.destroy()

    def setup_ui(self):
        """Create and arrange all UI elements."""
        frame = UISetup.create_main_layout(self.root)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(5, weight=1)

        files_frame, self.files_textbox = UISetup.create_file_list_section(frame, self)
        files_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        output_frame, self.output_dir_entry = UISetup.create_output_directory_section(frame, self)
        output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Added sync_slider to the returned tuple
        onomatopoeia_frame, self.animation_var, self.sync_slider = UISetup.create_onomatopoeia_section(frame, self)
        onomatopoeia_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        progress_frame, self.progress_label, self.progress_bar = UISetup.create_progress_section(frame)
        progress_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.process_button = UISetup.create_process_button(frame, self)
        self.process_button.grid(row=4, column=0, pady=10)

        log_frame, self.log_box = UISetup.create_log_section(frame)
        log_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)

    def log(self, message):
        """Logs message to UI and to the session log file if active."""
        # 1. Send to UI
        self.message_queue.put(message + "\n")
        
        # 2. Send to Log File
        if self.session_log_path:
            try:
                with open(self.session_log_path, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def process_log_messages(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log_box.insert(ctk.END, message)
                self.log_box.see(ctk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_messages)

    def check_system_status(self):
        TestDialogs.check_system_status(self)

    def add_files(self):
        if len(self.input_files) >= 30:
            messagebox.showwarning("Maximum Files Reached", "You can only process up to 30 files at once.")
            return
        file_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_paths: return

        for file_path in file_paths:
            if file_path in self.input_files: continue
            self.input_files.append(file_path)
            input_basename = os.path.basename(file_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_dir = self.output_dir_entry.get() or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            if not self.output_dir_entry.get():
                self.output_dir_entry.insert(0, output_dir)
            
            output_path = os.path.join(output_dir, output_filename)
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files.append(unique_output_path)
            self.generated_titles.append(None)
        self._refresh_files_display()

    def remove_selected_file(self):
        if not self.input_files:
            messagebox.showinfo("No Files", "There are no files to remove.")
            return
        self.input_files.pop()
        self.output_files.pop()
        self.generated_titles.pop()
        self._refresh_files_display()

    def clear_all_files(self):
        self.input_files = []
        self.output_files = []
        self.generated_titles.pop()
        self._refresh_files_display()

    def _refresh_files_display(self, completed_index=-1):
        """Refreshes the file list display, showing filenames, titles, and completion status."""
        self.files_textbox.delete("1.0", ctk.END)
        for i, (in_file, out_file) in enumerate(zip(self.input_files, self.output_files)):
            title = self.generated_titles[i] if i < len(self.generated_titles) and self.generated_titles[i] else None
            
            # Use the title for the output filename if it exists
            final_out_name = os.path.basename(out_file)
            if title:
                # This is for display only; the actual renaming happens in the processor
                from title_gen.title_generator import TitleGenerator
                temp_generator = TitleGenerator()
                filename = temp_generator.title_to_filename(title)
                final_out_name = f"{filename}.mp4"

            title_display = f" ({title})" if title else ""
            
            display_text = f"{os.path.basename(in_file)} → {final_out_name}{title_display}"
            
            if completed_index != -1 and i <= completed_index:
                display_text += " ✓"
            display_text += "\n"
            
            self.files_textbox.insert(ctk.END, display_text)

    def browse_output_dir(self):
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.update_output_paths(output_dir)

    def update_output_paths(self, output_dir):
        if not self.input_files: return
        self.output_files = []
        for input_path in self.input_files:
            input_basename = os.path.basename(input_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_path = os.path.join(output_dir, output_filename)
            self.output_files.append(self.get_unique_output_path(output_path))
        self._refresh_files_display()
        
    def get_unique_output_path(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(f"{filename}-{counter}{ext}"):
            counter += 1
        return f"{filename}-{counter}{ext}"

    def start_batch_processing_thread(self):
        if not self.input_files:
            messagebox.showinfo("No Files", "Please add at least one video file to process.")
            return
        if self.processing_active:
            messagebox.showinfo("Processing Active", "Already processing videos. Please wait.")
            return
        output_dir = self.output_dir_entry.get()
        if not output_dir:
            messagebox.showinfo("No Output Directory", "Please select an output directory.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # --- SETUP LOGGING ---
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.session_log_path = os.path.join(output_dir, f"{timestamp}.txt")
            
            with open(self.session_log_path, "w", encoding="utf-8") as f:
                f.write(f"--- BATCH PROCESSING STARTED: {timestamp} ---\n")
                f.write(f"Target Directory: {output_dir}\n")
                f.write(f"Files Queued: {len(self.input_files)}\n")
                f.write("="*60 + "\n\n")
            
            self.log(f"📄 Detailed log will be saved to: {self.session_log_path}")
        except Exception as e:
            self.log(f"⚠️ Warning: Could not create log file: {e}")
            self.session_log_path = None
        # ---------------------

        self.process_button.configure(state="disabled", text="Processing...")
        self.processing_active = True
        threading.Thread(target=self.process_all_videos, daemon=True).start()

    def process_all_videos(self):
        try:
            total_videos = len(self.input_files)
            self.log(f"Starting batch processing of {total_videos} videos...")
            animation_type = self.animation_var.get()
            sync_offset = self.sync_slider.get() # Get sync offset
            detailed_logs = True  # Always enabled

            for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
                self.current_process_index = i
                
                def update_ui_for_current_file():
                    self.progress_label.configure(text=f"Processing {i+1}/{total_videos}: {os.path.basename(input_file)}")
                    self.progress_bar.set(i / total_videos)
                self.root.after(0, update_ui_for_current_file)

                self.log(f"\n{'='*40}\nPROCESSING VIDEO {i+1}/{total_videos}\n{'='*40}")

                # Create a callback to update the title in the UI
                def title_callback(title):
                    self.generated_titles[i] = title
                    self.root.after(0, self._refresh_files_display)

                try:
                    final_path, suggested_title = VideoProcessor.process_single_video(
                        input_file=input_file,
                        output_file=output_file,
                        animation_type=animation_type,
                        sync_offset=sync_offset, # Pass sync offset
                        detailed_logs=detailed_logs,
                        log_func=self.log,
                        title_update_callback=title_callback
                    )
                    
                    self.output_files[i] = final_path
                    self.generated_titles[i] = suggested_title

                    # --- SUCCESS LOGGING ---
                    input_name = os.path.basename(input_file)
                    output_name = os.path.basename(final_path)
                    success_msg = f"SUCCESS: '{input_name}' -> '{output_name}'"
                    self.log("\n" + "*"*60)
                    self.log(success_msg)
                    self.log("*"*60 + "\n")
                    # -----------------------

                except Exception as file_error:
                    self.log(f"❌ ERROR processing file {os.path.basename(input_file)}: {file_error}")
                    self.log(traceback.format_exc())

                def update_ui_on_completion():
                    self.progress_bar.set((i + 1) / total_videos)
                    self._refresh_files_display(completed_index=i)
                self.root.after(0, update_ui_on_completion)

            def finalize_ui():
                self.progress_label.configure(text=f"All {total_videos} videos processed!")
                self.process_button.configure(state="normal", text="Process All Videos")
                messagebox.showinfo("Processing Complete", f"All {total_videos} videos processed!")
                if self.session_log_path:
                    self.log(f"Log file saved: {self.session_log_path}")
                    
            self.root.after(0, finalize_ui)

        except Exception as e:
            # Catch unexpected thread crashes
            crash_msg = f"🔥 FATAL BATCH PROCESSING CRASH: {e}"
            self.log(crash_msg)
            self.log(traceback.format_exc())
            
            def reset_ui_on_error():
                self.progress_label.configure(text="Error! See log.")
                self.process_button.configure(state="normal", text="Process All Videos")
            self.root.after(0, reset_ui_on_error)
        finally:
            self.processing_active = False
            self.current_process_index = -1

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = DualSubtitleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()