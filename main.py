# main.py - UPDATED
import os
import threading
import queue
import customtkinter as ctk
import traceback
import json
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
        self.file_indices = []
        self.setup_ui()
        self.root.after(100, self.process_log_messages)
        self.load_window_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        frame.grid_rowconfigure(6, weight=1) # Log box row is now 6

        files_frame, self.files_textbox = UISetup.create_file_list_section(frame, self)
        files_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        output_frame, self.output_dir_entry = UISetup.create_output_directory_section(frame, self)
        output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        onomatopoeia_frame, self.animation_var = UISetup.create_onomatopoeia_section(frame, self)
        onomatopoeia_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # --- NEW: Settings Section ---
        settings_frame, self.detailed_logs_var = UISetup.create_settings_section(frame)
        settings_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        progress_frame, self.progress_label, self.progress_bar = UISetup.create_progress_section(frame)
        progress_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5) # Row is now 4

        self.process_button = UISetup.create_process_button(frame, self)
        self.process_button.grid(row=5, column=0, pady=10) # Row is now 5

        log_frame, self.log_box = UISetup.create_log_section(frame)
        log_frame.grid(row=6, column=0, sticky="nsew", padx=5, pady=5) # Row is now 6

    def log(self, message):
        self.message_queue.put(message + "\n")

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
        # This method remains unchanged
        if len(self.input_files) >= 15:
            messagebox.showwarning("Maximum Files Reached", "You can only process up to 15 files at once.")
            return
        file_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_paths:
            return
        if len(self.input_files) + len(file_paths) > 15:
            remaining = 15 - len(self.input_files)
            messagebox.showwarning("Maximum Files Reached", f"You can only add {remaining} more file(s).")
            file_paths = file_paths[:remaining]
        for file_path in file_paths:
            if file_path in self.input_files:
                continue
            self.input_files.append(file_path)
            input_basename = os.path.basename(file_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_dir = self.output_dir_entry.get()
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                self.output_dir_entry.delete(0, ctk.END)
                self.output_dir_entry.insert(0, output_dir)
            output_path = os.path.join(output_dir, output_filename)
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files.append(unique_output_path)
            display_text = f"{os.path.basename(file_path)} → {os.path.basename(unique_output_path)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(len(self.input_files) - 1)

    def remove_selected_file(self):
        # This method remains unchanged
        if not self.input_files:
            messagebox.showinfo("No Files", "There are no files to remove.")
            return
        self.input_files.pop()
        self.output_files.pop()
        self.refresh_file_display()

    def clear_all_files(self):
        # This method remains unchanged
        self.input_files = []
        self.output_files = []
        self.file_indices = []
        self.files_textbox.delete("1.0", ctk.END)

    def refresh_file_display(self):
        # This method remains unchanged
        self.files_textbox.delete("1.0", ctk.END)
        self.file_indices = []
        for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
            display_text = f"{os.path.basename(input_file)} → {os.path.basename(output_file)}\n"
            self.files_textbox.insert(ctk.END, display_text)
            self.file_indices.append(i)

    def browse_output_dir(self):
        # This method remains unchanged
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.update_output_paths(output_dir)

    def update_output_paths(self, output_dir):
        # This method remains unchanged
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
        # This method remains unchanged
        if not os.path.exists(base_path):
            return base_path
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(f"{filename}-{counter}{ext}"):
            counter += 1
        return f"{filename}-{counter}{ext}"

    def start_batch_processing_thread(self):
        # This method remains unchanged
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
        self.process_button.configure(state="disabled", text="Processing...")
        self.processing_active = True
        threading.Thread(target=self.process_all_videos).start()

    def process_all_videos(self):
        """Process all videos, now passing the detailed_logs setting."""
        try:
            total_videos = len(self.input_files)
            self.log(f"Starting batch processing of {total_videos} videos...")

            animation_type = self.animation_var.get()
            detailed_logs = self.detailed_logs_var.get() # Get the value from the checkbox
            self.log("SETTINGS: Automatic dialogue transcription and onomatopoeia are ENABLED.")
            self.log(f"  - Animation Type: {animation_type}")
            self.log(f"  - Detailed AI Logs: {detailed_logs}")

            for i, (input_file, output_file) in enumerate(zip(self.input_files, self.output_files)):
                self.current_process_index = i
                
                def update_progress():
                    self.progress_label.configure(text=f"Processing {i+1}/{total_videos}: {os.path.basename(input_file)}")
                    self.progress_bar.set(i / total_videos)
                self.root.after(0, update_progress)

                self.log(f"\n{'='*40}\nPROCESSING VIDEO {i+1}/{total_videos}\n{'='*40}")

                VideoProcessor.process_single_video(
                    input_file=input_file,
                    output_file=output_file,
                    animation_type=animation_type,
                    detailed_logs=detailed_logs, # Pass the setting
                    log_func=self.log
                )

                def update_completion():
                    self.progress_bar.set((i+1) / total_videos)
                    self.update_file_display_with_completion(i)
                self.root.after(0, update_completion)

            self.log(f"\n{'='*40}\nBATCH PROCESSING COMPLETE!\n{'='*40}")

            def reset_ui():
                self.progress_label.configure(text=f"All {total_videos} videos processed!")
                self.progress_bar.set(1.0)
                self.process_button.configure(state="normal", text="Process All Videos")
                messagebox.showinfo("Processing Complete", f"All {total_videos} videos processed!")
            self.root.after(0, reset_ui)

        except Exception as e:
            self.log(f"FATAL ERROR during batch processing: {e}")
            self.log(traceback.format_exc())
            def reset_ui_error():
                self.progress_label.configure(text="Error! See log.")
                self.process_button.configure(state="normal", text="Process All Videos")
                messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.root.after(0, reset_ui_error)
        finally:
            self.processing_active = False
            self.current_process_index = -1

    def update_file_display_with_completion(self, completed_index):
        # This method remains unchanged
        self.files_textbox.delete("1.0", ctk.END)
        for j, (in_file, out_file) in enumerate(zip(self.input_files, self.output_files)):
            display_text = f"{os.path.basename(in_file)} → {os.path.basename(out_file)}"
            if j <= completed_index:
                display_text += " ✓"
            display_text += "\n"
            self.files_textbox.insert(ctk.END, display_text)


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = DualSubtitleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()