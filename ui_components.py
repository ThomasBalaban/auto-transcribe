"""
UI components and setup for the SimpleAutoSubs application.
Updated to include all new animation style options.
"""

import customtkinter as ctk # type: ignore
from tkinter import messagebox


class UISetup:
    """Handles UI setup and component creation."""
    
    @staticmethod
    def create_main_layout(root):
        """Create the main scrollable layout."""
        main_scroll_container = ctk.CTkScrollableFrame(root, width=660, height=600)
        main_scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        frame = ctk.CTkFrame(main_scroll_container)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        return frame
    
    @staticmethod
    def create_file_list_section(parent, app):
        """Create the file list section with buttons."""
        files_frame = ctk.CTkFrame(parent)
        files_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(files_frame, text="Input Files:").pack(anchor="w", pady=5)
        
        list_frame = ctk.CTkFrame(files_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        files_textbox = ctk.CTkTextbox(list_frame, height=120, width=600)
        files_textbox.pack(fill="both", expand=True)
        
        # File list buttons
        button_frame = ctk.CTkFrame(files_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(button_frame, text="Add Files", command=app.add_files).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(button_frame, text="Remove Last", command=app.remove_selected_file).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(button_frame, text="Clear All", command=app.clear_all_files).pack(side="left", padx=5, pady=5)
        
        return files_textbox
    
    @staticmethod
    def create_output_directory_section(parent, app):
        """Create the output directory selection section."""
        output_frame = ctk.CTkFrame(parent)
        output_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(output_frame, text="Output Directory:").pack(side="left", padx=5, pady=5)
        output_dir_entry = ctk.CTkEntry(output_frame, width=400)
        output_dir_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", command=app.browse_output_dir).pack(side="left", padx=5, pady=5)
        
        return output_dir_entry
    
    @staticmethod
    def create_model_settings_section(parent, app):
        """Create the model and device selection section."""
        model_frame = ctk.CTkFrame(parent)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        # Model selection
        ctk.CTkLabel(model_frame, text="Whisper Model:").pack(side="left", padx=5, pady=5)
        model_var = ctk.StringVar(value="large")
        ctk.CTkOptionMenu(
            model_frame,
            values=["tiny", "base", "small", "medium", "large"],
            variable=model_var
        ).pack(side="left", padx=5, pady=5)
        
        # Device selection
        ctk.CTkLabel(model_frame, text="Device:").pack(side="left", padx=20, pady=5)
        device_var = ctk.StringVar(value="cpu")
        ctk.CTkOptionMenu(
            model_frame,
            values=["cpu", "cuda"],
            variable=device_var
        ).pack(side="left", padx=5, pady=5)
        
        # WhisperX check button
        ctk.CTkButton(
            model_frame,
            text="Check WhisperX",
            command=app.check_whisperx_availability,
            width=120
        ).pack(side="left", padx=(20, 5), pady=5)
        
        return model_var, device_var
    
    @staticmethod
    def create_onomatopoeia_section(parent, app):
        """Create the onomatopoeia settings section with AI sensitivity controls."""
        onomatopoeia_frame = ctk.CTkFrame(parent)
        onomatopoeia_frame.pack(fill="x", padx=5, pady=5)
        
        # First row - AI sensitivity and test button
        first_row = ctk.CTkFrame(onomatopoeia_frame)
        first_row.pack(fill="x", padx=5, pady=5)
        
        # UPDATED LABEL: Changed from "Sound Effects Confidence" to "AI Decision Sensitivity"
        ctk.CTkLabel(first_row, text="AI Decision Sensitivity:").pack(side="left", padx=5, pady=5)
        confidence_var = ctk.StringVar(value="0.5")  # Default to medium sensitivity
        ctk.CTkOptionMenu(
            first_row,
            values=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
            variable=confidence_var
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            first_row,
            text="Test AI Detection",  # Also updated this button text
            command=app.test_onomatopoeia,
            width=140
        ).pack(side="left", padx=(20, 5), pady=5)
        
        # Second row - animation type selection with all options
        second_row = ctk.CTkFrame(onomatopoeia_frame)
        second_row.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(second_row, text="Animation Type:").pack(side="left", padx=5, pady=5)
        animation_var = ctk.StringVar(value="Random")
        
        # All available animation options
        animation_options = [
            "Random",
            "Drift & Fade", 
            "Wiggle",
            "Pop & Shrink",
            "Shake",
            "Pulse",
            "Wave",
            "Explode-Out",
            "Hyper Bounce"
        ]
        
        ctk.CTkOptionMenu(
            second_row,
            values=animation_options,
            variable=animation_var,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        return confidence_var, animation_var


    @staticmethod
    def create_progress_section(parent):
        """Create the progress indicator section."""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        progress_label = ctk.CTkLabel(progress_frame, text="Ready")
        progress_label.pack(side="top", pady=2)
        
        progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        progress_bar.pack(side="top", pady=5, fill="x")
        progress_bar.set(0)
        
        return progress_label, progress_bar
    
    @staticmethod
    def create_process_button(parent, app):
        """Create the main process button."""
        return ctk.CTkButton(
            parent, 
            text="Process All Videos", 
            command=app.start_batch_processing_thread,
            height=40,
            font=("Arial", 14, "bold")
        )
    
    @staticmethod
    def create_log_section(parent):
        """Create the log area."""
        ctk.CTkLabel(parent, text="Processing Log:").pack(anchor="w", pady=(10, 0))
        log_box = ctk.CTkTextbox(parent, height=450, width=600)
        log_box.pack(fill="both", expand=True, padx=5, pady=5)
        return log_box


class TestDialogs:
    """Handles test dialog windows and system checks."""
    
    @staticmethod
    def check_whisperx_availability(app):
        """Check if WhisperX is available and display detailed status."""
        try:
            app.log("="*50)
            app.log("CHECKING WHISPERX AVAILABILITY...")
            app.log("="*50)
            
            # Check if WhisperX can be imported
            try:
                import whisperx # type: ignore
                app.log("✓ WhisperX module found and imported successfully")
                whisperx_available = True
            except ImportError as e:
                app.log(f"✗ WhisperX module import failed: {e}")
                whisperx_available = False
            
            # Check specific components if main import worked
            if whisperx_available:
                try:
                    from whisperx import load_align_model, align  # type: ignore
                    app.log("✓ WhisperX alignment functions available")
                    components_available = True
                except ImportError as e:
                    app.log(f"✗ WhisperX alignment components missing: {e}")
                    components_available = False
            else:
                components_available = False
            
            # Check PyTorch
            try:
                import torch  # type: ignore
                app.log(f"✓ PyTorch found (version: {torch.__version__})")
                
                if app.device_var.get() == "cuda":
                    if torch.cuda.is_available():
                        app.log(f"✓ CUDA available (devices: {torch.cuda.device_count()})")
                        if torch.cuda.device_count() > 0:
                            cuda_info = f"Current device: {torch.cuda.get_device_name(0)}"
                            app.log(f"  {cuda_info}")
                    else:
                        app.log("⚠ CUDA not available (will fall back to CPU)")
            except ImportError:
                app.log("✗ PyTorch not found (required for WhisperX)")
            
            # Check other dependencies
            try:
                import faster_whisper # type: ignore
                app.log("✓ Faster-Whisper available")
            except ImportError:
                app.log("✗ Faster-Whisper not found")
            
            # Show results
            app.log("-" * 30)
            if whisperx_available and components_available:
                app.log("🎉 WHISPERX STATUS: FULLY AVAILABLE")
                messagebox.showinfo(
                    "WhisperX Status", 
                    "✓ WhisperX is fully available!\n\nYour transcriptions will use improved word-level alignment for better subtitle timing."
                )
            elif whisperx_available:
                app.log("⚠ WHISPERX STATUS: PARTIALLY AVAILABLE")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "⚠ WhisperX is partially available.\n\nSome components are missing. The app will use standard Whisper timestamps instead."
                )
            else:
                app.log("❌ WHISPERX STATUS: NOT AVAILABLE")
                messagebox.showwarning(
                    "WhisperX Status", 
                    "❌ WhisperX is not available.\n\nThe app will work fine but will use standard Whisper timestamps. For better accuracy, consider installing WhisperX:\n\npip install whisperx"
                )
            
            app.log("="*50)
            
        except Exception as e:
            error_msg = f"Error checking WhisperX availability: {e}"
            app.log(error_msg)
            messagebox.showerror("Check Error", error_msg)
    
    
    @staticmethod
    def test_onomatopoeia(app):
        """Test modern CLAP + LLM onomatopoeia detection system."""
        try:
            app.log("="*50)
            app.log("TESTING MODERN ONOMATOPOEIA DETECTION SYSTEM")
            app.log("="*50)
            
            # Check PyTorch (required for modern system)
            try:
                import torch
                app.log(f"✓ PyTorch found (version: {torch.__version__})")
            except ImportError:
                app.log("✗ PyTorch not found")
                messagebox.showerror("Missing Dependency", "PyTorch is required.\n\nInstall: pip install torch transformers")
                return
            
            # Check Transformers
            try:
                import transformers
                app.log(f"✓ Transformers found (version: {transformers.__version__})")
            except ImportError:
                app.log("✗ Transformers not found")
                messagebox.showerror("Missing Dependency", "Transformers required.\n\nInstall: pip install transformers")
                return
            
            # Test modern system
            from modern_onomatopoeia_detector import ModernOnomatopoeiaDetector
            detector = ModernOnomatopoeiaDetector(log_func=app.log)
            
            if detector.clap_model and detector.llm_model:
                app.log("🎉 MODERN SYSTEM STATUS: FULLY OPERATIONAL")
                app.log("✅ CLAP + LLM pipeline ready")
                app.log("✅ No legacy dependencies")
                
                messagebox.showinfo(
                    "Modern System Test Successful",
                    "✅ Modern CLAP + LLM system operational!\n\n"
                    "🔬 Two-stage AI pipeline ready\n"
                    "🚀 No legacy YAMNet dependencies\n"
                    "💡 Creative contextual sound effects"
                )
            else:
                app.log("⚠️ System partially loaded")
                messagebox.showwarning("Partial Load", "⚠️ System partially loaded - check logs")
                
        except Exception as e:
            app.log(f"❌ Error: {e}")
            messagebox.showerror("Test Failed", f"❌ {e}")