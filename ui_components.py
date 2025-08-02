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
                
                if torch.backends.mps.is_available():
                    app.log("✓ Apple Silicon GPU (MPS) available")
                elif torch.cuda.is_available():
                    app.log(f"✓ NVIDIA GPU (CUDA) available")
                else:
                    app.log("✓ CPU mode available")
            except ImportError:
                app.log("✗ PyTorch not found")
                messagebox.showerror("Missing Dependency", "PyTorch is required for the modern onomatopoeia system.\n\nInstall with: pip install torch transformers")
                return
            
            # Check Transformers
            try:
                import transformers
                app.log(f"✓ Transformers found (version: {transformers.__version__})")
            except ImportError:
                app.log("✗ Transformers not found")
                messagebox.showerror("Missing Dependency", "Transformers library is required.\n\nInstall with: pip install transformers")
                return
            
            # Test modern system
            app.log("\nInitializing modern CLAP + LLM system...")
            
            try:
                from modern_onomatopoeia_detector import ModernOnomatopoeiaDetector
                
                # Create detector (this will download models if needed)
                ai_detector = ModernOnomatopoeiaDetector(log_func=app.log)
                
                animation_type = app.animation_var.get()
                sensitivity = float(app.confidence_var.get())
                
                app.log(f"\nSystem Configuration:")
                app.log(f"  Animation Type: {animation_type}")
                app.log(f"  Detection Sensitivity: {sensitivity}")
                app.log(f"  Audio Sample Rate: {ai_detector.sample_rate}Hz")
                app.log(f"  Chunk Duration: {ai_detector.chunk_duration}s")
                app.log(f"  Processing Device: {ai_detector.device}")
                
                # Show available animations
                try:
                    from animations import OnomatopoeiaAnimator
                    all_animations = OnomatopoeiaAnimator.get_all_animation_types()
                    app.log(f"\nAvailable Animation Styles ({len(all_animations)}):")
                    for i, anim in enumerate(all_animations, 1):
                        display_name = anim.replace('_', ' ').title()
                        app.log(f"  {i}. {display_name}")
                except ImportError:
                    app.log("\n⚠️ Animation system not available")
                
                # Check model loading status
                models_loaded = []
                if ai_detector.clap_model is not None:
                    models_loaded.append("CLAP Audio Captioning")
                if ai_detector.llm_model is not None:
                    models_loaded.append("Local LLM Generation")
                
                if len(models_loaded) == 2:
                    app.log("\n🎉 MODERN ONOMATOPOEIA STATUS: FULLY OPERATIONAL")
                    app.log("✅ All models loaded successfully")
                    app.log(f"✅ Models loaded: {', '.join(models_loaded)}")
                    app.log("✅ Two-stage AI pipeline ready:")
                    app.log("   Stage 1: CLAP converts audio → natural language description")
                    app.log("   Stage 2: Local LLM converts description → comic sound effect")
                    app.log("✅ No hardcoded categories - unlimited creative potential")
                    app.log("✅ Context-aware onomatopoeia generation")
                    
                    messagebox.showinfo(
                        "Modern Onomatopoeia Test Successful",
                        f"✅ Modern CLAP + LLM system is fully operational!\n\n"
                        f"🔬 Two-stage AI pipeline:\n"
                        f"  • Stage 1: CLAP audio captioning\n"
                        f"  • Stage 2: Local LLM generation\n\n"
                        f"⚙️ Configuration:\n"
                        f"  • Animation: {animation_type}\n"
                        f"  • Sensitivity: {sensitivity}\n"
                        f"  • Device: {ai_detector.device}\n\n"
                        f"🚀 Features:\n"
                        f"  • No category limitations\n"
                        f"  • Natural language descriptions\n"
                        f"  • Creative contextual sound effects\n"
                        f"  • Modern 2023+ AI models"
                    )
                else:
                    app.log("\n❌ MODERN ONOMATOPOEIA STATUS: PARTIALLY LOADED")
                    app.log(f"⚠️ Only {len(models_loaded)}/2 models loaded")
                    if ai_detector.clap_model is None:
                        app.log("❌ CLAP model failed to load")
                    if ai_detector.llm_model is None:
                        app.log("❌ LLM model failed to load")
                    
                    messagebox.showerror(
                        "Modern System Partially Failed",
                        f"⚠️ Only {len(models_loaded)}/2 models loaded successfully.\n\n"
                        f"Check the log for details and ensure you have:\n"
                        f"• Stable internet connection for model downloads\n"
                        f"• Sufficient disk space (~2GB for models)\n"
                        f"• No firewall blocking Hugging Face downloads"
                    )
                
            except Exception as e:
                app.log(f"\n❌ MODERN SYSTEM INITIALIZATION FAILED")
                app.log(f"Error: {e}")
                
                # Check if it's a specific known issue
                error_str = str(e).lower()
                if "not a valid model identifier" in error_str:
                    suggestion = "Model identifier issue - check internet connection"
                elif "disk" in error_str or "space" in error_str:
                    suggestion = "Insufficient disk space - need ~2GB for models"
                elif "token" in error_str or "permission" in error_str:
                    suggestion = "Model access issue - try different internet connection"
                else:
                    suggestion = "Check dependencies: pip install torch transformers torchaudio"
                
                messagebox.showerror(
                    "Modern System Test Failed",
                    f"❌ Modern onomatopoeia system failed to initialize.\n\n"
                    f"Error: {str(e)[:200]}...\n\n"
                    f"💡 Suggestion: {suggestion}\n\n"
                    f"📋 Fallback: The app can still work with legacy YAMNet system."
                )
            
            app.log("="*50)
            
        except Exception as e:
            error_msg = f"Error testing modern onomatopoeia system: {e}"
            app.log(error_msg)
            messagebox.showerror("Test Error", error_msg)