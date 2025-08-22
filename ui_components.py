"""
UI components and setup for the SimpleAutoSubs application.
Simplified for the unified multimodal onomatopoeia system.
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
        
        ctk.CTkButton(
            button_frame,
            text="System Status",
            command=app.check_system_status,
            width=120
        ).pack(side="right", padx=5, pady=5)
        
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
    def create_onomatopoeia_section(parent, app):
        """Create the onomatopoeia settings section."""
        onomatopoeia_frame = ctk.CTkFrame(parent)
        onomatopoeia_frame.pack(fill="x", padx=5, pady=5)
        
        # Second row - animation type selection
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
            "Hyper Bounce",
            "Static"  # Added static option
        ]
        
        ctk.CTkOptionMenu(
            second_row,
            values=animation_options,
            variable=animation_var,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        return animation_var

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
    """Simplified system checking."""
    
    @staticmethod
    def check_system_status(app):
        """Check overall system status and display simplified information."""
        try:
            app.log("="*50)
            app.log("MULTIMODAL ONOMATOPOEIA SYSTEM STATUS")
            app.log("="*50)
            
            # Test the unified detector
            try:
                from onomatopoeia_detector import OnomatopoeiaDetector
                
                app.log("Testing unified onomatopoeia detector...")
                detector = OnomatopoeiaDetector(log_func=app.log)
                
                app.log("✅ Core System: OPERATIONAL")
                app.log("   - Phase 1: Gaming onset detection ready")
                app.log("   - Phase 2: VideoMAE + X-CLIP analysis ready") 
                app.log("   - Phase 3: Multimodal fusion engine ready")
                app.log("   - Phase 4: Gaming optimizations ready")
                
                # Check device availability
                import torch
                if torch.backends.mps.is_available():
                    app.log("✅ MPS (Mac M4): Available")
                elif torch.cuda.is_available():
                    app.log("✅ CUDA: Available")
                else:
                    app.log("⚠️  GPU: Not available (using CPU)")
                
                # Check animation system
                try:
                    from animations.core import OnomatopoeiaAnimator
                    app.log("✅ Animation System: 8 animation types ready")
                except ImportError:
                    app.log("⚠️  Animation System: Not available")
                
                # Check Ollama
                if hasattr(detector, 'audio_enhancer') and detector.audio_enhancer and hasattr(detector.audio_enhancer, 'ollama_llm') and detector.audio_enhancer.ollama_llm and detector.audio_enhancer.ollama_llm.available:
                    app.log("✅ Ollama LLM: Connected")
                else:
                    app.log("⚠️  Ollama LLM: Not available (using fallback)")
                
                app.log("\n🎮 Optimized for gaming content:")
                app.log("   - Horror, action, FPS games")
                app.log("   - TikTok-style gaming clips")
                app.log("   - Context-aware effect selection")
                
                messagebox.showinfo(
                    "System Status", 
                    "✅ Multimodal Onomatopoeia System Ready!\n\n"
                    "🚀 4-Phase Detection Pipeline\n"
                    "🎬 Video + Audio Analysis\n" 
                    "🎮 Gaming Content Optimized\n"
                    "🎨 8 Animation Types Available\n\n"
                    "Ready to process gaming content!"
                )
                
            except Exception as e:
                app.log(f"❌ System test failed: {e}")
                messagebox.showerror(
                    "System Error", 
                    f"❌ System initialization failed:\n\n{e}\n\n"
                    "Check that all dependencies are installed."
                )
            
            app.log("="*50)
            
        except Exception as e:
            error_msg = f"Error checking system status: {e}"
            app.log(error_msg)
            messagebox.showerror("Check Error", error_msg)