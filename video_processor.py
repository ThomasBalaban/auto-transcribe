"""
Video processing functionality for the SimpleAutoSubs application.
Handles the core video processing logic separated from UI code.
"""

import os
import tempfile
from transcriber import transcribe_audio, convert_to_audio
from embedder import convert_to_srt
from integration_update import create_onomatopoeia_srt
from subtitle_embedder import embed_subtitles


class VideoProcessor:
    """Handles video processing operations."""
    
    @staticmethod
    def has_meaningful_speech(transcriptions):
        """Check if transcriptions contain meaningful speech (not just silence/errors)."""
        if not transcriptions:
            return False
        
        meaningful_lines = []
        for line in transcriptions:
            if ":" in line:
                text_part = line.split(":", 1)[1].strip()
                if (not text_part or 
                    "Audio file not found" in text_part or
                    "Audio file is empty" in text_part or
                    "Transcription error" in text_part):
                    continue
                meaningful_lines.append(text_part)
        
        return len(meaningful_lines) > 0
    
    @staticmethod
    def process_audio_track(input_file, track_index, track_name, temp_dir, model_path, device, 
                           include_timecodes, selected_language, is_mic_track, log_func):
        """Process a single audio track (extraction + transcription + SRT creation)."""
        log_func(f"\nPROCESSING {track_name.upper()}:")
        
        # Set up paths
        audio_path = os.path.join(temp_dir, f"track{track_index}_audio.wav")
        srt_path = os.path.join(temp_dir, f"track{track_index}_subtitles.srt")
        
        # Extract audio
        log_func(f"Extracting audio from {track_name}...")
        try:
            convert_to_audio(input_file, audio_path, track_index)
        except Exception as e:
            log_func(f"ERROR converting {track_name} to audio: {e}")
            raise Exception(f"Failed to extract audio from {track_name}: {e}")
            
        # Verify audio file
        if os.path.exists(audio_path):
            log_func(f"{track_name} audio file created successfully: {audio_path} (Size: {os.path.getsize(audio_path)} bytes)")
        else:
            log_func(f"ERROR: {track_name} audio file was not created: {audio_path}")
            raise Exception(f"{track_name} audio file was not created")

        # Transcribe audio
        log_func(f"Transcribing {track_name} audio...")
        transcriptions = transcribe_audio(model_path, device, audio_path, include_timecodes, 
                                        log_func, selected_language, track_name)
        
        log_func(f"{track_name} transcription complete. Got {len(transcriptions)} lines.")
        
        # Check for meaningful speech and create SRT
        has_speech = VideoProcessor.has_meaningful_speech(transcriptions)
        if has_speech:
            log_func(f"{track_name} contains meaningful speech - will create subtitles")
            
            # Show first few lines for debugging
            for i, line in enumerate(transcriptions[:3]):
                log_func(f"{track_name} transcription line {i+1}: {line}")
                
            # Convert to SRT
            log_func(f"Converting {track_name} transcriptions to SRT format...")
            text = "\n".join(transcriptions)
            convert_to_srt(text, srt_path, input_file, log_func, is_mic_track=is_mic_track)
            return srt_path
        else:
            log_func(f"{track_name} has no meaningful speech - skipping subtitle creation")
            return None
    
   
    @staticmethod
    def process_onomatopoeia(track3_audio_path, temp_dir, animation_setting, ai_sensitivity, log_func, input_video_file):
        """Process onomatopoeia detection with complete multimodal system."""
        onomatopoeia_srt_path = os.path.join(temp_dir, "onomatopoeia_subtitles.srt")
        
        if not track3_audio_path or not os.path.exists(track3_audio_path):
            log_func("Onomatopoeia detection: No desktop audio available")
            return None, []
        
        log_func("\n=== COMPLETE MULTIMODAL ONOMATOPOEIA SYSTEM ===")
        log_func("Phase 1-4: Onset + Video + Fusion + Gaming Optimization")
        log_func(f"Animation type: {animation_setting}")
        log_func(f"AI sensitivity: {ai_sensitivity}")
        
        try:
            # Try the complete multimodal system first
            try:
                from complete_multimodal_detector import CompleteMultimodalDetector
                
                log_func("🚀 Using Complete Multimodal System")
                detector = CompleteMultimodalDetector(
                    sensitivity=ai_sensitivity,
                    device="mps",  # Mac M4 optimization
                    log_func=log_func
                )
                
                events = detector.analyze_gaming_content(input_video_file, audio_path=track3_audio_path)
                
                if not events:
                    log_func("Complete multimodal system found no onomatopoeia events")
                    return None, []
                
                log_func(f"✅ Complete multimodal analysis: {len(events)} events")
                
            except ImportError as e:
                log_func(f"Complete multimodal system not available: {e}")
                log_func("Falling back to modern onset detection...")
                
                # Fallback to modern system
                from modern_onomatopoeia_detector import ModernOnomatopoeiaDetector
                
                detector = ModernOnomatopoeiaDetector(sensitivity=ai_sensitivity, log_func=log_func)
                events = detector.analyze_audio_file(track3_audio_path)
                
                if not events:
                    log_func("Modern system found no onomatopoeia events")
                    return None, []
            
            # Create animated or static version
            try:
                from animations.core import OnomatopoeiaAnimator
                ass_path = os.path.splitext(onomatopoeia_srt_path)[0] + '.ass'
                animator = OnomatopoeiaAnimator()
                animated_content = animator.generate_animated_ass_content(events, animation_setting)
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(animated_content)
                
                log_func(f"✓ Multimodal Animated Effects: {len(events)} events")
                log_func(f"🎮 Gaming-optimized with context awareness")
                return ass_path, events
                
            except ImportError:
                # Fallback to static SRT
                if hasattr(detector, 'generate_srt_content'):
                    srt_content = detector.generate_srt_content(events)
                else:
                    # Manual SRT generation for compatibility
                    srt_lines = []
                    for i, event in enumerate(events, 1):
                        start_time = event['start_time']
                        end_time = event['end_time'] 
                        word = event['word']
                        
                        start_formatted = VideoProcessor._format_srt_time(start_time)
                        end_formatted = VideoProcessor._format_srt_time(end_time)
                        
                        srt_lines.extend([
                            str(i),
                            f"{start_formatted} --> {end_formatted}",
                            word,
                            ""
                        ])
                    srt_content = "\n".join(srt_lines)
                
                with open(onomatopoeia_srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                
                log_func(f"✓ Multimodal Static Effects: {len(events)} events")
                return onomatopoeia_srt_path, events
                    
        except Exception as e:
            log_func(f"Multimodal onomatopoeia processing failed: {e}")
            return None, []
   
    @staticmethod
    def _format_srt_time(seconds):
        """Format time for SRT: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    
    @staticmethod
    def process_single_video(input_file, output_file, model_path, device, ai_sensitivity, animation_setting, log_func):
        """Process a single video file with AI-determined onomatopoeia."""
        include_timecodes = True  # Always use timecodes
        selected_language = "English"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_func(f"Created temporary directory: {temp_dir}")
            
            # Process Track 2 (Microphone)
            track2_srt_path = VideoProcessor.process_audio_track(
                input_file, 2, "Track 2 (Microphone)", temp_dir, model_path, device,
                include_timecodes, selected_language, is_mic_track=True, log_func=log_func
            )
            
            # Process Track 3 (Desktop)
            track3_srt_path = VideoProcessor.process_audio_track(
                input_file, 3, "Track 3 (Desktop)", temp_dir, model_path, device,
                include_timecodes, selected_language, is_mic_track=False, log_func=log_func
            )
            
            # Get track 3 audio path for onomatopoeia
            track3_audio_path = os.path.join(temp_dir, "track3_audio.wav")
            
            # FIXED: Pass original input_file for multimodal analysis
            onomatopoeia_file_path, onomatopoeia_events = VideoProcessor.process_onomatopoeia(
                track3_audio_path, temp_dir, animation_setting, ai_sensitivity, log_func, input_file
            )
            
            # Rest of the method remains the same...
            # Embed subtitles
            log_func("\nEMBEDDING SUBTITLES:")
            try:
                embed_subtitles(
                    input_file, 
                    output_file, 
                    track2_srt_path, 
                    track3_srt_path, 
                    onomatopoeia_file_path,
                    onomatopoeia_events, 
                    log_func
                )
                
                # Create success message
                subtitle_types = []
                if track2_srt_path: subtitle_types.append("microphone")
                if track3_srt_path: subtitle_types.append("desktop")
                if onomatopoeia_file_path: 
                    file_type = "AI animated effects" if onomatopoeia_file_path.endswith('.ass') else "AI comic effects"
                    subtitle_types.append(f"{file_type} ({animation_setting.lower()})")
                
                if subtitle_types:
                    log_func(f"Video processing completed successfully with {', '.join(subtitle_types)} subtitles: {os.path.basename(output_file)}")
                else:
                    log_func(f"Video processing completed (no subtitles added): {os.path.basename(output_file)}")
                    
            except Exception as e:
                log_func(f"ERROR during subtitle embedding: {str(e)}")
                raise Exception(f"Failed to embed subtitles: {e}")