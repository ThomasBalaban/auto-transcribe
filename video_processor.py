"""
Video processing functionality for the SimpleAutoSubs application.
Simplified to use the new unified onomatopoeia detection system.
"""

import os
import tempfile
from transcriber import transcribe_audio, convert_to_audio
from embedder import convert_to_srt
from onomatopoeia_detector import OnomatopoeiaDetector
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
    def process_onomatopoeia(input_video_file, temp_dir, animation_setting, ai_sensitivity, log_func):
        """Process onomatopoeia detection using the unified system."""
        onomatopoeia_file_path = os.path.join(temp_dir, "onomatopoeia_subtitles")
        
        log_func("\n=== MULTIMODAL ONOMATOPOEIA DETECTION ===")
        log_func(f"Animation type: {animation_setting}")
        log_func(f"AI sensitivity: {ai_sensitivity}")
        
        try:
            # Initialize the unified detector (renamed from CompleteMultimodalDetector)
            detector = OnomatopoeiaDetector(
                sensitivity=ai_sensitivity,
                device="mps",  # Mac M4 optimization
                log_func=log_func
            )
            
            # Determine output file extension based on animation setting
            if animation_setting == "Static":
                output_path = onomatopoeia_file_path + ".srt"
            else:
                output_path = onomatopoeia_file_path + ".ass"
            
            # Create subtitle file
            success, events = detector.create_subtitle_file(
                input_video_file, output_path, animation_setting
            )
            
            if success and events:
                file_type = "animated" if output_path.endswith('.ass') else "static"
                log_func(f"✅ Multimodal onomatopoeia detection complete: {len(events)} {file_type} effects")
                return output_path, events
            else:
                log_func("No onomatopoeia effects generated")
                return None, []
                
        except Exception as e:
            log_func(f"Multimodal onomatopoeia detection failed: {e}")
            return None, []
    
    @staticmethod
    def process_single_video(input_file, output_file, model_path, device, ai_sensitivity, animation_setting, log_func):
        """Process a single video file with multimodal onomatopoeia detection."""
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
            
            # Process onomatopoeia using the unified system
            onomatopoeia_file_path, onomatopoeia_events = VideoProcessor.process_onomatopoeia(
                input_file, temp_dir, animation_setting, ai_sensitivity, log_func
            )
            
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
                if track2_srt_path: 
                    subtitle_types.append("microphone")
                if track3_srt_path: 
                    subtitle_types.append("desktop")
                if onomatopoeia_file_path: 
                    if onomatopoeia_file_path.endswith('.ass'):
                        subtitle_types.append(f"multimodal animated effects ({animation_setting.lower()})")
                    else:
                        subtitle_types.append(f"multimodal static effects")
                
                if subtitle_types:
                    log_func(f"✅ Video processing completed with {', '.join(subtitle_types)}: {os.path.basename(output_file)}")
                else:
                    log_func(f"Video processing completed (no subtitles added): {os.path.basename(output_file)}")
                    
            except Exception as e:
                log_func(f"ERROR during subtitle embedding: {str(e)}")
                raise Exception(f"Failed to embed subtitles: {e}")