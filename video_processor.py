"""
Video processing functionality for the SimpleAutoSubs application.
Handles the core video processing logic separated from UI code.
"""

import os
import tempfile
from transcriber import transcribe_audio, convert_to_audio
from embedder import convert_to_srt
from onomatopoeia_detector import create_onomatopoeia_srt
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
    def process_onomatopoeia(track3_audio_path, temp_dir, animation_setting, ai_sensitivity, log_func):
        """Process onomatopoeia detection with AI-determined durations."""
        onomatopoeia_srt_path = os.path.join(temp_dir, "onomatopoeia_subtitles.srt")
        
        if not track3_audio_path or not os.path.exists(track3_audio_path):
            log_func("Onomatopoeia detection: No desktop audio available")
            return None, []
        
        log_func("\n=== AI ONOMATOPOEIA PROCESSING ===")
        log_func("Using AI to determine all onomatopoeia timing naturally")
        log_func(f"Animation type: {animation_setting}")
        log_func(f"AI Sensitivity: {ai_sensitivity} (higher = more selective)")
        log_func("No confidence filters or energy thresholds applied")
        
        try:
            # Import here to pass the AI sensitivity
            from ai_onomatopoeia_detector import AIOnomatopoeiaDetector
            
            # Create AI detector with sensitivity
            ai_detector = AIOnomatopoeiaDetector(ai_sensitivity=ai_sensitivity, log_func=log_func)
            events = ai_detector.analyze_audio_file(track3_audio_path)
            
            if not events:
                log_func("AI determined no onomatopoeia events should be created")
                return None, []
            
            # Create animated or static version
            try:
                from animations.core import OnomatopoeiaAnimator
                log_func(f"Creating AI-timed animated effects (ASS format) - {animation_setting}")
                
                ass_path = os.path.splitext(onomatopoeia_srt_path)[0] + '.ass'
                animator = OnomatopoeiaAnimator()
                animated_content = animator.generate_animated_ass_content(events, animation_setting)
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(animated_content)
                
                onomatopoeia_file_path = ass_path
                log_func(f"✓ AI Animated Onomatopoeia: {len(events)} events (ASS format)")
                
            except ImportError:
                log_func("Animation module not available, creating static SRT")
                srt_content = ai_detector.generate_srt_content(events)
                with open(onomatopoeia_srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                onomatopoeia_file_path = onomatopoeia_srt_path
                log_func(f"✓ AI Static Onomatopoeia: {len(events)} events (SRT format)")
            
            # Show AI's decisions
            total_duration = sum(event['end_time'] - event['start_time'] for event in events)
            avg_duration = total_duration / len(events)
            
            log_func(f"AI Duration Analysis:")
            log_func(f"  Average duration: {avg_duration:.1f}s")
            log_func(f"  Total effect time: {total_duration:.1f}s")
            
            # Show examples of AI decisions
            for i, event in enumerate(events[:3]):
                duration = event['end_time'] - event['start_time']
                ai_score = event.get('ai_decision_score', 0)
                log_func(f"  Example {i+1}: '{event['word']}' - {duration:.1f}s @ {event['start_time']:.1f}s (AI score: {ai_score:.3f})")
            
            return onomatopoeia_file_path, events
                
        except Exception as e:
            log_func(f"AI Onomatopoeia processing failed: {e}")
            log_func("Continuing without comic book effects...")
            return None, []

    
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
            
            # Process AI onomatopoeia detection with sensitivity setting
            onomatopoeia_file_path, onomatopoeia_events = VideoProcessor.process_onomatopoeia(
                track3_audio_path, temp_dir, animation_setting, ai_sensitivity, log_func
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