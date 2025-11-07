# core/video_processor.py
import os
import shutil
import tempfile
import gc

from onomatopoeia_detector import OnomatopoeiaDetector
import core.transcriber
from core.subtitle_converter import convert_to_srt
from core.subtitle_embedder import embed_subtitles
from ai_director.master_director import MasterDirector
from ai_director.video_editor import VideoEditor
from video_utils import get_video_duration
from title_gen.title_generator import TitleGenerator
from clip_editor.intelligent_trimmer import IntelligentTrimmer
from utils.timestamp_processor import extend_segments_for_dialogue


class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        detailed_logs: bool,
        log_func,
        title_update_callback=None,
        enable_trimming: bool = True 
    ):
        temp_dir = tempfile.gettempdir()
        final_output_path = output_file
        suggested_title = None
        trim_segments = None
        DIALOGUE_TRIM_BUFFER = 0.5

        try:
            log_func("="*60)
            log_func(f"STARTING FULL VIDEO PROCESSING: {os.path.basename(input_file)}")
            log_func("="*60)

            # --- PHASE 1: Dialogue Transcription ---
            log_func("\n--- PHASE 1: Dialogue Transcription ---")
            
            # Store BOTH raw and adjusted versions
            mic_transcriptions_raw = []
            mic_transcriptions_adjusted = []
            desktop_transcriptions_raw = []
            desktop_transcriptions_adjusted = []
            mic_subtitle_path, desktop_subtitle_path = None, None
            
            # Transcribe Microphone Track
            mic_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            
            if core.transcriber.convert_to_audio(input_file, mic_audio_path, "a:1", log_func):
                # Get BOTH versions from transcriber
                mic_transcriptions_raw, mic_transcriptions_adjusted = core.transcriber.transcribe_audio(
                    "large", "cpu", mic_audio_path, True, log_func, "English", "Track 2 (Mic)"
                )
                
                # Create subtitle file using ADJUSTED version (for display)
                mic_subtitle_path_srt = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt("\n".join(mic_transcriptions_adjusted), mic_subtitle_path_srt, input_file, log_func, is_mic_track=True)
                mic_subtitle_path = mic_subtitle_path_srt.replace(".srt", ".ass")
                
                os.remove(mic_audio_path)
                log_func(f"✅ Mic transcription complete:")
                log_func(f"   - {len(mic_transcriptions_raw)} words transcribed")
                log_func(f"   - Raw timestamps: Whisper's best guess at actual timing")
                log_func(f"   - Adjusted timestamps: Display-optimized for subtitles")
            else:
                log_func("⚠️ Mic audio extraction failed")

            # Transcribe Desktop Track
            desktop_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            
            if core.transcriber.convert_to_audio(input_file, desktop_audio_path, "a:2", log_func):
                # Get BOTH versions from transcriber
                desktop_transcriptions_raw, desktop_transcriptions_adjusted = core.transcriber.transcribe_audio(
                    "large", "cpu", desktop_audio_path, True, log_func, "English", "Track 3 (Desktop)"
                )
                
                # Create subtitle file using ADJUSTED version (for display)
                desktop_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt("\n".join(desktop_transcriptions_adjusted), desktop_subtitle_path, input_file, log_func)
                
                os.remove(desktop_audio_path)
                log_func(f"✅ Desktop transcription complete:")
                log_func(f"   - {len(desktop_transcriptions_raw)} words transcribed")
            else:
                log_func("⚠️ Desktop audio extraction failed")

            # --- PHASE 2: AI Title Generation (Context-Aware) ---
            log_func("\n--- PHASE 2: AI Title Generation (Context-Aware) ---")
            log_func("   Using RAW timestamps - Gemini will see Whisper's best guess at actual timing")
            
            title_generator = TitleGenerator(log_func=log_func)
            title_details = title_generator.generate_title(
                video_path=input_file,
                shorts_analysis_path="shorts_analysis.json",
                mic_transcriptions=mic_transcriptions_raw,      # Use RAW - closer to actual speech timing
                desktop_transcriptions=desktop_transcriptions_raw  # Use RAW - closer to actual speech timing
            )

            if title_details:
                suggested_title, _, _ = title_details
                if title_update_callback:
                    title_update_callback(suggested_title)
                log_func(f"✅ Generated title: '{suggested_title}'")
            else:
                log_func("⚠️ Title generation returned no details.")

            # --- PHASE 3: Intelligent Trimming Analysis (Context-Aware) ---
            log_func("\n--- PHASE 3: Intelligent Trimming Analysis (Context-Aware) ---")
            log_func("   Using RAW timestamps - Gemini needs approximate actual timing, not display timing")
            
            if enable_trimming:
                trimmer = IntelligentTrimmer(log_func=log_func)
                
                # Analyze with RAW dialogue context
                # Gemini needs to understand WHEN things were said relative to video events
                # Raw timestamps are Whisper's best guess at actual audio timing
                trim_segments = trimmer.analyze_for_trim(
                    video_path=input_file,
                    title_details=title_details,
                    mic_transcriptions=mic_transcriptions_raw,      # Use RAW for actual timing context
                    desktop_transcriptions=desktop_transcriptions_raw  # Use RAW for actual timing context
                )
                
                if trim_segments:
                    total_kept_duration = sum(end - start for start, end in trim_segments)
                    log_func(f"✅ Trim plan ready: {len(trim_segments)} segments, {total_kept_duration:.1f}s total")
                    log_func("   (Actual trimming will occur after subtitle processing)")
                else:
                    log_func("⚠️ No trim decisions made, will keep original video")
            else:
                log_func("   Trimming disabled - skipping analysis")

            # Use original video for all processing
            video_to_process = input_file
            video_duration = get_video_duration(video_to_process, log_func)
            
            # --- PHASE 4: Onomatopoeia Detection ---
            log_func("\n--- PHASE 4: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = '.ass' if animation_type != "Static" else '.srt'
            onomatopoeia_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_ono{subtitle_ext}")
            success, onomatopoeia_events, video_analysis_map = detector.create_subtitle_file(
                input_path=video_to_process, output_path=onomatopoeia_subtitle_path, animation_type=animation_type
            )
            if not success: 
                onomatopoeia_subtitle_path = None
            del detector
            gc.collect()

            # --- PHASE 5: AI Director Editing ---
            log_func("\n--- PHASE 5: AI Director Editing ---")
            log_func("   Using RAW mic transcriptions for content analysis")
            
            director = MasterDirector(log_func=log_func, detailed_logs=detailed_logs)
            decision_timeline = director.analyze_video_and_create_timeline(
                video_path=video_to_process, 
                video_duration=video_duration, 
                mic_transcription=mic_transcriptions_raw,  # Use RAW for timing context
                audio_events=onomatopoeia_events, 
                video_analysis_map=video_analysis_map
            )

            video_to_subtitle = video_to_process
            edited_video_path = None
            if decision_timeline:
                editor = VideoEditor(log_func=log_func)
                edited_video_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_edited.mp4")
                editor.apply_edits(input_video=video_to_process, output_video=edited_video_path, timeline=decision_timeline)
                video_to_subtitle = edited_video_path

            # --- PHASE 6: Embedding All Subtitles ---
            log_func("\n--- PHASE 6: Embedding All Subtitles ---")
            log_func("   Embedding ADJUSTED subtitles (optimized for display)")
            
            # Create intermediate file for subtitles
            intermediate_with_subs = os.path.join(temp_dir, f"{os.path.basename(input_file)}_with_subs.mp4")
            
            embed_subtitles(
                input_video=video_to_subtitle, 
                output_video=intermediate_with_subs,
                track2_srt=mic_subtitle_path,  # These were created from ADJUSTED timestamps
                track3_srt=desktop_subtitle_path,  # These were created from ADJUSTED timestamps
                onomatopoeia_srt=onomatopoeia_subtitle_path, 
                onomatopoeia_events=onomatopoeia_events, 
                log=log_func
            )
            log_func(f"✅ Subtitles embedded successfully")

            # --- PHASE 7: Execute Intelligent Trimming ---
            log_func("\n--- PHASE 7: Execute Intelligent Trimming ---")
            
            if enable_trimming and trim_segments:
                log_func("   Applying trim plan to video with embedded subtitles...")
                log_func(f"   Using RAW transcriptions for dialogue protection (conservative buffer)")
                log_func(f"   Extending trim segments with {DIALOGUE_TRIM_BUFFER}s buffer...")
                
                # Use RAW for buffer extension - more conservative
                # If Whisper says dialogue at 45s but it's really at 50s,
                # we want to protect the 45-50s range to be safe
                extended_trim_segments = extend_segments_for_dialogue(
                    trim_segments, 
                    mic_transcriptions_raw,  # Use RAW for conservative dialogue protection
                    log_func,
                    max_extension_seconds=4.0,
                    buffer_seconds=DIALOGUE_TRIM_BUFFER
                )
                log_func(f"   Original segments: {trim_segments}")
                log_func(f"   Buffered segments: {extended_trim_segments}")
                
                trimmer = IntelligentTrimmer(log_func=log_func)
                
                trim_success = trimmer.apply_trim(
                    input_video=intermediate_with_subs,
                    output_video=output_file,
                    segments_to_keep=extended_trim_segments
                )
                
                if trim_success:
                    final_duration = get_video_duration(output_file, log_func)
                    log_func(f"✅ Trim applied successfully - Final duration: {final_duration:.1f}s")
                else:
                    log_func("⚠️ Trim execution failed, using untrimmed video")
                    shutil.copy2(intermediate_with_subs, output_file)
            else:
                log_func("   No trimming - using video with subtitles as-is")
                shutil.copy2(intermediate_with_subs, output_file)

            # --- PHASE 8: Final Renaming ---
            log_func("\n--- PHASE 8: Final Renaming ---")
            if suggested_title:
                filename = title_generator.title_to_filename(suggested_title)
                output_dir = os.path.dirname(output_file)
                new_output_path = os.path.join(output_dir, f"{filename}.mp4")
                
                if os.path.exists(output_file):
                    os.rename(output_file, new_output_path)
                    final_output_path = new_output_path
                    log_func(f"📁 Renamed final video to: {os.path.basename(final_output_path)}")
                else:
                    log_func(f"⚠️ Could not find processed file {output_file} to rename.")
            else:
                log_func("No title was generated, keeping original filename.")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file) and os.path.exists(input_file):
                shutil.copy2(input_file, output_file)
        finally:
            log_func("\n--- Cleaning up temporary files ---")
            temp_files_to_clean = [
                onomatopoeia_subtitle_path, 
                mic_subtitle_path, 
                desktop_subtitle_path,
                edited_video_path,
                intermediate_with_subs
            ]
            for path in temp_files_to_clean:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        log_func(f"🗑️ Removed temp file: {os.path.basename(path)}")
                    except Exception as e:
                        log_func(f"Warning: Could not clean up {path}: {e}")

        return final_output_path, suggested_title