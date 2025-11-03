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

class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        detailed_logs: bool,
        log_func,
        title_update_callback=None,
        enable_trimming: bool = True  # New parameter
    ):
        temp_dir = tempfile.gettempdir()
        final_output_path = output_file
        suggested_title = None
        trim_segments = None  # Store trim decisions for later

        try:
            log_func("="*60)
            log_func(f"STARTING FULL VIDEO PROCESSING: {os.path.basename(input_file)}")
            log_func("="*60)

            # --- PHASE 1: AI Title Generation ---
            log_func("\n--- PHASE 1: AI Title Generation ---")
            title_generator = TitleGenerator(log_func=log_func)
            title_details = title_generator.generate_title(
                video_path=input_file,
                shorts_analysis_path="shorts_analysis.json"
            )

            if title_details:
                suggested_title, _, _ = title_details
                if title_update_callback:
                    title_update_callback(suggested_title)
            else:
                log_func("⚠️  Title generation returned no details.")

            # --- PHASE 2: Analyze for Intelligent Clip Trimming ---
            log_func("\n--- PHASE 2: Analyze Trimming Decisions (Execution Delayed) ---")
            
            if enable_trimming:
                trimmer = IntelligentTrimmer(log_func=log_func)
                
                # Only analyze, don't actually cut yet
                trim_segments = trimmer.analyze_for_trim(
                    video_path=input_file,
                    title_details=title_details
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
            
            # --- PHASE 3: Onomatopoeia Detection ---
            log_func("\n--- PHASE 3: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = '.ass' if animation_type != "Static" else '.srt'
            onomatopoeia_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_ono{subtitle_ext}")
            success, onomatopoeia_events, video_analysis_map = detector.create_subtitle_file(
                input_path=video_to_process, output_path=onomatopoeia_subtitle_path, animation_type=animation_type
            )
            if not success: onomatopoeia_subtitle_path = None
            del detector
            gc.collect()

            # --- PHASE 4: Dialogue Transcription ---
            log_func("\n--- PHASE 4: Dialogue Transcription ---")
            mic_subtitle_path, desktop_subtitle_path = None, None
            mic_transcriptions_list = []
            
            mic_audio_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_mic.wav")
            if core.transcriber.convert_to_audio(video_to_process, mic_audio_path, track_index="a:1"):
                mic_transcriptions_list = core.transcriber.transcribe_audio("large", "cpu", mic_audio_path, True, log_func, "English", "Track 2 (Mic)")
                mic_subtitle_path_srt = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_mic.srt")
                convert_to_srt("\n".join(mic_transcriptions_list), mic_subtitle_path_srt, video_to_process, log_func, is_mic_track=True)
                mic_subtitle_path = mic_subtitle_path_srt.replace(".srt", ".ass")
                os.remove(mic_audio_path)

            desktop_audio_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_desktop.wav")
            if core.transcriber.convert_to_audio(video_to_process, desktop_audio_path, track_index="a:2"):
                desktop_transcriptions = core.transcriber.transcribe_audio("large", "cpu", desktop_audio_path, True, log_func, "English", "Track 3 (Desktop)")
                desktop_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_desktop.srt")
                convert_to_srt("\n".join(desktop_transcriptions), desktop_subtitle_path, video_to_process, log_func)
                os.remove(desktop_audio_path)

            # --- PHASE 5: AI Director Editing ---
            log_func("\n--- PHASE 5: AI Director Editing ---")
            director = MasterDirector(log_func=log_func, detailed_logs=detailed_logs)
            decision_timeline = director.analyze_video_and_create_timeline(
                video_path=video_to_process, 
                video_duration=video_duration, 
                mic_transcription=mic_transcriptions_list,
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
            
            # Create intermediate file for subtitles
            intermediate_with_subs = os.path.join(temp_dir, f"{os.path.basename(input_file)}_with_subs.mp4")
            
            embed_subtitles(
                input_video=video_to_subtitle, 
                output_video=intermediate_with_subs,
                track2_srt=mic_subtitle_path, 
                track3_srt=desktop_subtitle_path,
                onomatopoeia_srt=onomatopoeia_subtitle_path, 
                onomatopoeia_events=onomatopoeia_events, 
                log=log_func
            )
            log_func(f"✅ Subtitles embedded successfully")

            # --- PHASE 6.5: Execute Intelligent Trimming ---
            log_func("\n--- PHASE 6.5: Execute Intelligent Trimming ---")
            
            if enable_trimming and trim_segments:
                log_func("   Applying trim plan to video with embedded subtitles...")
                
                # Need to reference trimmer again (it was created in Phase 2)
                # or just call the method directly
                trimmer = IntelligentTrimmer(log_func=log_func)  # Add this line
                
                # Apply trim to the video with subtitles
                trim_success = trimmer.apply_trim(
                    input_video=intermediate_with_subs,
                    output_video=output_file,
                    segments_to_keep=trim_segments
                )
                
                if trim_success:
                    final_duration = get_video_duration(output_file, log_func)
                    log_func(f"✅ Trim applied successfully - Final duration: {final_duration:.1f}s")
                else:
                    log_func("⚠️ Trim execution failed, using untrimmed video")
                    shutil.copy2(intermediate_with_subs, output_file)
            else:
                # No trimming, just copy the subtitled video
                log_func("   No trimming - using video with subtitles as-is")
                shutil.copy2(intermediate_with_subs, output_file)

            # --- PHASE 7: Final Renaming ---
            log_func("\n--- PHASE 7: Final Renaming ---")
            if suggested_title:
                filename = title_generator.title_to_filename(suggested_title)
                output_dir = os.path.dirname(output_file)
                new_output_path = os.path.join(output_dir, f"{filename}.mp4")
                
                if os.path.exists(output_file):
                    os.rename(output_file, new_output_path)
                    final_output_path = new_output_path
                    log_func(f"📁 Renamed final video to: {os.path.basename(final_output_path)}")
                else:
                    log_func(f"⚠️  Could not find processed file {output_file} to rename.")
            else:
                log_func("No title was generated, keeping original filename.")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file) and os.path.exists(input_file):
                shutil.copy2(input_file, output_file)
        finally:
            # --- Cleanup ---
            log_func("\n--- Cleaning up temporary files ---")
            temp_files_to_clean = [
                onomatopoeia_subtitle_path, 
                mic_subtitle_path, 
                desktop_subtitle_path,
                edited_video_path,
                intermediate_with_subs  # Clean up intermediate file
            ]
            for path in temp_files_to_clean:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        log_func(f"🗑️ Removed temp file: {os.path.basename(path)}")
                    except Exception as e:
                        log_func(f"Warning: Could not clean up {path}: {e}")

        return final_output_path, suggested_title