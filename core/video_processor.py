# core/video_processor.py - UPDATED
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

class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        detailed_logs: bool,
        log_func
    ):
        temp_dir = tempfile.gettempdir()
        onomatopoeia_subtitle_path = None
        mic_subtitle_path = None
        desktop_subtitle_path = None
        onomatopoeia_events = []
        video_analysis_map = {} # To store the vision analysis
        mic_transcriptions_list = []
        mic_subtitle_path_srt = None
        edited_video_path = None

        try:
            log_func("="*60)
            log_func(f"STARTING FULL VIDEO PROCESSING: {os.path.basename(input_file)}")
            log_func("="*60)
            
            video_duration = get_video_duration(input_file, log_func)

            # --- 1. ONOMATOPOEIA DETECTION ---
            log_func("\n--- PHASE 1: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = '.ass' if animation_type != "Static" else '.srt'
            onomatopoeia_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_ono{subtitle_ext}")
            
            # Now captures the video_analysis_map
            success, onomatopoeia_events, video_analysis_map = detector.create_subtitle_file(
                input_path=input_file,
                output_path=onomatopoeia_subtitle_path,
                animation_type=animation_type
            )
            if success:
                 log_func(f"Vision analysis captured for {len(video_analysis_map)} events.")
            else:
                log_func("WARNING: Onomatopoeia detection failed or produced no events.")
                onomatopoeia_subtitle_path = None
            
            log_func("INFO: Releasing onomatopoeia detector resources...")
            del detector
            gc.collect()
            log_func("INFO: Resources released.")

            # --- 2. DIALOGUE TRANSCRIPTION ---
            # This phase remains the same
            log_func("\n--- PHASE 2: Dialogue Transcription ---")
            mic_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            if core.transcriber.convert_to_audio(input_file, mic_audio_path, track_index="a:1"):
                mic_transcriptions_list = core.transcriber.transcribe_audio("large", "cpu", mic_audio_path, True, log_func, "English", "Track 2 (Mic)")
                mic_subtitle_path_srt = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt("\n".join(mic_transcriptions_list), mic_subtitle_path_srt, input_file, log_func, is_mic_track=True)
                mic_subtitle_path = mic_subtitle_path_srt.replace(".srt", ".ass")
                os.remove(mic_audio_path)
            desktop_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            if core.transcriber.convert_to_audio(input_file, desktop_audio_path, track_index="a:2"):
                desktop_transcriptions = core.transcriber.transcribe_audio("large", "cpu", desktop_audio_path, True, log_func, "English", "Track 3 (Desktop)")
                desktop_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt("\n".join(desktop_transcriptions), desktop_subtitle_path, input_file, log_func)
                os.remove(desktop_audio_path)

            # --- 3. AI DIRECTOR ANALYSIS & EDITING ---
            log_func("\n--- PHASE 3: AI Director Editing ---")
            director = MasterDirector(log_func=log_func, detailed_logs=detailed_logs)
            
            decision_timeline = director.analyze_video_and_create_timeline(
                video_path=input_file,
                video_duration=video_duration,
                mic_transcription=mic_transcriptions_list,
                audio_events=onomatopoeia_events, # Use onomatopoeia events as audio triggers
                video_analysis_map=video_analysis_map # Pass the cached analysis
            )

            video_to_subtitle = input_file
            if decision_timeline:
                editor = VideoEditor(log_func=log_func)
                edited_video_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_edited.mp4")
                editor.apply_edits(
                    input_video=input_file,
                    output_video=edited_video_path,
                    timeline=decision_timeline
                )
                video_to_subtitle = edited_video_path
                log_func(f"✅ AI Director edits applied. Intermediate video created: {edited_video_path}")
            else:
                log_func("No AI Director edits were made. Proceeding with original video.")

            # --- 4. EMBED SUBTITLES ---
            log_func("\n--- PHASE 4: Embedding All Subtitles ---")
            embed_subtitles(
                input_video=video_to_subtitle,
                output_video=output_file,
                track2_srt=mic_subtitle_path,
                track3_srt=desktop_subtitle_path,
                onomatopoeia_srt=onomatopoeia_subtitle_path,
                onomatopoeia_events=onomatopoeia_events,
                log=log_func
            )
            log_func(f"✅ Successfully created final video with all subtitles: {output_file}")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file):
                shutil.copy2(input_file, output_file)
        finally:
            # --- 5. CLEANUP ---
            log_func("\n--- Cleaning up temporary files ---")
            temp_files_to_clean = [
                onomatopoeia_subtitle_path, mic_subtitle_path, desktop_subtitle_path, 
                mic_subtitle_path_srt, edited_video_path
            ]
            for path in temp_files_to_clean:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        log_func(f"🗑️ Removed temp file: {path}")
                    except Exception as e:
                        log_func(f"Warning: Could not clean up {path}: {e}")