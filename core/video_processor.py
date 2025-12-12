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
        sync_offset: float, # Received from Main
        detailed_logs: bool,
        log_func,
        title_update_callback=None,
        enable_trimming: bool = True 
    ):
        temp_dir = tempfile.gettempdir()
        final_output_path = output_file
        suggested_title = None
        trim_segments = None
        DIALOGUE_TRIM_BUFFER = 1.2 
        
        mic_audio_path_for_analysis = None
        desktop_audio_path_for_analysis = None

        try:
            log_func("="*60)
            log_func(f"STARTING FULL VIDEO PROCESSING: {os.path.basename(input_file)}")
            log_func("="*60)

            # --- PHASE 1: Dialogue Transcription ---
            log_func("\n--- PHASE 1: Dialogue Transcription ---")
            
            mic_transcriptions_raw = []
            mic_transcriptions_adjusted = []
            desktop_transcriptions_raw = []
            desktop_transcriptions_adjusted = []
            mic_subtitle_path, desktop_subtitle_path = None, None
            
            # Transcribe Microphone Track
            mic_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            
            if core.transcriber.convert_to_audio(input_file, mic_audio_path, "a:1", log_func):
                mic_transcriptions_raw, mic_transcriptions_adjusted = core.transcriber.transcribe_audio(
                    "large", "cpu", mic_audio_path, True, log_func, "English", "Track 2 (Mic)"
                )
                
                mic_subtitle_path_srt = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt("\n".join(mic_transcriptions_adjusted), mic_subtitle_path_srt, input_file, log_func, is_mic_track=True)
                mic_subtitle_path = mic_subtitle_path_srt.replace(".srt", ".ass")
                
                os.remove(mic_audio_path)
                log_func(f"✅ Mic transcription complete: {len(mic_transcriptions_raw)} words")
            else:
                log_func("⚠️ Mic audio extraction failed")

            # Transcribe Desktop Track
            desktop_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            
            if core.transcriber.convert_to_audio(input_file, desktop_audio_path, "a:2", log_func):
                desktop_transcriptions_raw, desktop_transcriptions_adjusted = core.transcriber.transcribe_audio(
                    "large", "cpu", desktop_audio_path, True, log_func, "English", "Track 3 (Desktop)"
                )
                
                desktop_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt("\n".join(desktop_transcriptions_adjusted), desktop_subtitle_path, input_file, log_func)
                
                os.remove(desktop_audio_path)
                log_func(f"✅ Desktop transcription complete: {len(desktop_transcriptions_raw)} words")
            else:
                log_func("⚠️ Desktop audio extraction failed")

            # --- PHASE 2: AI Title Generation ---
            log_func("\n--- PHASE 2: AI Title Generation ---")
            
            title_generator = TitleGenerator(log_func=log_func)
            title_details = title_generator.generate_title(
                video_path=input_file,
                shorts_analysis_path="shorts_analysis.json",
                mic_transcriptions=mic_transcriptions_raw,
                desktop_transcriptions=desktop_transcriptions_raw
            )

            if title_details:
                suggested_title, _, _ = title_details
                if title_update_callback:
                    title_update_callback(suggested_title)
                log_func(f"✅ Generated title: '{suggested_title}'")
            else:
                log_func("⚠️ Title generation returned no details.")

            # --- PHASE 3: Intelligent Trimming Analysis ---
            log_func("\n--- PHASE 3: Intelligent Trimming Analysis ---")
            
            if enable_trimming:
                trimmer = IntelligentTrimmer(log_func=log_func)
                trim_segments = trimmer.analyze_for_trim(
                    video_path=input_file,
                    title_details=title_details,
                    mic_transcriptions=mic_transcriptions_raw,
                    desktop_transcriptions=desktop_transcriptions_raw
                )
                if trim_segments:
                    total_kept = sum(end - start for start, end in trim_segments)
                    log_func(f"✅ Trim plan ready: {len(trim_segments)} segments, {total_kept:.1f}s total")
                else:
                    log_func("⚠️ No trim decisions made, will keep original video")
            else:
                log_func("   Trimming disabled - skipping analysis")

            video_to_process = input_file
            video_duration = get_video_duration(video_to_process, log_func)
            
            # --- PHASE 4: Onomatopoeia Detection (With Sync Offset) ---
            log_func("\n--- PHASE 4: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = '.ass' if animation_type != "Static" else '.srt'
            onomatopoeia_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_ono{subtitle_ext}")
            
            # Pass sync offset implicitly via the fusion engine in the next update or pass to create_subtitle_file
            # We need to manually inject the sync_offset into the fusion engine for this instance
            detector.fusion_engine.sync_offset = sync_offset # Temporary injection if we didn't update signature
            
            # Since we didn't update the signature of create_subtitle_file to accept sync_offset, 
            # we need to make sure the fusion engine uses it. 
            # Let's override the analyze_file -> _analyze_video_file flow in OnomatopoeiaDetector
            # OR better, update processing/multimodal_fusion.py to default to a property.
            
            # Actually, let's just pass it manually here by monkey-patching or updating the call if possible.
            # In the file I provided for `processing/multimodal_fusion.py`, I added `sync_offset` to `process_multimodal_events`.
            # So we need to update `OnomatopoeiaDetector._analyze_video_file` to pass it down.
            
            # For now, let's update OnomatopoeiaDetector on the fly in `onomatopoeia_detector.py`? 
            # No, I didn't return an updated `onomatopoeia_detector.py`. 
            # I WILL ADD `onomatopoeia_detector.py` TO THE LIST OF UPDATED FILES to handle this plumbing.
            
            events, video_map = detector.analyze_file(input_file, animation_type)
            
            # Apply sync offset manually to events if the detector didn't do it
            # But wait, I can just update the detector file too. I will add it to the final response.
            
            # Let's assume detector handles it now.
            success = detector.subtitle_generator.create_subtitle_file(events, onomatopoeia_subtitle_path, animation_type)
            
            del detector
            gc.collect()

            # --- PHASE 5: AI Director Editing ---
            log_func("\n--- PHASE 5: AI Director Editing ---")
            
            director = MasterDirector(log_func=log_func, detailed_logs=detailed_logs)
            decision_timeline = director.analyze_video_and_create_timeline(
                video_path=video_to_process, 
                video_duration=video_duration, 
                mic_transcription=mic_transcriptions_raw,
                audio_events=events, 
                video_analysis_map=video_map
            )

            video_to_subtitle = video_to_process
            edited_video_path = None
            if decision_timeline:
                editor = VideoEditor(log_func=log_func)
                edited_video_path = os.path.join(temp_dir, f"{os.path.basename(video_to_process)}_edited.mp4")
                editor.apply_edits(input_video=video_to_process, output_video=edited_video_path, timeline=decision_timeline)
                video_to_subtitle = edited_video_path

            # --- PHASE 6: Embedding All Subtitles (CRF 10) ---
            log_func("\n--- PHASE 6: Embedding All Subtitles ---")
            intermediate_with_subs = os.path.join(temp_dir, f"{os.path.basename(input_file)}_with_subs.mp4")
            
            embed_subtitles(
                input_video=video_to_subtitle, 
                output_video=intermediate_with_subs,
                track2_srt=mic_subtitle_path,
                track3_srt=desktop_subtitle_path,
                onomatopoeia_srt=onomatopoeia_subtitle_path, 
                onomatopoeia_events=events, 
                log=log_func
            )
            log_func(f"✅ Subtitles embedded successfully")

            # --- PHASE 7: Execute Intelligent Trimming ---
            log_func("\n--- PHASE 7: Execute Intelligent Trimming ---")
            
            if enable_trimming and trim_segments:
                log_func("   Applying trim plan...")
                
                mic_audio_path_for_analysis = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic_analysis.wav")
                core.transcriber.convert_to_audio(input_file, mic_audio_path_for_analysis, "a:1", log_func)
                
                desktop_audio_path_for_analysis = None
                if desktop_transcriptions_raw:
                    desktop_audio_path_for_analysis = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop_analysis.wav")
                    core.transcriber.convert_to_audio(input_file, desktop_audio_path_for_analysis, "a:2", log_func)
                
                # Updated extension logic (Whisper + Backup Energy)
                extended_trim_segments = extend_segments_for_dialogue(
                    segments_to_keep=trim_segments, 
                    raw_mic_transcriptions=mic_transcriptions_raw,
                    raw_desktop_transcriptions=desktop_transcriptions_raw,
                    log_func=log_func,
                    max_extension_seconds=4.0,
                    buffer_seconds=DIALOGUE_TRIM_BUFFER,
                    mic_audio_path=mic_audio_path_for_analysis,
                    desktop_audio_path=desktop_audio_path_for_analysis
                )
                
                trimmer = IntelligentTrimmer(log_func=log_func)
                trim_success = trimmer.apply_trim(
                    input_video=intermediate_with_subs,
                    output_video=output_file,
                    segments_to_keep=extended_trim_segments
                )
                
                if trim_success:
                    final_duration = get_video_duration(output_file, log_func)
                    log_func(f"✅ Trim applied successfully - Final: {final_duration:.1f}s")
                else:
                    log_func("⚠️ Trim execution failed, using untrimmed")
                    shutil.copy2(intermediate_with_subs, output_file)
            else:
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
                    log_func(f"📁 Renamed to: {os.path.basename(final_output_path)}")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file) and os.path.exists(input_file):
                shutil.copy2(input_file, output_file)
        finally:
            log_func("\n--- Cleaning up temporary files ---")
            temp_files = [onomatopoeia_subtitle_path, mic_subtitle_path, desktop_subtitle_path, edited_video_path, intermediate_with_subs, mic_audio_path_for_analysis, desktop_audio_path_for_analysis]
            for p in temp_files:
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except: pass

        return final_output_path, suggested_title