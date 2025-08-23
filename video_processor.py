# video_processor.py

import os
import shutil
import tempfile

# Import the main onomatopoeia detector
from onomatopoeia_detector import OnomatopoeiaDetector
# Import transcription and subtitle modules
import transcriber
from subtitle_converter import convert_to_srt
from subtitle_embedder import embed_subtitles


class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        log_func
    ):
        """
        Main orchestrator for processing a single video.
        Automatically handles dialogue transcription and onomatopoeia detection.
        """
        temp_dir = tempfile.gettempdir()
        onomatopoeia_subtitle_path = None
        mic_subtitle_path = None
        desktop_subtitle_path = None
        onomatopoeia_events = []

        try:
            log_func("="*60)
            log_func(f"STARTING FULL VIDEO PROCESSING: {os.path.basename(input_file)}")
            log_func("="*60)

            # --- 1. ONOMATOPOEIA DETECTION (ALWAYS ON) ---
            log_func("\n--- PHASE 1: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = '.ass' if animation_type != "Static" else '.srt'
            onomatopoeia_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_ono{subtitle_ext}")

            success, onomatopoeia_events = detector.create_subtitle_file(
                input_path=input_file,
                output_path=onomatopoeia_subtitle_path,
                animation_type=animation_type
            )
            if not success:
                log_func("WARNING: Onomatopoeia detection failed or produced no events.")
                onomatopoeia_subtitle_path = None


            # --- 2. DIALOGUE TRANSCRIPTION (ALWAYS ON) ---
            log_func("\n--- PHASE 2: Dialogue Transcription ---")
            # Transcribe Microphone (Track 2)
            log_func("\n-- Transcribing Microphone Audio (Track 2) --")
            mic_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            if transcriber.convert_to_audio(input_file, mic_audio_path, track_index="a:1"):
                mic_transcriptions = transcriber.transcribe_audio("large", "cpu", mic_audio_path, True, log_func, "English", "Track 2 (Mic)")
                mic_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt("\n".join(mic_transcriptions), mic_subtitle_path, input_file, log_func, is_mic_track=True)
                os.remove(mic_audio_path)
            else:
                log_func("ERROR: Failed to extract microphone audio. Skipping.")

            # Transcribe Desktop (Track 3)
            log_func("\n-- Transcribing Desktop Audio (Track 3) --")
            desktop_audio_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            if transcriber.convert_to_audio(input_file, desktop_audio_path, track_index="a:2"):
                desktop_transcriptions = transcriber.transcribe_audio("large", "cpu", desktop_audio_path, True, log_func, "English", "Track 3 (Desktop)")
                desktop_subtitle_path = os.path.join(temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt("\n".join(desktop_transcriptions), desktop_subtitle_path, input_file, log_func)
                os.remove(desktop_audio_path)
            else:
                log_func("ERROR: Failed to extract desktop audio. Skipping.")


            # --- 3. EMBED SUBTITLES ---
            log_func("\n--- PHASE 3: Embedding All Subtitles ---")
            if not any([mic_subtitle_path, desktop_subtitle_path, onomatopoeia_subtitle_path]):
                log_func("WARNING: No subtitles were generated. Copying original video.")
                shutil.copy2(input_file, output_file)
                return

            embed_subtitles(
                input_video=input_file,
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
            shutil.copy2(input_file, output_file)
        finally:
            # --- 4. CLEANUP ---
            log_func("\n--- Cleaning up temporary files ---")
            for path in [onomatopoeia_subtitle_path, mic_subtitle_path, desktop_subtitle_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        log_func(f"🗑️ Removed temp file: {path}")
                    except Exception as e:
                        log_func(f"Warning: Could not clean up {path}: {e}")