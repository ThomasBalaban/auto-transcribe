# core/video_processor.py
import os
import shutil
import tempfile
import gc
import datetime

from onomatopoeia_detector import OnomatopoeiaDetector
import core.transcriber
from core.subtitle_converter import convert_to_srt
from core.subtitle_embedder import embed_subtitles
from ai_director.master_director import MasterDirector
from ai_director.video_editor import VideoEditor
from video_utils import get_video_duration
from clip_editor.intelligent_trimmer import IntelligentTrimmer
from utils.timestamp_processor import extend_segments_for_dialogue


class VideoProcessor:
    """
    Full per-video processing pipeline.

    Phases:
      1. Dialogue transcription (mic + game tracks)
      2. Intelligent trim analysis          (Gemini 3.1 Pro, high thinking)
      3. Onomatopoeia detection             (audio + Gemini 3 Flash vision)
      4. AI Director edit decisions         (zoom choreography)
      5. Subtitle embedding                 (ffmpeg)
      6. Execute intelligent trim           (ffmpeg)

    Title generation was moved out to a separate tool and is no longer
    part of this pipeline.
    """

    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        animation_type: str,
        sync_offset: float,
        detailed_logs: bool,
        log_func,
        enable_trimming: bool = True,
    ):
        temp_dir = tempfile.gettempdir()
        trim_segments = None
        decision_timeline = None
        events = []
        video_metadata = None
        DIALOGUE_TRIM_BUFFER = 0.8

        # Holders for cleanup
        mic_audio_path_for_analysis = None
        desktop_audio_path_for_analysis = None
        mic_subtitle_path = None
        desktop_subtitle_path = None
        onomatopoeia_subtitle_path = None
        edited_video_path = None
        intermediate_with_subs = None

        try:
            log_func("=" * 60)
            log_func(
                f"STARTING FULL VIDEO PROCESSING: "
                f"{os.path.basename(input_file)}"
            )
            log_func("=" * 60)

            # ── PHASE 1: Dialogue Transcription ────────────────────────
            log_func("\n--- PHASE 1: Dialogue Transcription ---")

            mic_transcriptions_raw = []
            mic_transcriptions_adjusted = []
            desktop_transcriptions_raw = []
            desktop_transcriptions_adjusted = []

            # Mic track
            mic_audio_path = os.path.join(
                temp_dir, f"{os.path.basename(input_file)}_mic.wav")
            if core.transcriber.convert_to_audio(
                input_file, mic_audio_path, "a:1", log_func
            ):
                (
                    mic_transcriptions_raw,
                    mic_transcriptions_adjusted,
                ) = core.transcriber.transcribe_audio(
                    "large", "cpu", mic_audio_path, True, log_func,
                    "English", "Track 2 (Mic)",
                )
                mic_subtitle_path_srt = os.path.join(
                    temp_dir, f"{os.path.basename(input_file)}_mic.srt")
                convert_to_srt(
                    "\n".join(mic_transcriptions_adjusted),
                    mic_subtitle_path_srt,
                    input_file,
                    log_func,
                    is_mic_track=True,
                )
                # Animated mic-track output is ASS, not SRT
                mic_subtitle_path = mic_subtitle_path_srt.replace(
                    ".srt", ".ass")

                try:
                    os.remove(mic_audio_path)
                except Exception:
                    pass
                log_func(
                    f"✅ Mic transcription complete: "
                    f"{len(mic_transcriptions_raw)} words"
                )
            else:
                log_func("⚠️ Mic audio extraction failed")

            # Desktop track
            desktop_audio_path = os.path.join(
                temp_dir, f"{os.path.basename(input_file)}_desktop.wav")
            if core.transcriber.convert_to_audio(
                input_file, desktop_audio_path, "a:2", log_func
            ):
                (
                    desktop_transcriptions_raw,
                    desktop_transcriptions_adjusted,
                ) = core.transcriber.transcribe_audio(
                    "large", "cpu", desktop_audio_path, True, log_func,
                    "English", "Track 3 (Desktop)",
                )
                desktop_subtitle_path = os.path.join(
                    temp_dir, f"{os.path.basename(input_file)}_desktop.srt")
                convert_to_srt(
                    "\n".join(desktop_transcriptions_adjusted),
                    desktop_subtitle_path,
                    input_file,
                    log_func,
                )
                try:
                    os.remove(desktop_audio_path)
                except Exception:
                    pass
                log_func(
                    f"✅ Desktop transcription complete: "
                    f"{len(desktop_transcriptions_raw)} words"
                )
            else:
                log_func("⚠️ Desktop audio extraction failed")

            # ── PHASE 2: Intelligent Trimming Analysis ─────────────────
            log_func("\n--- PHASE 2: Intelligent Trimming Analysis ---")
            if enable_trimming:
                trimmer = IntelligentTrimmer(log_func=log_func)
                trim_segments = trimmer.analyze_for_trim(
                    video_path=input_file,
                    mic_transcriptions=mic_transcriptions_raw,
                    desktop_transcriptions=desktop_transcriptions_raw,
                )
                if trim_segments:
                    total_kept = sum(e - s for s, e in trim_segments)
                    log_func(
                        f"✅ Trim plan ready: {len(trim_segments)} segments, "
                        f"{total_kept:.1f}s total"
                    )
                else:
                    log_func(
                        "⚠️ No trim decisions made, will keep original video")
            else:
                log_func("   Trimming disabled - skipping analysis")

            video_to_process = input_file
            video_duration = get_video_duration(video_to_process, log_func)

            # ── PHASE 3: Onomatopoeia Detection ────────────────────────
            log_func("\n--- PHASE 3: Onomatopoeia Detection ---")
            detector = OnomatopoeiaDetector(log_func=log_func)
            subtitle_ext = ".ass" if animation_type != "Static" else ".srt"
            onomatopoeia_subtitle_path = os.path.join(
                temp_dir,
                f"{os.path.basename(video_to_process)}_ono{subtitle_ext}",
            )
            detector.fusion_engine.sync_offset = sync_offset
            events, video_map = detector.analyze_file(
                input_file, animation_type, sync_offset=sync_offset)
            detector.subtitle_generator.create_subtitle_file(
                events, onomatopoeia_subtitle_path, animation_type)
            del detector
            gc.collect()

            # ── PHASE 4: AI Director Editing ───────────────────────────
            log_func("\n--- PHASE 4: AI Director Editing ---")
            director = MasterDirector(
                log_func=log_func, detailed_logs=detailed_logs)
            decision_timeline = director.analyze_video_and_create_timeline(
                video_path=video_to_process,
                video_duration=video_duration,
                mic_transcription=mic_transcriptions_raw,
                audio_events=events,
                video_analysis_map=video_map,
            )

            video_to_subtitle = video_to_process
            if decision_timeline:
                editor = VideoEditor(log_func=log_func)
                edited_video_path = os.path.join(
                    temp_dir,
                    f"{os.path.basename(video_to_process)}_edited.mp4",
                )
                editor.apply_edits(
                    input_video=video_to_process,
                    output_video=edited_video_path,
                    timeline=decision_timeline,
                )
                video_to_subtitle = edited_video_path

            # ── PHASE 5: Embed All Subtitles ───────────────────────────
            log_func("\n--- PHASE 5: Embedding All Subtitles ---")
            intermediate_with_subs = os.path.join(
                temp_dir, f"{os.path.basename(input_file)}_with_subs.mp4")
            embed_subtitles(
                input_video=video_to_subtitle,
                output_video=intermediate_with_subs,
                track2_srt=mic_subtitle_path,
                track3_srt=desktop_subtitle_path,
                onomatopoeia_srt=onomatopoeia_subtitle_path,
                onomatopoeia_events=events,
                log=log_func,
            )
            log_func("✅ Subtitles embedded successfully")

            # ── PHASE 6: Execute Intelligent Trimming ──────────────────
            log_func("\n--- PHASE 6: Execute Intelligent Trimming ---")
            if enable_trimming and trim_segments:
                log_func("   Applying trim plan...")

                mic_audio_path_for_analysis = os.path.join(
                    temp_dir,
                    f"{os.path.basename(input_file)}_mic_analysis.wav",
                )
                core.transcriber.convert_to_audio(
                    input_file,
                    mic_audio_path_for_analysis,
                    "a:1",
                    log_func,
                )

                if desktop_transcriptions_raw:
                    desktop_audio_path_for_analysis = os.path.join(
                        temp_dir,
                        f"{os.path.basename(input_file)}_desktop_analysis.wav",
                    )
                    core.transcriber.convert_to_audio(
                        input_file,
                        desktop_audio_path_for_analysis,
                        "a:2",
                        log_func,
                    )

                extended_trim_segments = extend_segments_for_dialogue(
                    segments_to_keep=trim_segments,
                    raw_mic_transcriptions=mic_transcriptions_raw,
                    raw_desktop_transcriptions=desktop_transcriptions_raw,
                    log_func=log_func,
                    max_extension_seconds=3.0,
                    buffer_seconds=DIALOGUE_TRIM_BUFFER,
                    mic_audio_path=mic_audio_path_for_analysis,
                    desktop_audio_path=desktop_audio_path_for_analysis,
                )

                trimmer = IntelligentTrimmer(log_func=log_func)
                trim_success = trimmer.apply_trim(
                    input_video=intermediate_with_subs,
                    output_video=output_file,
                    segments_to_keep=extended_trim_segments,
                )

                if trim_success:
                    final_duration = get_video_duration(
                        output_file, log_func)
                    log_func(
                        f"✅ Trim applied successfully — "
                        f"Final: {final_duration:.1f}s"
                    )
                else:
                    log_func("⚠️ Trim execution failed, using untrimmed")
                    shutil.copy2(intermediate_with_subs, output_file)
            else:
                shutil.copy2(intermediate_with_subs, output_file)

            # ── PHASE 7: Metadata (no rename, no title) ────────────────
            video_metadata = {
                "file_info": {
                    "original_filename": os.path.basename(input_file),
                    "output_filename": os.path.basename(output_file),
                    "processed_at": datetime.datetime.now().isoformat(),
                    "original_duration": video_duration,
                    "final_duration": (
                        get_video_duration(output_file, log_func)
                        if os.path.exists(output_file) else None
                    ),
                },
            }
            log_func("✅ Metadata generated (queued for batch file)")

        except Exception as e:
            log_func(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
            if not os.path.exists(output_file) and os.path.exists(input_file):
                shutil.copy2(input_file, output_file)
        finally:
            log_func("\n--- Cleaning up temporary files ---")
            for p in (
                onomatopoeia_subtitle_path,
                mic_subtitle_path,
                desktop_subtitle_path,
                edited_video_path,
                intermediate_with_subs,
                mic_audio_path_for_analysis,
                desktop_audio_path_for_analysis,
            ):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        # Preserve the existing 3-tuple return signature (callers in main.py
        # and api_server.py already ignore the middle "title" slot when we
        # pass None).
        return output_file, None, video_metadata