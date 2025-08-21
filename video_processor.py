# video_processor.py

import os
import subprocess
import shutil

# Import the new main orchestrator
from onomatopoeia_detector import OnomatopoeiaDetector

class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        model_name: str,       # Kept for compatibility with UI
        device_name: str,
        ai_sensitivity: float,
        animation_type: str,
        log
    ):
        """
        This function is called by the UI. It now initializes and runs the
        complete multimodal onomatopoeia detection pipeline.
        """
        try:
            log("="*60)
            log(f"INITIALIZING VIDEO PROCESSOR FOR: {os.path.basename(input_file)}")
            log(f"Animation: {animation_type}, AI Sensitivity: {ai_sensitivity}")
            log("="*60)

            # 1. Initialize our powerful detector with settings from the UI
            log("Initializing the multimodal onomatopoeia detector...")
            detector = OnomatopoeiaDetector(
                sensitivity=ai_sensitivity,
                device=device_name,
                log_func=log
            )

            # 2. Define the output path for the subtitle file (.ass or .srt)
            subtitle_output_ext = '.ass' if animation_type != "Static" else '.srt'
            subtitle_path = os.path.splitext(output_file)[0] + subtitle_output_ext

            # 3. Run the entire pipeline to create the subtitle file
            log("Starting the main analysis and subtitle generation pipeline...")
            success, events = detector.create_subtitle_file(
                input_path=input_file,
                output_path=subtitle_path,
                animation_type=animation_type
            )

            if not success or not os.path.exists(subtitle_path):
                log("WARNING: Onomatopoeia detection did not produce a subtitle file. Copying original video.")
                shutil.copy2(input_file, output_file)
                return

            # 4. Embed the generated subtitles into the final video
            log(f"Embedding generated '{os.path.basename(subtitle_path)}' into final video...")

            # Use the appropriate ffmpeg filter based on file type
            if subtitle_output_ext == '.ass':
                subtitle_path_fmt = subtitle_path.replace('\\', '/').replace(':', r'\\:')
                vf_command = f"ass='{subtitle_path_fmt}'"
            else: # .srt
                subtitle_path_fmt = subtitle_path.replace('\\', '/').replace(':', r'\\:')
                vf_command = f"subtitles='{subtitle_path_fmt}'"

            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-vf', vf_command,
                '-c:a', 'copy',
                output_file
            ]

            log(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                log(f"✅ Successfully embedded subtitles into: {output_file}")
            else:
                log(f"❌ FFmpeg Error during embedding: {result.stderr}")
                log("Copying original video as a fallback.")
                shutil.copy2(input_file, output_file)

        except Exception as e:
            log(f"FATAL ERROR in VideoProcessor: {e}")
            import traceback
            log(f"Traceback: {traceback.format_exc()}")
            shutil.copy2(input_file, output_file)