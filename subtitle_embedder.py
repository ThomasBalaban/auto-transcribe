import os
import subprocess


def embed_dual_subtitles(input_video, output_video, track2_srt, track3_srt, log):
    """Embed two subtitle tracks into a video file with different positions"""
    log(f"Embedding dual subtitles into video...")
    input_video_orig = input_video  # Keep track of original input

    # Validate subtitle files exist
    if not os.path.exists(track2_srt) or not os.path.exists(track3_srt):
        missing = [f for f in (track2_srt, track3_srt) if not os.path.exists(f)]
        log(f"ERROR: Subtitle file(s) do not exist: {', '.join(missing)}")
        raise FileNotFoundError(f"Subtitle file(s) not found: {', '.join(missing)}")

    # Validate subtitle files are not empty
    if os.path.getsize(track2_srt) == 0 or os.path.getsize(track3_srt) == 0:
        empty = [f for f in (track2_srt, track3_srt) if os.path.getsize(f) == 0]
        log(f"ERROR: Subtitle file(s) are empty: {', '.join(empty)}")
        raise ValueError(f"Subtitle file(s) are empty: {', '.join(empty)}")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log(f"Created output directory: {out_dir}")

    # Prepare paths for ffmpeg (Windows escaping)
    track2_srt_fmt = track2_srt.replace('\\', '/')
    track3_srt_fmt = track3_srt.replace('\\', '/')
    if os.name == 'nt':
        track2_srt_fmt = track2_srt_fmt.replace(':', r'\\:')
        track3_srt_fmt = track3_srt_fmt.replace(':', r'\\:')

    # Convert MKV to MP4 first if necessary
    temp_mp4 = None
    if input_video.lower().endswith('.mkv'):
        temp_mp4 = f"{os.path.splitext(output_video)[0]}_temp.mp4"
        log("Input is MKV format. Converting to MP4 first...")
        cmd = [
            'ffmpeg', '-i', input_video,
            '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
            temp_mp4
        ]
        log(f"Running conversion command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        input_video = temp_mp4  # Use converted file for embedding

    # Define styles for subtitle tracks
    style2 = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=1,Italic=0,BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=60"
    )
    style3 = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=1,Italic=0,BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=20"
    )

    # Embed the subtitles into the video
    cmd = [
        'ffmpeg', '-i', input_video,
        '-vf',
        f"subtitles='{track2_srt_fmt}':force_style='{style2}',subtitles='{track3_srt_fmt}':force_style='{style3}'",
        '-c:a', 'copy', output_video
    ]
    log(f"Running embedding command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    log(f"Dual subtitles embedded into {output_video} successfully\n")

    # Clean up temp MP4 file if created
    if temp_mp4 and os.path.exists(temp_mp4):
        try:
            os.remove(temp_mp4)
            log(f"Deleted temporary file: {temp_mp4}")
        except Exception as e:
            log(f"WARNING: Failed to delete temporary file {temp_mp4}: {e}")
