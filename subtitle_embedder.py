import os
import subprocess
from subtitle_styles import TRACK2_STYLE, TRACK3_STYLE, OnomatopoeiaStyle

def embed_dual_subtitles(input_video, output_video, track2_srt, track3_srt, log):
    """Embed two subtitle tracks into a video file with different positions in a single pass"""
    log(f"Embedding dual subtitles into video...")
    
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

    # Use a single-pass approach with multiple subtitle filters
    cmd = [
        'ffmpeg', '-y', 
        '-i', input_video,
        '-vf', f"subtitles='{track2_srt_fmt}':force_style='{TRACK2_STYLE}',subtitles='{track3_srt_fmt}':force_style='{TRACK3_STYLE}'",
        '-c:a', 'copy', 
        output_video
    ]
    
    log(f"Running combined embedding command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    log(f"Dual subtitles embedded into {output_video} successfully\n")
        
    # Clean up temp MP4 file if created
    if temp_mp4 and os.path.exists(temp_mp4):
        try:
            os.remove(temp_mp4)
            log(f"Deleted temporary file: {temp_mp4}")
        except Exception as e:
            log(f"WARNING: Failed to delete temporary file {temp_mp4}: {e}")

def embed_single_subtitles(input_video, output_video, srt_file, log, is_mic_track=True):
    """Embed a single subtitle track into a video file with appropriate styling"""
    log(f"Embedding single subtitle track into video...")
    
    # Validate subtitle file exists and is not empty
    if not os.path.exists(srt_file):
        log(f"ERROR: Subtitle file does not exist: {srt_file}")
        raise FileNotFoundError(f"Subtitle file not found: {srt_file}")

    if os.path.getsize(srt_file) == 0:
        log(f"ERROR: Subtitle file is empty: {srt_file}")
        raise ValueError(f"Subtitle file is empty: {srt_file}")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log(f"Created output directory: {out_dir}")

    # Prepare paths for ffmpeg (Windows escaping)
    srt_fmt = srt_file.replace('\\', '/')
    if os.name == 'nt':
        srt_fmt = srt_fmt.replace(':', r'\\:')

    # Choose style based on track type
    style = TRACK2_STYLE if is_mic_track else TRACK3_STYLE
    track_type = "microphone" if is_mic_track else "desktop"
    
    log(f"Using {track_type} styling for single subtitle track")

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

    # Embed single subtitle track
    cmd = [
        'ffmpeg', '-y', 
        '-i', input_video,
        '-vf', f"subtitles='{srt_fmt}':force_style='{style}'",
        '-c:a', 'copy', 
        output_video
    ]
    
    log(f"Running single subtitle embedding command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    log(f"Single subtitle track embedded into {output_video} successfully\n")
        
    # Clean up temp MP4 file if created
    if temp_mp4 and os.path.exists(temp_mp4):
        try:
            os.remove(temp_mp4)
            log(f"Deleted temporary file: {temp_mp4}")
        except Exception as e:
            log(f"WARNING: Failed to delete temporary file {temp_mp4}: {e}")

def embed_triple_subtitles(input_video, output_video, track2_srt, track3_srt, onomatopoeia_srt, onomatopoeia_events, log):
    """Embed three subtitle tracks into a video file: mic, desktop, and onomatopoeia - STEP BY STEP"""
    log(f"Embedding triple subtitles into video using step-by-step approach...")
    
    # Validate all subtitle files exist
    subtitle_files = [track2_srt, track3_srt, onomatopoeia_srt]
    existing_files = [f for f in subtitle_files if f and os.path.exists(f) and os.path.getsize(f) > 0]
    
    if len(existing_files) < 3:
        missing = [f for f in subtitle_files if not f or not os.path.exists(f) or os.path.getsize(f) == 0]
        log(f"WARNING: Some subtitle files missing or empty: {missing}")
        # Fall back to dual subtitles if onomatopoeia is missing
        if track2_srt and track3_srt and os.path.exists(track2_srt) and os.path.exists(track3_srt):
            return embed_dual_subtitles(input_video, output_video, track2_srt, track3_srt, log)
        else:
            raise FileNotFoundError(f"Essential subtitle files not found")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log(f"Created output directory: {out_dir}")

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
        input_video = temp_mp4

    # STEP 1: First embed mic and desktop subtitles (we know this works)
    temp_dual_video = f"{os.path.splitext(output_video)[0]}_dual_temp.mp4"
    log("Step 1: Embedding mic and desktop subtitles...")
    
    # Prepare paths for ffmpeg (Windows escaping)
    track2_srt_fmt = track2_srt.replace('\\', '/')
    track3_srt_fmt = track3_srt.replace('\\', '/')
    if os.name == 'nt':
        track2_srt_fmt = track2_srt_fmt.replace(':', r'\\:')
        track3_srt_fmt = track3_srt_fmt.replace(':', r'\\:')

    # Embed dual subtitles first
    cmd_dual = [
        'ffmpeg', '-y', 
        '-i', input_video,
        '-vf', f"subtitles='{track2_srt_fmt}':force_style='{TRACK2_STYLE}',subtitles='{track3_srt_fmt}':force_style='{TRACK3_STYLE}'",
        '-c:a', 'copy', 
        temp_dual_video
    ]
    
    log(f"Running dual subtitle command: {' '.join(cmd_dual)}")
    subprocess.run(cmd_dual, check=True)
    log("✓ Dual subtitles embedded successfully")
    
    # STEP 2: Add onomatopoeia subtitles to the dual-subtitle video
    log("Step 2: Adding onomatopoeia subtitles...")
    
    # Debug: Check if dual video was created properly
    if not os.path.exists(temp_dual_video):
        raise FileNotFoundError(f"Dual subtitle video not created: {temp_dual_video}")
    
    dual_size = os.path.getsize(temp_dual_video)
    log(f"Dual subtitle video size: {dual_size} bytes")
    
    # Debug: Check onomatopoeia SRT file
    if not os.path.exists(onomatopoeia_srt):
        raise FileNotFoundError(f"Onomatopoeia SRT not found: {onomatopoeia_srt}")
    
    srt_size = os.path.getsize(onomatopoeia_srt)
    log(f"Onomatopoeia SRT size: {srt_size} bytes")
    
    # Debug: Read and display SRT content
    try:
        with open(onomatopoeia_srt, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        log(f"SRT content:\n{srt_content}")
    except Exception as e:
        log(f"Error reading SRT: {e}")
    
    onomatopoeia_srt_fmt = onomatopoeia_srt.replace('\\', '/')
    if os.name == 'nt':
        onomatopoeia_srt_fmt = onomatopoeia_srt_fmt.replace(':', r'\\:')
    
    # Use simple, reliable style - EXACTLY like the test that worked
    # onomatopoeia_style = "FontName=Arial,FontSize=64,PrimaryColour=&H0000FFFF,Bold=1,MarginV=200"
    onomatopoeia_style = OnomatopoeiaStyle.get_simple_style()
    log(f"Using onomatopoeia style: {onomatopoeia_style}")
    
    cmd_onomatopoeia = [
        'ffmpeg', '-y', 
        '-i', temp_dual_video,
        '-vf', f"subtitles='{onomatopoeia_srt_fmt}':force_style='{onomatopoeia_style}'",
        '-c:a', 'copy', 
        output_video
    ]
    
    log(f"Running onomatopoeia subtitle command: {' '.join(cmd_onomatopoeia)}")
    
    try:
        # Run with more detailed output
        result = subprocess.run(cmd_onomatopoeia, capture_output=True, text=True, check=True)
        log("✓ Onomatopoeia subtitles added successfully")
        
        # Debug: Check if output file was created and has reasonable size
        if os.path.exists(output_video):
            output_size = os.path.getsize(output_video)
            log(f"Final output video size: {output_size} bytes")
        else:
            log("ERROR: Final output video was not created!")
            
    except subprocess.CalledProcessError as e:
        log(f"ERROR in onomatopoeia embedding: {e}")
        log(f"FFmpeg stdout: {e.stdout}")
        log(f"FFmpeg stderr: {e.stderr}")
        raise
    
    log(f"Triple subtitles embedded into {output_video} successfully\n")
        
    # Clean up temp files
    temp_files = [temp_mp4, temp_dual_video]
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                log(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                log(f"WARNING: Failed to delete temporary file {temp_file}: {e}")

def embed_subtitles(input_video, output_video, track2_srt, track3_srt, onomatopoeia_srt, onomatopoeia_events, log):
    """
    Main subtitle embedding function - always tries to include onomatopoeia if available.
    """
    # Check what subtitle files we have
    has_track2 = track2_srt and os.path.exists(track2_srt) and os.path.getsize(track2_srt) > 0
    has_track3 = track3_srt and os.path.exists(track3_srt) and os.path.getsize(track3_srt) > 0
    has_onomatopoeia = onomatopoeia_srt and os.path.exists(onomatopoeia_srt) and os.path.getsize(onomatopoeia_srt) > 0
    
    log(f"Subtitle availability - Track2 (Mic): {has_track2}, Track3 (Desktop): {has_track3}, Onomatopoeia: {has_onomatopoeia}")
    
    # Choose embedding strategy - always prefer more subtitle tracks
    if has_track2 and has_track3 and has_onomatopoeia:
        log("Using triple subtitle embedding (mic + desktop + comic effects)")
        embed_triple_subtitles(input_video, output_video, track2_srt, track3_srt, onomatopoeia_srt, onomatopoeia_events, log)
    elif has_track2 and has_track3:
        log("Using dual subtitle embedding (mic + desktop only)")
        embed_dual_subtitles(input_video, output_video, track2_srt, track3_srt, log)
    elif has_track2:
        log("Using single subtitle embedding (microphone only)")
        embed_single_subtitles(input_video, output_video, track2_srt, log, is_mic_track=True)
    elif has_track3:
        log("Using single subtitle embedding (desktop only)")
        embed_single_subtitles(input_video, output_video, track3_srt, log, is_mic_track=False)
    else:
        log("No subtitle tracks available - copying video without subtitles")
        import shutil
        shutil.copy2(input_video, output_video)
        log(f"Video copied without subtitles: {os.path.basename(output_video)}")