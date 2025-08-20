# video_processor.py - Fixed to use multimodal_events properly

import os
import numpy as np
import cv2
import librosa
from multimodal_events import MultimodalOnomatopoeia, Config, sliding_windows

def _load_video_frames(video_path, log):
    """Load video frames with timestamps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, times = [], []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every few frames to avoid memory issues
        if frame_count % max(1, int(fps / 8)) == 0:  # Sample at ~8 FPS
            t = frame_count / fps
            frames.append(frame)
            times.append(t)
        
        frame_count += 1
        
        # Limit total frames to prevent memory issues
        if len(frames) > 1000:  # ~2 minutes at 8 FPS
            log("Warning: Video too long, truncating frame analysis")
            break
    
    cap.release()
    log(f"Loaded {len(frames)} frames @ ~{fps:.2f} fps (sampled at ~8 FPS)")
    return frames, times

def _load_audio_mono(video_path, log, sr=16000):
    """Load audio as mono"""
    try:
        y, sr = librosa.load(video_path, sr=sr, mono=True)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        log(f"Loaded audio mono @ {sr} Hz, {len(y)/sr:.2f}s")
        return y, sr
    except Exception as e:
        log(f"Error loading audio: {e}")
        # Return silence if audio loading fails
        return np.zeros(sr * 10, dtype=np.float32), sr

def _convert_events_to_subtitle_format(events, engine, log):
    """Convert multimodal events to format expected by animation system"""
    subtitle_events = []
    
    for event in events:
        try:
            # Generate word using the engine's word picker
            word = engine.pick_word(event)
            
            # Convert to expected format
            subtitle_event = {
                'word': word,
                'start_time': float(event.t_start),
                'end_time': float(event.t_end),
                'confidence': 0.8,  # Default confidence
                'energy': event.score / 5.0,  # Normalize score to energy-like value
                'context': list(event.context) if event.context else [],
                'multimodal': True,
                'event_class': event.cls
            }
            
            subtitle_events.append(subtitle_event)
            log(f"Generated effect: '{word}' at {event.t_start:.2f}s-{event.t_end:.2f}s")
            
        except Exception as e:
            log(f"Error converting event: {e}")
            continue
    
    return subtitle_events

class VideoProcessor:
    @staticmethod
    def process_single_video(
        input_file: str,
        output_file: str,
        model_name: str,
        device_name: str,
        ai_sensitivity: float,
        animation_type: str,
        log
    ):
        """
        Process video using multimodal onomatopoeia detection
        """
        try:
            log("="*60)
            log(f"MULTIMODAL ONOMATOPOEIA PROCESSING")
            log(f"Input: {os.path.basename(input_file)}")
            log(f"Animation: {animation_type}")
            log(f"AI Sensitivity: {ai_sensitivity}")
            log("="*60)

            # Initialize multimodal engine with sensitivity
            log("Initializing multimodal onomatopoeia engine...")
            cfg = Config()
            # Adjust thresholds based on sensitivity
            cfg.min_audio_conf = 0.8 - (ai_sensitivity * 0.3)  # 0.5-0.8 range
            engine = MultimodalOnomatopoeia(cfg=cfg)

            # Load video and audio
            log("Loading video frames...")
            frames, frame_times = _load_video_frames(input_file, log)
            
            log("Loading audio...")
            y, sr = _load_audio_mono(input_file, log, sr=16000)
            
            total_dur = max(
                frame_times[-1] if frame_times else 0.0, 
                len(y)/sr
            )
            log(f"Total duration: {total_dur:.2f}s")

            # Process in sliding windows
            log("Detecting multimodal onomatopoeia events...")
            all_events = []
            window_count = 0
            
            for t0, t1 in sliding_windows(total_dur, win=1.0, hop=0.5):
                window_count += 1
                
                # Extract audio chunk
                a0, a1 = int(t0*sr), int(min(t1*sr, len(y)))
                audio_chunk = y[a0:a1]
                
                # Extract video frames for this window
                window_frames = []
                window_times = []
                for frame, frame_time in zip(frames, frame_times):
                    if t0 <= frame_time <= t1:
                        window_frames.append(frame)
                        window_times.append(frame_time - t0)  # Relative to window start
                
                # Process window
                try:
                    events = engine.process_window(
                        audio_chunk, sr, window_frames, window_times, t0, t1
                    )
                    all_events.extend(events)
                    
                    if events:
                        log(f"Window {window_count} ({t0:.1f}s-{t1:.1f}s): {len(events)} events")
                    
                except Exception as e:
                    log(f"Error processing window {window_count}: {e}")
                    continue

            log(f"Total events detected: {len(all_events)}")

            # Convert events to subtitle format
            log("Converting events to subtitle format...")
            subtitle_events = _convert_events_to_subtitle_format(all_events, engine, log)

            if not subtitle_events:
                log("No onomatopoeia events detected!")
                # Create empty subtitle file
                base_name = os.path.splitext(output_file)[0]
                subtitle_path = base_name + ('.ass' if animation_type != 'Static' else '.srt')
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    f.write("")
                log(f"Created empty subtitle file: {subtitle_path}")
                return

            # Create subtitle file using existing animation system
            log("Creating animated subtitle file...")
            try:
                if animation_type == "Static":
                    # Create SRT file
                    from subtitle_converter import convert_to_srt
                    
                    # Convert to text format for SRT conversion
                    text_lines = []
                    for event in subtitle_events:
                        start = event['start_time']
                        end = event['end_time']
                        word = event['word']
                        text_lines.append(f"{start:.2f}-{end:.2f}: {word}")
                    
                    text_content = '\n'.join(text_lines)
                    srt_path = os.path.splitext(output_file)[0] + '.srt'
                    convert_to_srt(text_content, srt_path, input_file, log)
                    
                else:
                    # Create animated ASS file
                    from animations.renderer import create_animated_onomatopoeia_ass
                    
                    ass_path = os.path.splitext(output_file)[0] + '.ass'
                    success, events = create_animated_onomatopoeia_ass(
                        input_file, ass_path, animation_type, log
                    )
                    
                    if not success:
                        log("Animated ASS creation failed, falling back to direct creation...")
                        # Direct ASS creation as fallback
                        from animations.core import OnomatopoeiaAnimator
                        animator = OnomatopoeiaAnimator()
                        ass_content = animator.generate_animated_ass_content(
                            subtitle_events, animation_type
                        )
                        with open(ass_path, 'w', encoding='utf-8') as f:
                            f.write(ass_content)
                        log(f"Created ASS file directly: {ass_path}")
                
                log("✅ Multimodal onomatopoeia processing complete!")
                
            except Exception as e:
                log(f"Error creating subtitle file: {e}")
                log("Creating basic subtitle file as fallback...")
                
                # Basic fallback subtitle creation
                subtitle_path = os.path.splitext(output_file)[0] + '.srt'
                with open(subtitle_path, 'w', encoding='utf-8') as f:
                    for i, event in enumerate(subtitle_events, 1):
                        start_time = event['start_time']
                        end_time = event['end_time']
                        word = event['word']
                        
                        # Format as SRT
                        start_srt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
                        end_srt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"
                        
                        f.write(f"{i}\n{start_srt} --> {end_srt}\n{word}\n\n")
                
                log(f"Created fallback SRT: {subtitle_path}")

            # EMBED SUBTITLES INTO VIDEO
            log("Embedding subtitles into final video...")
            try:
                if animation_type == "Static":
                    # Embed SRT file
                    import subprocess
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', input_file,
                        '-vf', f"subtitles='{subtitle_path}':force_style='FontName=Bold Marker,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Bold=1,Outline=3,Shadow=2,Alignment=2,MarginV=100'",
                        '-c:a', 'copy',
                        output_file
                    ]
                    
                    log(f"Embedding SRT subtitles...")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        log(f"✅ Final video with SRT subtitles: {output_file}")
                    else:
                        log(f"❌ FFmpeg SRT error: {result.stderr}")
                        raise Exception("SRT embedding failed")
                        
                else:
                    # Embed ASS file with native ASS styling
                    import subprocess
                    
                    # Use ass filter for proper ASS file handling
                    ass_path_for_ffmpeg = ass_path.replace('\\', '/').replace(':', r'\\:')
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', input_file,
                        '-vf', f"ass='{ass_path_for_ffmpeg}'",
                        '-c:a', 'copy',
                        output_file
                    ]
                    
                    log(f"Embedding ASS subtitles...")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        log(f"✅ Final video with animated subtitles: {output_file}")
                    else:
                        log(f"❌ FFmpeg ASS error: {result.stderr}")
                        raise Exception("ASS embedding failed")
                        
            except Exception as e:
                log(f"Error embedding subtitles: {e}")
                # Copy original video as fallback
                import shutil
                shutil.copy2(input_file, output_file)
                log(f"⚠️  Copied original video without subtitles: {output_file}")

        except Exception as e:
            log(f"ERROR in multimodal processing: {e}")
            import traceback
            log(f"Traceback: {traceback.format_exc()}")
            raise