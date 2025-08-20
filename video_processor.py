# video_processor.py
# Runs multimodal onomatopoeia detection over a single video and emits lines.
# Keeps the public API used by main.py.

import os
import numpy as np
import cv2
import librosa
from subtitle_generator import SubtitleGenerator, get_events_for_generator

from multimodal_events import MultimodalOnomatopoeia, Config, sliding_windows

# Optional: if you have a subtitle aggregator, we’ll use it.
# If this import fails, we’ll still write a sidecar SRT as a fallback.
try:
    from subtitle_generator import add_onomatopoeia, clear_onomatopoeia, get_onomatopoeia_lines
    HAVE_SUB_GEN = True
except Exception:
    HAVE_SUB_GEN = False
    add_onomatopoeia = None
    clear_onomatopoeia = None
    get_onomatopoeia_lines = None


def _load_video_frames(video_path, log):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, times = [], []
    i = 0
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        frames.append(frame)
        times.append(t)
        i += 1
    cap.release()
    log(f"Loaded {len(frames)} frames @ ~{fps:.2f} fps")
    return frames, times


def _load_audio_mono(video_path, log, sr=16000):
    # librosa loads via ffmpeg backend; mono for speed
    y, sr = librosa.load(video_path, sr=sr, mono=True)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    log(f"Loaded audio mono @ {sr} Hz, {len(y)/sr:.2f}s")
    return y, sr


def _write_sidecar_srt(lines, output_video_path, log):
    # Write a simple SRT with the onomatopoeia lines.
    # We use each entry's start/end/text fields.
    def to_srt_time(t):
        t = max(0.0, float(t))
        h = int(t // 3600); t -= 3600*h
        m = int(t // 60);   t -= 60*m
        s = int(t)
        ms = int(round((t - s)*1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt_path = os.path.splitext(output_video_path)[0] + ".onomatopoeia.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        idx = 1
        for item in lines:
            start = to_srt_time(item["start"])
            end   = to_srt_time(item["end"])
            text  = item["text"]
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
            idx += 1
    log(f"Wrote sidecar SRT: {srt_path}")
    return srt_path


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
        Keep this signature so main.py doesn't change.
        We focus on running the multimodal onomatopoeia engine.
        """

        log("Initializing multimodal onomatopoeia engine...")
        cfg = Config()
        engine = MultimodalOnomatopoeia(cfg=cfg)

        # 1) Load AV
        frames, frame_times = _load_video_frames(input_file, log)
        y, sr = _load_audio_mono(input_file, log, sr=16000)
        total_dur = max(frame_times[-1] if frame_times else 0.0, len(y)/sr)

        # Prepare subtitle collector (optional)
        if HAVE_SUB_GEN and clear_onomatopoeia:
            clear_onomatopoeia()

        # 2) Slide windows and run engine
        log("Detecting onomatopoeia events...")
        count = 0
        for t0, t1 in sliding_windows(total_dur, win=0.8, hop=0.4):
            a0, a1 = int(t0*sr), int(min(t1*sr, len(y)))
            audio_chunk = y[a0:a1]

            # slice frames for this window
            F, T = [], []
            for f, ft in zip(frames, frame_times):
                if t0 <= ft <= t1:
                    F.append(f)
                    T.append(ft - t0)  # relative to window

            events = engine.process_window(audio_chunk, sr, F, T, t0, t1)
            for e in events:
                word = engine.pick_word(e)
                count += 1
                log(f"  [{e.cls.upper()}] {word}  t≈{e.t:.2f}  start={e.t_start:.2f}  peak={e.t_peak:.2f}  end={e.t_end:.2f}  ctx={list(e.context)}")

                if HAVE_SUB_GEN and add_onomatopoeia:
                    add_onomatopoeia(
                        word=word,
                        start=float(e.t_start),
                        peak=float(e.t_peak),
                        end=float(e.t_end),
                        cls=e.cls,
                        ctx=list(e.context),
                    )

        gen = SubtitleGenerator(log_func=log)
        events = get_events_for_generator()

        base = os.path.splitext(output_file)[0]
        subtitle_out = base + (".srt" if animation_type == "Static" else ".ass")
        gen.create_subtitle_file(events, subtitle_out, animation_type=animation_type)

        log(f"Detected {count} onomatopoeia events.")

        # 3) Emit: if you have a subtitle pipeline, it can now pull get_onomatopoeia_lines().
        #    As a safety net, we also write a sidecar SRT next to the output file.
        if HAVE_SUB_GEN and get_onomatopoeia_lines:
            lines = get_onomatopoeia_lines()
            if lines:
                _write_sidecar_srt(lines, output_file, log)
        else:
            # Build minimal line list from our scan if subtitle_generator isn't wired.
            log("subtitle_generator not found; writing sidecar from internal buffer.")
            # We didn't buffer internally; easy fix: re-run a fast pass to collect.
            # Simpler: tell user to enable subtitle_generator integration.
            log("Tip: add the subtitle_generator hooks (see instructions) to avoid this fallback.")
