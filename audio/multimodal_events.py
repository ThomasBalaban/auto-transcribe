"""
Save this file as: multimodal_events.py

Generalized multimodal onomatopoeia engine.
- Detects events based on universal audio-visual cues (transients, motion, flashes).
- Uses adaptive thresholds and relative scoring for broad applicability.
- Chooses onomatopoeia based on the physical characteristics of the event.

Dependencies: numpy, scipy, librosa, opencv-python
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional, Tuple, Set
import numpy as np
from scipy.signal import find_peaks
import cv2

try:
    import librosa # type: ignore
except ImportError:
    librosa = None

# --------------------------
# Data Models & Config
# --------------------------

@dataclass
class Event:
    """Represents a significant audio-visual event."""
    t_peak: float
    t_start: float
    t_end: float
    score: float = 0.0
    intensity: float = 0.0  # Normalized 0-1 measure of energy/impact
    context: Set[str] = field(default_factory=set)
    features: Dict[str, float] = field(default_factory=dict)
    position_hint: Optional[Tuple[float, float]] = None

@dataclass
class Candidate:
    """A potential event candidate from either audio or video stream."""
    t: float
    source: Literal["audio", "video"]
    label: str  # e.g., "audio_transient", "visual_motion_burst"
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)

class Config:
    """General configuration for the multimodal engine."""
    # Time windows
    VERIFY_WINDOW_SEC: float = 0.5  # How long to wait for a matching event from another modality
    NMS_RADIUS_SEC: float = 0.3     # Non-maximum suppression window to avoid duplicate events
    COOLDOWN_SEC: float = 0.4       # Cooldown period after a major event to reduce noise

    # Audio thresholds (adaptive)
    TRANSIENT_PROMINENCE_PERCENTILE: float = 92.0 # Detects sharp sounds relative to local audio
    MIN_AUDIO_CONFIDENCE: float = 0.4

    # Video thresholds (adaptive)
    MOTION_BURST_MULTIPLIER: float = 2.5  # How much motion must exceed the rolling average
    FLASH_DELTA_THRESHOLD: float = 50.0   # Brightness change to be considered a flash
    MIN_VIDEO_CONFIDENCE: float = 0.5

    # Context detection
    WATER_HUE_RANGE: Tuple[int, int] = (90, 130) # HSV Hue range for blues/cyans
    WATER_SATURATION_MIN: int = 70

# --------------------------
# Feature Extractors
# --------------------------

def get_audio_candidates(y: np.ndarray, sr: int, cfg: Config) -> List[Candidate]:
    """Proposes audio event candidates based on acoustic properties."""
    if not librosa or y.size < sr // 10:
        return []

    # 1. Detect sharp transients using spectral flux
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    if not np.any(onset_env) or np.max(onset_env) == 0: return []

    prominence_thresh = np.percentile(onset_env, cfg.TRANSIENT_PROMINENCE_PERCENTILE)
    peaks, props = find_peaks(onset_env, prominence=prominence_thresh)

    candidates = []
    for p, prom in zip(peaks, props['prominences']):
        t = librosa.frames_to_time(p, sr=sr, hop_length=256)
        confidence = min(1.0, prom / (np.max(onset_env) * 0.5))
        
        # Analyze frequency content at the transient
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        # Ensure peak index is within STFT bounds
        if p >= stft.shape[1]: continue
        frame = stft[:, p]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        
        low_energy = np.sum(frame[freqs < 400])
        high_energy = np.sum(frame[freqs > 2000])
        total_energy = np.sum(frame) + 1e-9

        candidates.append(Candidate(
            t=t,
            source="audio",
            label="audio_transient",
            confidence=float(confidence),
            features={
                "low_freq_ratio": float(low_energy / total_energy),
                "high_freq_ratio": float(high_energy / total_energy),
            }
        ))
    return candidates

def get_video_candidates(frames: List[np.ndarray], times: List[float], cfg: Config) -> List[Candidate]:
    """Proposes video event candidates based on motion and brightness."""
    candidates = []
    if len(frames) < 2: return []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    motion_magnitudes = []
    brightness_levels = [np.mean(prev_gray)]

    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_magnitudes.append(np.mean(mag))
        brightness_levels.append(np.mean(gray))
        prev_gray = gray

    if not motion_magnitudes: return []

    motion_median = np.median(motion_magnitudes) + 1e-6
    for i, t in enumerate(times[1:]):
        # Motion burst detection
        mag = motion_magnitudes[i]
        if mag > motion_median * cfg.MOTION_BURST_MULTIPLIER:
            candidates.append(Candidate(
                t=t, source="video", label="visual_motion_burst",
                confidence=min(1.0, mag / (motion_median * 4 * cfg.MOTION_BURST_MULTIPLIER)), # safer normalization
                features={"motion_magnitude": mag}
            ))
        # Flash detection
        brightness_delta = abs(brightness_levels[i+1] - brightness_levels[i])
        if brightness_delta > cfg.FLASH_DELTA_THRESHOLD:
            candidates.append(Candidate(
                t=t, source="video", label="visual_flash",
                confidence=0.8, # Flashes are usually high-confidence events
                features={"brightness_delta": brightness_delta}
            ))
    return candidates

# --------------------------
# Main Engine
# --------------------------

class MultimodalOnomatopoeia:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.unmatched_candidates: List[Candidate] = []
        self.event_history: List[Event] = []

    def process_window(
        self,
        audio_chunk: np.ndarray,
        sr: int,
        video_frames: List[np.ndarray],
        frame_times: List[float],
        t_start_abs: float,
    ) -> List[Event]:
        """Processes one window of audio/video data and returns confirmed events."""
        # 1. Generate new candidates for this window and adjust timestamps to be absolute
        audio_cands = get_audio_candidates(audio_chunk, sr, self.cfg)
        for c in audio_cands: c.t += t_start_abs
        
        video_cands = get_video_candidates(video_frames, frame_times, self.cfg)
        for c in video_cands: c.t += t_start_abs

        # 2. Match new candidates with any lingering unmatched ones
        events = []
        all_candidates = sorted(self.unmatched_candidates + audio_cands + video_cands, key=lambda x: x.t)
        
        matched_indices = set()
        for i in range(len(all_candidates)):
            if i in matched_indices: continue
            cand1 = all_candidates[i]
            
            # Find the best cross-modal match within the verification window
            best_match_idx = -1
            for j in range(i + 1, len(all_candidates)):
                if j in matched_indices: continue
                cand2 = all_candidates[j]
                
                # Stop searching if we're past the time window
                if cand2.t - cand1.t > self.cfg.VERIFY_WINDOW_SEC:
                    break
                
                # Found a cross-modal match
                if cand1.source != cand2.source:
                    best_match_idx = j
                    break

            if best_match_idx != -1:
                cand2 = all_candidates[best_match_idx]
                matched_indices.add(i)
                matched_indices.add(best_match_idx)
                
                # Create an event from the matched pair
                combined_features = {cand1.label: cand1.confidence, cand2.label: cand2.confidence, **cand1.features, **cand2.features}
                peak_time = (cand1.t + cand2.t) / 2.0
                score = (cand1.confidence + cand2.confidence) / 2.0
                intensity = np.clip(score + combined_features.get("motion_magnitude", 0) / 10.0, 0, 1)

                events.append(Event(
                    t_peak=peak_time,
                    t_start=peak_time - 0.15,
                    t_end=peak_time + 0.6,
                    score=score,
                    intensity=intensity,
                    features=combined_features
                ))

        # 3. Update the list of unmatched candidates for the next window
        self.unmatched_candidates = [
            cand for i, cand in enumerate(all_candidates) 
            if i not in matched_indices and (t_start_abs - cand.t < self.cfg.VERIFY_WINDOW_SEC)
        ]

        # 4. Post-process the generated events
        if not events:
            return []
            
        # Add context (e.g., underwater)
        for event in events:
            if not video_frames or not frame_times: continue
            closest_frame_idx = np.argmin([abs((t + t_start_abs) - event.t_peak) for t in frame_times])
            frame = video_frames[closest_frame_idx]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            water_mask = cv2.inRange(hsv, (self.cfg.WATER_HUE_RANGE[0], self.cfg.WATER_SATURATION_MIN, 20), (self.cfg.WATER_HUE_RANGE[1], 255, 255))
            if np.mean(water_mask) > 20: # If > 20% of pixels are in water color range
                event.context.add("underwater")

        # Non-maximum suppression and cooldown
        events.sort(key=lambda e: e.score, reverse=True)
        final_events = []
        for event in events:
            # NMS check
            if any(abs(event.t_peak - fe.t_peak) < self.cfg.NMS_RADIUS_SEC for fe in final_events):
                continue
            # Cooldown check
            if any(event.t_peak - he.t_peak > 0 and event.t_peak - he.t_peak < self.cfg.COOLDOWN_SEC for he in self.event_history):
                continue
            final_events.append(event)

        self.event_history.extend(final_events)
        self.event_history = self.event_history[-50:]

        return sorted(final_events, key=lambda e: e.t_peak)

    def pick_word(self, event: Event) -> str:
        """Selects an appropriate onomatopoeia based on event characteristics."""
        features = event.features
        is_underwater = "underwater" in event.context
        
        # Determine sound character from features
        is_sharp = features.get("high_freq_ratio", 0) > 0.35
        is_deep = features.get("low_freq_ratio", 0) > 0.45
        has_flash = "visual_flash" in features
        
        word = "BUMP" # Default
        if has_flash and is_sharp:
            word = "BLAM" if event.intensity > 0.6 else "ZAP"
        elif is_sharp and not is_deep:
            word = "CRACK" if event.intensity > 0.7 else "CLICK"
        elif is_deep and not is_sharp:
            word = "BOOM" if event.intensity > 0.8 else "THUD"
        elif is_sharp and is_deep: # broadband sound
             word = "CRASH" if event.intensity > 0.7 else "BANG"
        else: # mid-range
            word = "WHACK"

        # Apply context modifiers
        if is_underwater:
            word = {"CRACK": "BLIP", "CLICK": "plink", "BOOM": "THOOM", "THUD": "THUMP", "CRASH": "SPLOOSH", "BANG": "GLUG", "WHACK": "SWISH", "BLAM": "BWUMP"}.get(word, "BLUB")

        # Apply intensity modifiers
        if event.intensity > 0.85 and len(word) > 2:
            # Lengthen vowel for intense sounds
            for vowel in "AEIOU":
                if vowel in word:
                    word = word.replace(vowel, vowel * 2, 1)
                    break
            word += "!"
        
        return word.upper()

def sliding_windows(total_dur: float, win: float = 1.0, hop: float = 0.5):
    """Convenience generator for sliding windows."""
    t = 0.0
    while t < total_dur:
        yield t, min(t + win, total_dur)
        t += hop