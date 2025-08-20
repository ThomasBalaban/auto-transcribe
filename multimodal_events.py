"""
Save this file as: multimodal_events.py

Drop-in multimodal onomatopoeia engine.
- Audio + video candidate proposals
- Hypothesis tracking & verification (handles "who shot?" and underwater cases)
- Loudness-aware scoring so big moments win
- Context-aware word picker with constrained morphs

Dependencies: numpy, scipy, librosa, opencv-python

Typical use from your pipeline (pseudo):

    engine = MultimodalOnomatopoeia(cfg=Config())
    for win in windows:  # each win ~0.5-1.0s
        events = engine.process_window(audio_chunk, sr, video_frames, frame_times, t0, t1)
        for e in events:
            word = engine.pick_word(e)
            # pass (word, e.t_start, e.t_peak, e.t_end, e.position_hint) to your subtitle/animation layer

Notes:
- This file includes minimal heuristics for SED and optical flow so you can run today.
  Swap DummySED with a real model (e.g., PANNs/HTS-AT) when ready.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional, Tuple, Set
import numpy as np
from scipy.signal import find_peaks
import cv2

try:
    import librosa
except Exception:  # fallback if librosa missing
    librosa = None

# --------------------------
# Data models
# --------------------------

@dataclass
class LoudnessProfile:
    lufs_short: float
    lufs_integrated: float
    crest_db: float
    floor_db: float
    broadband_ratio: float

@dataclass
class Candidate:
    t: float
    source: Literal["audio", "video"]
    labels: List[str]
    conf: float
    feats: Dict[str, float] = field(default_factory=dict)

@dataclass
class Event:
    cls: str
    t: float
    t_start: float
    t_peak: float
    t_end: float
    score: float = 0.0
    context: Set[str] = field(default_factory=set)
    features: Dict[str, float] = field(default_factory=dict)
    position_hint: Optional[Tuple[float, float]] = None  # normalized (x,y) if known

@dataclass
class Hypothesis:
    cls: str
    t_audio: Optional[float] = None
    t_video: Optional[float] = None
    conf_audio: float = 0.0
    conf_video: float = 0.0
    context: Set[str] = field(default_factory=set)
    state: Literal["pending", "verified", "rejected"] = "pending"
    expires_at: float = 0.0

# --------------------------
# Config
# --------------------------

class Config:
    verify_window_sec: float = 2.0
    nms_radius_sec: float = 0.35
    cooldown_hi_sec: float = 0.8
    min_audio_conf: float = 0.6
    # thresholds (initial heuristics, tune per project)
    transient_peak_prominence: float = 0.6
    motion_burst_thresh: float = 1.5  # multiplier vs rolling median
    flash_thresh: float = 40.0         # brightness delta
    water_hue_lo: int = 75             # HSV ~ blue-green
    water_hue_hi: int = 110
    water_sat_min: int = 60
    underwater_lowpass_hz: int = 1200
    reverb_rt60_proxy_ms: int = 250

    # Priority mapping for classes
    priority: Dict[str, float] = {
        "punch": 3.2,
        "impact": 3.0,
        "gunshot": 3.0,
        "explosion": 2.8,
        "splash": 2.4,
        "glass": 2.2,
        "vehicle": 1.8,
        "footsteps": 0.8,
        "ladder": 0.6,
        "typing": 0.3,
    }

# --------------------------
# Utility: Loudness & dynamics
# --------------------------

def compute_loudness_profile(y: np.ndarray, sr: int) -> LoudnessProfile:
    """Compute simple loudness/dynamics features (lufs-ish values are relative)."""
    if y.size == 0:
        return LoudnessProfile(-60.0, -60.0, 0.0, -60.0, 0.0)

    # RMS and peak
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    peak = np.max(np.abs(y) + 1e-12)
    crest = 20*np.log10(peak/(rms+1e-12) + 1e-12)

    # Integrated level (pseudo-LUFS)
    lufs_integrated = 20*np.log10(rms + 1e-12)

    # Short term over 400 ms
    win = max(1, int(0.4*sr))
    if y.size >= win:
        frames = y[: y.size - (y.size % win)].reshape(-1, win)
        rms_short = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        lufs_short = float(np.median(20*np.log10(rms_short + 1e-12)))
        floor_db = float(np.percentile(20*np.log10(np.abs(y)+1e-12), 15))
    else:
        lufs_short = lufs_integrated
        floor_db = lufs_integrated - 20

    # Broadband ratio via STFT bands
    if librosa is not None and y.size > sr//10:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        wide = S[(freqs >= 200) & (freqs <= 6000)].sum()
        narrow = S[(freqs >= 1000) & (freqs <= 2000)].sum() + 1e-9
        broadband_ratio = float(wide / narrow)
    else:
        broadband_ratio = 1.0

    return LoudnessProfile(
        lufs_short=float(lufs_short),
        lufs_integrated=float(lufs_integrated),
        crest_db=float(crest),
        floor_db=float(floor_db),
        broadband_ratio=float(broadband_ratio),
    )

# --------------------------
# Dummy SED + Flow wrappers (replace with real models later)
# --------------------------

class DummySED:
    """Very crude SED based on spectral heuristics; replace with PANNs/HTS-AT."""
    def infer(self, y: np.ndarray, sr: int) -> List[Tuple[float, str, float]]:
        # Returns list of (t, label, conf)
        out = []
        if y.size < sr//5:
            return out
        # Detect broadband transient as impact/punch
        if librosa is not None:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks, _ = find_peaks(onset_env, prominence=np.percentile(onset_env, 85))
            for p in peaks:
                t = float(p * (512/sr))  # approx hop 512 default
                out.append((t, "impact", 0.6))
        # Very rough gunshot: high crest + high HF energy
        crest = compute_loudness_profile(y, sr).crest_db
        if crest > 14:
            out.append((0.0, "gunshot", min(0.9, (crest-10)/10)))
        return out

class SimpleFlow:
    def __init__(self):
        self.prev_gray = None
        self.med_win: List[float] = []

    def motion_burst(self, frames: List[np.ndarray], times: List[float]) -> List[Tuple[float, float]]:
        """Return list of (t, flow_magnitude_mean)."""
        out = []
        for f, t in zip(frames, times):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            if self.prev_gray is None:
                self.prev_gray = gray
                out.append((t, 0.0))
                continue
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            m = float(np.mean(mag))
            out.append((t, m))
            self.prev_gray = gray
        return out

# --------------------------
# Proposals: audio & video
# --------------------------

def propose_audio(y: np.ndarray, sr: int, loud: LoudnessProfile, cfg: Config, sed: DummySED) -> List[Candidate]:
    cands: List[Candidate] = []
    # Spectral flux / onset curve
    if librosa is not None and y.size > sr//10:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)
        flux = flux / (np.max(flux) + 1e-9)
        peaks, props = find_peaks(flux, prominence=np.percentile(flux, 80))
        for p, prom in zip(peaks, props.get('prominences', np.zeros_like(peaks))):
            t = float(p * (256/sr))
            rel = float(20*np.log10(np.max(np.abs(y)) + 1e-12) - loud.floor_db)
            cands.append(Candidate(
                t=t,
                source="audio",
                labels=["transient"],
                conf=float(min(1.0, 0.4 + prom)),
                feats={
                    "audio_transient_rel": rel/20.0,
                    "broadband_ratio": loud.broadband_ratio,
                }
            ))
    # SED labels
    for t, label, conf in sed.infer(y, sr):
        cands.append(Candidate(
            t=float(t), source="audio", labels=[label], conf=float(conf),
            feats={"audio_transient_rel": max(0.0, (loud.crest_db)/20.0), "broadband_ratio": loud.broadband_ratio}
        ))
    return cands

def _detect_flash_and_water(frame: np.ndarray, cfg: Config) -> Tuple[bool, bool]:
    # flash: global brightness spike (use frame std/mean ratios)
    mean = float(np.mean(frame))
    std = float(np.std(frame))
    flash = (mean > cfg.flash_thresh) or (std > cfg.flash_thresh)
    # water/underwater regions in HSV
    if frame.ndim == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (cfg.water_hue_lo, cfg.water_sat_min, 0), (cfg.water_hue_hi, 255, 255))
        water = (np.mean(mask) > 10)  # percent of pixels
    else:
        water = False
    return flash, water

def propose_video(frames: List[np.ndarray], times: List[float], flow: SimpleFlow, cfg: Config) -> List[Candidate]:
    cands: List[Candidate] = []
    flows = flow.motion_burst(frames, times)
    mags = np.array([m for _, m in flows])
    if mags.size:
        med = np.median(mags) + 1e-6
    else:
        med = 1e-6
    for (t, m), f in zip(flows, frames):
        lbls = []
        conf = 0.0
        if m > cfg.motion_burst_thresh * med:
            lbls.append("impact_visual")
            conf = max(conf, min(1.0, (m/med)/4.0))
        flash, water = _detect_flash_and_water(f, cfg)
        if flash:
            lbls.append("flash")
            conf = max(conf, 0.6)
        if water:
            lbls.append("water_region")
            conf = max(conf, 0.5)
        if lbls:
            cands.append(Candidate(t=float(t), source="video", labels=lbls, conf=float(conf), feats={"motion_burst": float(m/med)}))
    return cands

# --------------------------
# Hypothesis tracking & verification
# --------------------------

class HypothesisTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hyps: List[Hypothesis] = []

    def add_audio(self, label: str, t: float, conf: float):
        h = Hypothesis(cls=label, t_audio=t, conf_audio=conf, state="pending", expires_at=t + self.cfg.verify_window_sec)
        self.hyps.append(h)

    def attach_video(self, cls: str, t: float, conf: float):
        # Attach to nearest matching pending hypothesis or create new
        best = None
        best_dt = 9e9
        for h in self.hyps:
            if h.cls != cls: continue
            dt = abs(((h.t_audio if h.t_audio is not None else t) - t))
            if dt < best_dt and h.state == "pending":
                best, best_dt = h, dt
        if best is None:
            best = Hypothesis(cls=cls, t_video=t, conf_video=conf, state="pending", expires_at=t + self.cfg.verify_window_sec)
            self.hyps.append(best)
        else:
            best.t_video = t
            best.conf_video = max(best.conf_video, conf)

    def add_context_near(self, t: float, tag: str, radius: float = 0.6):
        for h in self.hyps:
            t_ref = h.t_audio if h.t_audio is not None else h.t_video
            if t_ref is None: continue
            if abs(t_ref - t) <= radius:
                h.context.add(tag)

    def finalize_ready(self, t_now: float) -> List[Hypothesis]:
        ready, keep = [], []
        for h in self.hyps:
            if h.state == "pending" and t_now >= h.expires_at:
                if h.conf_audio >= self.cfg.min_audio_conf or h.conf_video >= 0.6:
                    h.state = "verified"
                else:
                    h.state = "rejected"
            if h.state == "verified":
                ready.append(h)
            elif h.state == "pending":
                keep.append(h)
        self.hyps = keep
        return ready

# --------------------------
# Scoring, NMS, masking
# --------------------------

def _priority(cls: str, cfg: Config) -> float:
    return cfg.priority.get(cls, 0.0)

def _context_boost(cls: str, context: Set[str]) -> float:
    # simple boosts; tune later
    if cls == "splash" and ("water" in context or "underwater" in context):
        return 0.5
    if cls in {"gunshot", "explosion"} and "indoor_reverb" in context:
        return 0.2
    return 0.0

def score_events(events: List[Event], recent: List[Event], cfg: Config) -> List[Event]:
    def uniqueness(t: float) -> float:
        # boost if nothing fired recently
        return float(min(1.0, np.mean([abs(t - e.t) for e in recent[-10:]] or [1.0])))

    for e in events:
        f = e.features
        e.score = (
            1.4 * f.get("audio_transient_rel", 0.0) +
            1.2 * f.get("motion_burst", 0.0) +
            0.8 * f.get("broadband_ratio", 0.0) +
            1.6 * _priority(e.cls, cfg) +
            1.0 * uniqueness(e.t) -
            1.2 * f.get("repetitiveness", 0.0) +
            0.6 * _context_boost(e.cls, e.context)
        )
    return events

def temporal_nms(events: List[Event], radius: float) -> List[Event]:
    kept: List[Event] = []
    for e in sorted(events, key=lambda x: x.score, reverse=True):
        if all(abs(e.t - k.t) > radius for k in kept):
            kept.append(e)
    return kept

def priority_masking(events: List[Event], cfg: Config) -> List[Event]:
    out: List[Event] = []
    last_hi = -1e9
    for e in sorted(events, key=lambda x: x.t):
        if _priority(e.cls, cfg) >= 2.0:
            out.append(e)
            last_hi = e.t
        else:
            if e.t - last_hi > cfg.cooldown_hi_sec:
                out.append(e)
    return out

# --------------------------
# Timestamp refinement
# --------------------------

def refine_timestamps(events: List[Event], y: np.ndarray, sr: int, frames: List[np.ndarray]) -> List[Event]:
    # Audio peak refinement
    for e in events:
        # refine around e.t within ±120 ms
        half = int(0.12*sr)
        center = int(e.t * sr)
        lo = max(0, center - half)
        hi = min(y.size, center + half)
        seg = y[lo:hi]
        if seg.size > 10:
            # pick max absolute derivative as transient peak
            d = np.abs(np.diff(seg))
            p = int(np.argmax(d))
            t_peak = (lo + p) / sr
            e.t_peak = t_peak
            # crude start/end
            thresh = 0.2 * np.max(d + 1e-9)
            # start
            s = p
            while s > 1 and d[s-1] > thresh:
                s -= 1
            # end
            q = p
            while q < d.size-1 and d[q] > thresh:
                q += 1
            e.t_start = (lo + max(0, s-1)) / sr
            e.t_end = (lo + min(d.size-1, q+1)) / sr
        else:
            e.t_peak = e.t
            e.t_start = max(0.0, e.t - 0.08)
            e.t_end = e.t + 0.15
    return events

# --------------------------
# Word selection (prototypes + constrained morphs)
# --------------------------

PROTOTYPES: Dict[str, List[str]] = {
    "punch": ["SMACK", "THUD", "WHAM"],
    "impact": ["THUD", "WHAM", "SMACK"],
    "gunshot": ["BANG", "BLAM", "KRAK"],
    "explosion": ["BOOM", "KABOOM"],
    "splash": ["SPLASH", "SPLISH"],
    "ladder": ["CLANK", "CLINK"],
    "typing": ["TAP", "TIK"],
}

UNDERWATER_MAP: Dict[str, List[str]] = {
    "punch": ["FWOOMP", "BLUP", "THWUMP"],
    "impact": ["FWOOMP", "THWUMP"],
    "gunshot": ["BWUMP", "BLUP"],
    "splash": ["FWOOMP", "SPLUSH"],
}

class WordMemory:
    def __init__(self, max_len: int = 20):
        self.recent: List[str] = []
        self.max_len = max_len
    def push(self, w: str):
        self.recent.append(w)
        if len(self.recent) > self.max_len:
            self.recent = self.recent[-self.max_len:]


def constrained_morph(word: str, intensity: float, damped: bool) -> str:
    # Intensity scales letter doubling and punctuation; damped removes sharp endings
    w = word
    if intensity > 1.5 and len(w) >= 3:
        mid = len(w)//2
        w = w[:mid] + w[mid]*int(1+min(3, intensity-1)) + w[mid:]
    if not damped and intensity > 1.2:
        w = w + "!"
    if damped:
        # soften endings: SPLASH -> SPLUSH, BANG -> BWUNG
        w = w.replace("A", "U").replace("A", "U")
    return w[:10]

# --------------------------
# Main Engine
# --------------------------

class MultimodalOnomatopoeia:
    def __init__(self, cfg: Config | None = None, sed=None, flow=None):
        self.cfg = cfg or Config()
        self.sed = sed or DummySED()
        self.flow = flow or SimpleFlow()
        self.tracker = HypothesisTracker(self.cfg)
        self.recent_events: List[Event] = []
        self.word_memory = WordMemory()

    def process_window(
        self,
        audio_chunk: np.ndarray,
        sr: int,
        video_frames: List[np.ndarray],
        frame_times: List[float],
        t0: float,
        t1: float,
    ) -> List[Event]:
        """Process one window and emit verified, refined Events within [t0,t1]."""
        # 1) Loudness
        loud = compute_loudness_profile(audio_chunk, sr)
        # 2) Proposals
        acands = propose_audio(audio_chunk, sr, loud, self.cfg, self.sed)
        vcands = propose_video(video_frames, frame_times, self.flow, self.cfg)
        # 3) Update hypotheses
        for c in acands:
            for lbl in c.labels:
                if lbl in {"impact", "punch", "gunshot", "explosion", "splash", "ladder"}:
                    self.tracker.add_audio("punch" if lbl=="impact" else lbl, t0 + c.t, c.conf)
        for c in vcands:
            if "flash" in c.labels:
                self.tracker.attach_video("gunshot", t0 + c.t, c.conf)
            if "impact_visual" in c.labels:
                self.tracker.attach_video("punch", t0 + c.t, c.conf)
            if "water_region" in c.labels:
                self.tracker.add_context_near(t0 + c.t, "water")
        # 4) Finalize ready hyps
        hyps = self.tracker.finalize_ready(t1)
        # 5) Build events from hyps (rough times)
        events: List[Event] = []
        for h in hyps:
            t_ref = h.t_audio if h.t_audio is not None else h.t_video
            if t_ref is None: continue
            # initial windowed start/end
            e = Event(
                cls=h.cls,
                t=t_ref,
                t_start=max(0.0, t_ref - 0.08),
                t_peak=t_ref,
                t_end=t_ref + 0.18,
                context=set(h.context),
                features={
                    "audio_transient_rel": 1.0,
                    "motion_burst": 1.0,
                    "broadband_ratio": 1.0,
                    "repetitiveness": 0.0,
                },
            )
            events.append(e)
        # 6) Score + NMS + masking
        events = score_events(events, self.recent_events, self.cfg)
        events = temporal_nms(events, self.cfg.nms_radius_sec)
        events = priority_masking(events, self.cfg)
        # 7) Refine timestamps
        events = refine_timestamps(events, audio_chunk, sr, video_frames)
        # 8) Save to history and return
        self.recent_events.extend(events)
        self.recent_events = self.recent_events[-100:]
        return events

    def pick_word(self, e: Event) -> str:
        # Underwater / damped context?
        damped = ("water" in e.context) or ("underwater" in e.context)
        base_list = UNDERWATER_MAP.get(e.cls, PROTOTYPES.get(e.cls, [e.cls.upper()])) if damped else PROTOTYPES.get(e.cls, [e.cls.upper()])
        # Select first not recently used
        for w in base_list:
            if w not in self.word_memory.recent:
                chosen = w
                break
        else:
            chosen = base_list[0]
        intensity = 1.0 + min(1.5, e.score/4.0)
        out = constrained_morph(chosen, intensity=intensity, damped=damped)
        self.word_memory.push(out)
        return out

# --------------------------
# Convenience: simple windowing driver (optional)
# --------------------------

def sliding_windows(total_dur: float, win: float = 0.8, hop: float = 0.4):
    t = 0.0
    while t < total_dur:
        yield t, min(t+win, total_dur)
        t += hop

"""
Integration sketch (in your pipeline):

from multimodal_events import MultimodalOnomatopoeia, Config

engine = MultimodalOnomatopoeia(cfg=Config())

# Suppose you already have mono audio `y`, sample rate `sr`, and a frame buffer with timestamps.
# Provide helpers to slice the right segment for each window.

for t0, t1 in sliding_windows(total_dur=len(y)/sr):
    a0, a1 = int(t0*sr), int(t1*sr)
    audio_chunk = y[a0:a1]
    # Collect frames whose timestamps lie in [t0, t1]
    frames = [F for (F, Ft) in frame_buffer if t0 <= Ft <= t1]
    ftimes = [Ft - t0 for (F, Ft) in frame_buffer if t0 <= Ft <= t1]
    events = engine.process_window(audio_chunk, sr, frames, ftimes, t0, t1)
    for e in events:
        word = engine.pick_word(e)
        # hand off to your subtitle/animation layer
"""
