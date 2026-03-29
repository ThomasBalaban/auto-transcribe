"""
Hallucination detection and filtering for Whisper transcriptions.

Two distinct filtering strategies:
  - Mic track:     Quiet-segment detection (mostly silence → hallucination)
  - Desktop track: Noise-aware detection (game audio/music → hallucination)
    Catches looping phrases, music lyrics, incoherent bursts, and short
    isolated words caused by SFX triggers.
"""

import os
import re
import difflib
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared blacklists / patterns
# ---------------------------------------------------------------------------

HALLUCINATION_BLACKLIST = {
    "thanks for watching", "like and subscribe", "don't forget to subscribe",
    "please like and subscribe", "see you next time",
    "make sure to hit that notification bell", "check out the description",
    "link in the description", "smash that like button", "hit the subscribe button",
    "music", "♪ music ♪", "background music", "♪", "♫", "♬",
    "subtítulos por la comunidad de amara.org", "subtitle by", "subtitles by",
    "audio", "video",
}

HALLUCINATION_PATTERNS = [
    r"^music\s*$",
    r"^♪.*♪$",
    r"^[a-z]\s+[a-z]\s+[a-z]$",
    r"thanks.*watching",
    r"like.*subscribe",
    r"subtítulos.*amara",
]

QUIET_SEGMENT_BLACKLIST = {
    "thank", "thanks", "you", "watching", "subscribe", "like", "please",
    "smash", "hit", "bell", "notification", "description", "link",
    "see", "next", "time", "make", "sure", "check", "out",
}

# ---------------------------------------------------------------------------
# Desktop-specific blacklists
# ---------------------------------------------------------------------------

# Phrases that Whisper commonly generates from game music / menu sounds.
DESKTOP_PHRASE_BLACKLIST = {
    # Generic music hallucinations
    "la la la", "na na na", "oh oh oh", "ah ah ah", "mm mm mm",
    "doo doo doo", "ba ba ba", "da da da",
    # Common game-audio misreads
    "loading", "press any key", "press start", "continue", "game over",
    "mission complete", "mission failed", "checkpoint reached",
    "player one", "insert coin", "high score", "new record",
    # Ambient / cinematic score hallucinations
    "aah", "ooh", "hmm", "ugh",
}

# Single words that almost never appear as real in-game speech but are
# common Whisper outputs when it hears SFX or musical stings.
DESKTOP_WORD_BLACKLIST = {
    "yeah", "okay", "alright", "right", "uh", "um", "er", "hm",
    "yes", "no", "hey", "oh", "ah", "ow", "aw", "wow",
    "hmm", "huh", "whoa", "whoo", "woo",
}

# Words strongly associated with music lyrics Whisper hallucinates.
MUSIC_LYRIC_SIGNALS = {
    "chorus", "verse", "bridge", "melody", "harmony", "rhythm",
    "sing", "song", "singing", "lyric",
}

# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------

def _load_audio_segment(
    audio_path: Optional[str],
    start: float,
    duration: float,
    sr: int = 16000,
) -> Optional[np.ndarray]:
    if not LIBROSA_AVAILABLE or not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration)
        return audio if len(audio) > 0 else None
    except Exception:
        return None


def is_quiet_segment(
    audio_path: Optional[str],
    start_time: float,
    end_time: float,
    quiet_threshold: float = 0.02,
) -> bool:
    """True when the segment is near-silent (mic-track logic)."""
    audio = _load_audio_segment(audio_path, start_time, end_time - start_time)
    if audio is None:
        return False
    return float(np.sqrt(np.mean(audio ** 2))) < quiet_threshold


def _spectral_flatness_mean(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Spectral flatness (0 = tonal/music, 1 = noise/white noise).
    Speech sits around 0.1–0.4; pure music tones are near 0;
    broadband game SFX can be 0.4–0.9.
    """
    try:
        stft = np.abs(librosa.stft(audio))
        flatness = librosa.feature.spectral_flatness(S=stft)
        return float(np.mean(flatness))
    except Exception:
        return 0.5


def _speech_likelihood(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Heuristic 0-1 score: how likely this audio window contains speech
    rather than music or SFX.

    Uses:
      - ZCR (speech ≈ 0.02–0.15)
      - Spectral centroid (speech ≈ 1000–3000 Hz)
      - Spectral flatness (speech ≈ 0.1–0.4)
    """
    try:
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        )
        flatness = _spectral_flatness_mean(audio, sr)

        # ZCR score: penalise very high (noise) and very low (music drone)
        zcr_score = 1.0 - abs(zcr - 0.08) / 0.12
        zcr_score = max(0.0, min(1.0, zcr_score))

        # Centroid score: speech sits in 1000–3000 Hz
        if 1000 <= centroid <= 3000:
            centroid_score = 1.0
        elif centroid < 1000:
            centroid_score = centroid / 1000.0
        else:
            centroid_score = max(0.0, 1.0 - (centroid - 3000) / 4000.0)

        # Flatness score: penalise very tonal (music) or very noisy (SFX)
        flatness_score = 1.0 - abs(flatness - 0.25) / 0.25
        flatness_score = max(0.0, min(1.0, flatness_score))

        return (zcr_score * 0.3 + centroid_score * 0.4 + flatness_score * 0.3)

    except Exception:
        return 0.5  # neutral when analysis fails


def desktop_audio_has_speech(
    audio_path: Optional[str],
    start_time: float,
    end_time: float,
    speech_threshold: float = 0.35,
) -> bool:
    """
    Returns True if the audio window looks like it contains speech rather
    than music / game SFX.  Used as a gate for desktop-track words.
    """
    duration = end_time - start_time
    audio = _load_audio_segment(audio_path, start_time, duration)
    if audio is None:
        return True  # Can't check → assume valid

    # Very short clips are unreliable to analyse; be permissive
    if len(audio) < 1600:  # < 0.1 s
        return True

    score = _speech_likelihood(audio)
    return score >= speech_threshold


# ---------------------------------------------------------------------------
# Repetition / loop detection  (desktop-specific)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def detect_looping_sequence(
    words: List[Tuple[float, float, str]],  # (start, end, text)
    window: int = 20,
    min_repeat: int = 3,
    similarity: float = 0.80,
) -> List[int]:
    """
    Finds indices of words that are part of a repeating / looping pattern.
    Whisper gets stuck in loops on noisy audio.

    Strategy: slide a window, detect if any short phrase (1–3 words)
    appears ≥ min_repeat times within the window.
    """
    if len(words) < min_repeat * 2:
        return []

    flagged = set()
    texts = [_normalise(w[2]) for w in words]

    for gram_size in (1, 2, 3):
        for i in range(len(texts) - gram_size + 1):
            gram = " ".join(texts[i : i + gram_size])
            if not gram.strip():
                continue

            # Count similar grams in the local window
            lo = max(0, i - window)
            hi = min(len(texts), i + window + gram_size)
            window_grams = [
                " ".join(texts[j : j + gram_size])
                for j in range(lo, hi - gram_size + 1)
            ]

            count = sum(
                1
                for g in window_grams
                if difflib.SequenceMatcher(None, gram, g).ratio() >= similarity
            )

            if count >= min_repeat:
                # Flag all occurrences in the window
                for j in range(lo, hi - gram_size + 1):
                    g = " ".join(texts[j : j + gram_size])
                    if difflib.SequenceMatcher(None, gram, g).ratio() >= similarity:
                        for k in range(j, j + gram_size):
                            flagged.add(k)

    return sorted(flagged)


# ---------------------------------------------------------------------------
# Coherence check
# ---------------------------------------------------------------------------

def _word_is_coherent(word: str) -> bool:
    """
    Very lightweight coherence check.  Rejects strings that look like
    random phonemes / non-words that Whisper emits on SFX.
    """
    word = word.strip().lower()
    if len(word) <= 1:
        return False
    # Must have at least one vowel
    if not re.search(r"[aeiou]", word):
        return False
    # Reject strings of the same character repeated
    if len(set(word)) <= 2 and len(word) >= 4:
        return False
    return True


# ---------------------------------------------------------------------------
# Shared pattern matching
# ---------------------------------------------------------------------------

def matches_hallucination_pattern(text: str) -> Tuple[bool, bool]:
    text_lower = text.lower().strip()
    exact = text_lower in HALLUCINATION_BLACKLIST
    partial = any(re.search(p, text_lower) for p in HALLUCINATION_PATTERNS)
    return exact, partial


def has_low_confidence(word_data: dict, threshold: float = 0.5) -> bool:
    if isinstance(word_data, dict) and "score" in word_data:
        return word_data["score"] < threshold
    return False


# ---------------------------------------------------------------------------
# Per-word decision: MIC track
# ---------------------------------------------------------------------------

def should_filter_word_mic(
    word_text: str,
    word_data: dict,
    audio_path: Optional[str],
    start_time: float,
    end_time: float,
) -> Tuple[bool, str]:
    """Original quiet-segment logic, unchanged for the mic track."""
    quiet = is_quiet_segment(audio_path, start_time, end_time)
    low_conf = has_low_confidence(word_data)
    exact, partial = matches_hallucination_pattern(word_text)

    if exact and quiet:
        return True, f"Exact hallucination match in quiet segment: '{word_text}'"
    if exact and low_conf:
        return True, f"Exact hallucination match with low confidence: '{word_text}'"

    word_lower = word_text.lower().strip()
    if word_lower in QUIET_SEGMENT_BLACKLIST:
        if quiet:
            return True, f"Suspicious word in quiet segment: '{word_text}'"
        if low_conf:
            return True, f"Suspicious word with low confidence: '{word_text}'"

    red_flags = sum([quiet, low_conf, partial])
    if red_flags >= 2:
        flags = (
            (["quiet"] if quiet else [])
            + (["low_conf"] if low_conf else [])
            + (["pattern"] if partial else [])
        )
        return True, f"Multiple red flags ({', '.join(flags)}): '{word_text}'"

    return False, "OK"


# ---------------------------------------------------------------------------
# Per-word decision: DESKTOP track
# ---------------------------------------------------------------------------

def should_filter_word_desktop(
    word_text: str,
    word_data: dict,
    audio_path: Optional[str],
    start_time: float,
    end_time: float,
    loop_flagged: bool = False,
) -> Tuple[bool, str]:
    """
    Desktop-track filtering.  More aggressive because game audio is noisy.
    """
    word_lower = word_text.lower().strip()
    exact, partial = matches_hallucination_pattern(word_text)
    low_conf = has_low_confidence(word_data, threshold=0.45)

    # 1. Loop / repetition flag from sequence analysis
    if loop_flagged:
        return True, f"Part of a repeating loop: '{word_text}'"

    # 2. Shared blacklist exact match
    if exact:
        return True, f"Exact blacklist match: '{word_text}'"

    # 3. Desktop-specific phrase blacklist
    if word_lower in DESKTOP_PHRASE_BLACKLIST:
        return True, f"Desktop phrase blacklist: '{word_text}'"

    # 4. Single-word SFX trigger (very short utterance, suspicious word)
    duration = end_time - start_time
    if duration < 0.25 and word_lower in DESKTOP_WORD_BLACKLIST:
        return True, f"Short SFX trigger word: '{word_text}' ({duration:.2f}s)"

    # 5. Music lyric signal words
    if word_lower in MUSIC_LYRIC_SIGNALS:
        return True, f"Music lyric signal word: '{word_text}'"

    # 6. Incoherent / non-word
    if not _word_is_coherent(word_lower):
        return True, f"Incoherent token: '{word_text}'"

    # 7. Spectral / speech-likelihood gate (most expensive — run last)
    has_speech = desktop_audio_has_speech(audio_path, start_time, end_time)
    if not has_speech:
        # Audio looks like music/SFX; apply stricter checks
        if low_conf:
            return True, f"Non-speech audio + low confidence: '{word_text}'"
        if partial:
            return True, f"Non-speech audio + pattern match: '{word_text}'"
        # Even without a blacklist hit, isolated single filler words in
        # music/SFX windows are almost always hallucinations.
        if word_lower in DESKTOP_WORD_BLACKLIST:
            return True, f"Non-speech audio + filler word: '{word_text}'"

    # 8. Partial pattern with low confidence (permissive combo)
    if partial and low_conf:
        return True, f"Pattern match + low confidence: '{word_text}'"

    return False, "OK"


# ---------------------------------------------------------------------------
# Public API — filter_hallucinations (called from transcriber)
# ---------------------------------------------------------------------------

def filter_hallucinations(
    transcriptions: List[str],
    audio_path: Optional[str],
    track_name: str = "",
    log_func=None,
) -> List[str]:
    """
    Filter hallucinations from transcription lines ("start-end: text").

    Routes to mic or desktop strategy based on track_name.
    """
    log = log_func or print

    if not transcriptions:
        return transcriptions

    is_desktop = "desktop" in track_name.lower() or "track 3" in track_name.lower()

    if not LIBROSA_AVAILABLE:
        log(f"⚠️  librosa not available — hallucination filtering limited for {track_name}")

    log(f"🔍 Hallucination filter: {track_name} ({'desktop' if is_desktop else 'mic'} mode, {len(transcriptions)} words)")

    # Parse all lines
    parsed: List[Tuple[float, float, str]] = []
    raw_lines: List[str] = []
    for line in transcriptions:
        try:
            time_part, text = line.split(":", 1)
            s, e = time_part.split("-")
            parsed.append((float(s), float(e), text.strip()))
            raw_lines.append(line)
        except ValueError:
            parsed.append((-1.0, -1.0, ""))
            raw_lines.append(line)

    # Desktop: run loop/repetition detection across the whole sequence first
    loop_flags: List[bool] = [False] * len(parsed)
    if is_desktop:
        valid = [(i, p) for i, p in enumerate(parsed) if p[0] >= 0]
        if valid:
            idxs, words = zip(*valid)
            flagged_local = detect_looping_sequence(list(words))
            for local_i in flagged_local:
                loop_flags[idxs[local_i]] = True

    filtered: List[str] = []
    removed = 0

    for i, ((start, end, text), line) in enumerate(zip(parsed, raw_lines)):
        if start < 0:
            filtered.append(line)
            continue

        if is_desktop:
            drop, reason = should_filter_word_desktop(
                text, {}, audio_path, start, end, loop_flagged=loop_flags[i]
            )
        else:
            drop, reason = should_filter_word_mic(
                text, {}, audio_path, start, end
            )

        if drop:
            removed += 1
            log(f"   ✂️  Filtered [{track_name}]: {reason}")
        else:
            filtered.append(line)

    log(
        f"   ✅ Done: {removed} removed, {len(filtered)} kept "
        f"({len(transcriptions)} total) for {track_name}"
    )
    return filtered


# ---------------------------------------------------------------------------
# WhisperX path (passes confidence data directly)
# ---------------------------------------------------------------------------

def filter_hallucinations_with_whisperx_data(
    segments: list,
    audio_path: Optional[str],
    track_name: str = "",
    log_func=None,
) -> list:
    log = log_func or print
    if not segments:
        return segments

    is_desktop = "desktop" in track_name.lower() or "track 3" in track_name.lower()
    log(f"🔍 WhisperX hallucination filter: {track_name} ({'desktop' if is_desktop else 'mic'} mode)")

    # For desktop: build word list for loop detection first
    loop_index: dict = {}
    if is_desktop:
        all_words_flat: List[Tuple[float, float, str]] = []
        for seg in segments:
            for w in seg.get("words", []):
                all_words_flat.append(
                    (w.get("start", 0), w.get("end", 0), w.get("word", "").strip())
                )
        flagged_local = detect_looping_sequence(all_words_flat)
        for idx in flagged_local:
            t = all_words_flat[idx][0]
            loop_index[t] = True

    filtered_segments = []
    total_removed = 0

    for segment in segments:
        if "words" not in segment:
            filtered_segments.append(segment)
            continue

        kept = []
        for w in segment["words"]:
            text = w.get("word", "").strip()
            start = w.get("start", 0)
            end = w.get("end", 0)
            is_loop = loop_index.get(start, False)

            if is_desktop:
                drop, reason = should_filter_word_desktop(
                    text, w, audio_path, start, end, loop_flagged=is_loop
                )
            else:
                drop, reason = should_filter_word_mic(text, w, audio_path, start, end)

            if drop:
                total_removed += 1
                log(f"   ✂️  Filtered [{track_name}]: {reason}")
            else:
                kept.append(w)

        if kept:
            seg_copy = dict(segment)
            seg_copy["words"] = kept
            filtered_segments.append(seg_copy)

    log(f"   ✅ WhisperX filter done: {total_removed} words removed for {track_name}")
    return filtered_segments


# ---------------------------------------------------------------------------
# Legacy alias kept for callers that import should_filter_word directly
# ---------------------------------------------------------------------------

def should_filter_word(
    word_text: str,
    word_data: dict,
    audio_path: Optional[str],
    start_time: float,
    end_time: float,
    log_func=None,
    is_desktop: bool = False,
) -> Tuple[bool, str]:
    if is_desktop:
        return should_filter_word_desktop(word_text, word_data, audio_path, start_time, end_time)
    return should_filter_word_mic(word_text, word_data, audio_path, start_time, end_time)