# onset_detector.py

import numpy as np
import librosa
from typing import List, Dict

class GamingOnsetDetector:
    """Multi-tier onset detection optimized for gaming content."""

    def __init__(self, sensitivity: float = 0.5, log_func=None):
        self.sensitivity = sensitivity
        self.log_func = log_func or print
        self.sample_rate = 48000
        self.major_onset_threshold = 0.7 - (sensitivity * 0.2)
        self.medium_onset_threshold = 0.4 - (sensitivity * 0.15)
        self.quick_onset_threshold = 0.2 - (sensitivity * 0.1)
        self.major_min_spacing = 1.0
        self.medium_min_spacing = 0.3
        self.quick_min_spacing = 0.15
        self.log_func(f"🎯 Gaming onset detector initialized")

    def detect_multi_tier_onsets(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        # ... (this function remains the same)
        try:
            major_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.major_onset_threshold, wait=self.major_min_spacing)
            medium_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.medium_onset_threshold, wait=self.medium_min_spacing)
            quick_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.quick_onset_threshold, wait=self.quick_min_spacing)
            return {'major': major_onsets.tolist(), 'medium': medium_onsets.tolist(), 'quick': quick_onsets.tolist()}
        except Exception:
            basic_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
            return {'major': basic_onsets.tolist(), 'medium': [], 'quick': []}


    def merge_onset_tiers(self, major_onsets: List[float], medium_onsets: List[float], quick_onsets: List[float]) -> List[Dict]:
        # ... (this function remains the same)
        events = []
        for onset in major_onsets: events.append({'time': onset, 'tier': 'major', 'priority': 3, 'min_spacing': self.major_min_spacing})
        for onset in medium_onsets: events.append({'time': onset, 'tier': 'medium', 'priority': 2, 'min_spacing': self.medium_min_spacing})
        for onset in quick_onsets: events.append({'time': onset, 'tier': 'quick', 'priority': 1, 'min_spacing': self.quick_min_spacing})
        events.sort(key=lambda x: x['time'])
        filtered_events = []
        for event in events:
            conflict = False
            for existing in filtered_events:
                if abs(event['time'] - existing['time']) < max(event['min_spacing'], existing['min_spacing']):
                    if event['priority'] > existing['priority']:
                        filtered_events.remove(existing)
                    else:
                        conflict = True
                    break
            if not conflict:
                filtered_events.append(event)
        return filtered_events


    def calculate_onset_energy_and_peak(self, audio: np.ndarray, onset_time: float, sr: int) -> tuple[float, float]:
        """Calculates the energy and the precise peak time of the sound event."""
        window_start_s = max(0, onset_time - 0.05) # Look slightly before onset
        window_end_s = onset_time + 0.4 # Look up to 400ms after for the peak
        
        start_sample = int(window_start_s * sr)
        end_sample = int(window_end_s * sr)
        
        window = audio[start_sample:end_sample]
        if len(window) == 0:
            return 0.0, onset_time

        energy = float(np.sqrt(np.mean(window**2)))
        
        # Find the index of the absolute maximum value in the window
        peak_index = np.argmax(np.abs(window))
        # Convert the local index to a global timestamp
        peak_time = window_start_s + (peak_index / sr)
        
        return energy, peak_time

    def classify_onset_type(self, audio: np.ndarray, onset_time: float, sr: int) -> str:
        # ... (this function remains the same)
        window_start = max(0, int((onset_time - 0.1) * sr))
        window_end = min(len(audio), int((onset_time + 0.3) * sr))
        window = audio[window_start:window_end]
        if len(window) == 0: return "UNKNOWN"
        stft = librosa.stft(window)
        magnitude = np.abs(stft)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0, 0]
        if spectral_centroid < 500: return "LOW_FREQ"
        elif spectral_centroid > 2000: return "HIGH_FREQ"
        else: return "GENERAL"

    def detect_gaming_onsets(self, audio_path: str) -> List[Dict]:
        """Main onset detection pipeline, now including peak time calculation."""
        try:
            self.log_func(f"\n🚀 Gaming onset detection: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            self.log_func(f"📊 Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")
            
            onset_tiers = self.detect_multi_tier_onsets(audio, sr)
            merged_onsets = self.merge_onset_tiers(onset_tiers['major'], onset_tiers['medium'], onset_tiers['quick'])
            
            events = []
            for onset_event in merged_onsets:
                onset_time = onset_event['time']
                
                # Get both energy and the new peak_time
                energy, peak_time = self.calculate_onset_energy_and_peak(audio, onset_time, sr)
                
                if energy < 0.001: continue

                event = {
                    'time': onset_time,
                    'peak_time': peak_time, # Add the new peak time
                    'tier': onset_event['tier'],
                    'energy': energy,
                    'onset_type': self.classify_onset_type(audio, onset_time, sr),
                }
                events.append(event)
            
            self.log_func(f"🎉 Gaming onset detection complete: {len(events)} events")
            return events
            
        except Exception as e:
            self.log_func(f"💥 Gaming onset detection failed: {e}")
            return []