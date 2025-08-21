# onset_detector.py

import numpy as np
import librosa
from typing import List, Dict
import os

class GamingOnsetDetector:
    """Multi-tier onset detection optimized for gaming content with energy-based filtering."""

    def __init__(self, sensitivity: float = 0.5, log_func=None, min_energy_threshold: float = 0.02):
        self.sensitivity = sensitivity
        self.log_func = log_func or print
        self.sample_rate = 48000
        self.min_energy_threshold = min_energy_threshold
        
        self.major_onset_threshold = 0.7 - (sensitivity * 0.2)
        self.medium_onset_threshold = 0.4 - (sensitivity * 0.15)
        self.quick_onset_threshold = 0.2 - (sensitivity * 0.1)
        self.major_min_spacing = 1.0
        self.medium_min_spacing = 0.3
        self.quick_min_spacing = 0.15
        self.log_func(f"🎯 Gaming onset detector initialized with min energy threshold: {self.min_energy_threshold}")

    def detect_multi_tier_onsets(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        try:
            major_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.major_onset_threshold, wait=self.major_min_spacing)
            medium_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.medium_onset_threshold, wait=self.medium_min_spacing)
            quick_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=self.quick_onset_threshold, wait=self.quick_min_spacing)
            return {'major': major_onsets.tolist(), 'medium': medium_onsets.tolist(), 'quick': quick_onsets.tolist()}
        except Exception:
            basic_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
            return {'major': basic_onsets.tolist(), 'medium': [], 'quick': []}

    def merge_onset_tiers(self, major_onsets: List[float], medium_onsets: List[float], quick_onsets: List[float]) -> List[Dict]:
        events = []
        for onset in major_onsets: events.append({'time': onset, 'tier': 'major', 'priority': 3, 'min_spacing': self.major_min_spacing})
        for onset in medium_onsets: events.append({'time': onset, 'tier': 'medium', 'priority': 2, 'min_spacing': self.medium_min_spacing})
        for onset in quick_onsets: events.append({'time': onset, 'tier': 'quick', 'priority': 1, 'min_spacing': self.quick_min_spacing})
        events.sort(key=lambda x: x['time'])
        filtered_events = []
        if not events:
            return []
            
        filtered_events.append(events[0])
        for current_event in events[1:]:
            last_event = filtered_events[-1]
            if abs(current_event['time'] - last_event['time']) < max(current_event['min_spacing'], last_event['min_spacing']):
                if current_event['priority'] > last_event['priority']:
                    filtered_events[-1] = current_event
            else:
                filtered_events.append(current_event)
        return filtered_events

    def calculate_onset_properties(self, audio: np.ndarray, onset_time: float, sr: int) -> tuple[float, float, float]:
        """Calculates energy, peak time, and spectral flux of the sound event."""
        window_start_s = max(0, onset_time - 0.05)
        window_end_s = onset_time + 0.4
        
        start_sample = int(window_start_s * sr)
        end_sample = int(window_end_s * sr)
        
        window = audio[start_sample:end_sample]
        if len(window) < 2048: # Ensure window is large enough for STFT
            return 0.0, onset_time, 0.0

        energy = float(np.sqrt(np.mean(window**2)))
        
        peak_index = np.argmax(np.abs(window))
        peak_time = window_start_s + (peak_index / sr)
        
        # Calculate Spectral Flux for transientness
        S = librosa.stft(window)
        onset_env = librosa.onset.onset_strength(S=np.abs(S), sr=sr)
        spectral_flux = float(np.mean(onset_env))
        
        return energy, peak_time, spectral_flux

    def classify_onset_type(self, audio: np.ndarray, onset_time: float, sr: int) -> str:
        window_start = max(0, int((onset_time - 0.1) * sr))
        window_end = min(len(audio), int((onset_time + 0.3) * sr))
        window = audio[window_start:window_end]
        if len(window) < 2048: return "UNKNOWN"
        stft = librosa.stft(window)
        magnitude = np.abs(stft)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0, 0]
        if spectral_centroid < 500: return "LOW_FREQ"
        elif spectral_centroid > 2000: return "HIGH_FREQ"
        else: return "GENERAL"

    def detect_gaming_onsets(self, audio_path: str) -> List[Dict]:
        """Main onset detection pipeline, now including spectral flux calculation."""
        try:
            self.log_func(f"\n🚀 Gaming onset detection: {os.path.basename(audio_path)}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            self.log_func(f"📊 Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")
            
            onset_tiers = self.detect_multi_tier_onsets(audio, sr)
            merged_onsets = self.merge_onset_tiers(onset_tiers['major'], onset_tiers['medium'], onset_tiers['quick'])
            
            self.log_func(f"🎤 Found {len(merged_onsets)} potential onsets before energy filtering.")
            events = []
            for onset_event in merged_onsets:
                onset_time = onset_event['time']
                
                energy, peak_time, spectral_flux = self.calculate_onset_properties(audio, onset_time, sr)
                
                if energy < self.min_energy_threshold:
                    self.log_func(f"  -> Filtered quiet event at {onset_time:.2f}s (Energy: {energy:.4f})")
                    continue

                event = {
                    'time': onset_time,
                    'peak_time': peak_time,
                    'tier': onset_event['tier'],
                    'energy': energy,
                    'spectral_flux': spectral_flux, # Add sharpness score
                    'onset_type': self.classify_onset_type(audio, onset_time, sr),
                }
                events.append(event)
            
            self.log_func(f"🎉 Gaming onset detection complete: {len(events)} events passed the energy filter.")
            return events
            
        except Exception as e:
            self.log_func(f"💥 Gaming onset detection failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []