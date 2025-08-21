# onset_detector.py - CLEAN ENHANCED VERSION

import numpy as np
import librosa
from typing import List, Dict
import os
from scipy import signal

class GamingOnsetDetector:
    """Enhanced multi-tier onset detection optimized for gaming content with precise timing."""

    def __init__(self, sensitivity: float = 0.5, log_func=None, min_energy_threshold: float = 0.015):
        self.sensitivity = sensitivity
        self.log_func = log_func or print
        self.sample_rate = 48000
        self.min_energy_threshold = min_energy_threshold
        
        # MORE AGGRESSIVE thresholds for better detection timing
        self.major_onset_threshold = 0.45 - (sensitivity * 0.3)    # Reduced from 0.65
        self.medium_onset_threshold = 0.25 - (sensitivity * 0.25)  # Reduced from 0.35  
        self.quick_onset_threshold = 0.10 - (sensitivity * 0.15)   # Reduced from 0.15
        
        # TIGHTER spacing to catch more events
        self.major_min_spacing = 0.6    # Reduced from 0.8
        self.medium_min_spacing = 0.2   # Reduced from 0.25
        self.quick_min_spacing = 0.08   # Reduced from 0.12
        
        self.log_func(f"🎯 Enhanced gaming onset detector initialized:")
        self.log_func(f"   - Sensitivity: {sensitivity} (lower = more sensitive)")
        self.log_func(f"   - Min energy threshold: {self.min_energy_threshold}")
        self.log_func(f"   - Major threshold: {self.major_onset_threshold}")
        self.log_func(f"   - AGGRESSIVE MODE: Earlier detection enabled")

    def detect_multi_tier_onsets(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        """Enhanced onset detection with multiple algorithms."""
        try:
            # Method 1: Standard onset detection (good for percussive sounds)
            major_onsets = librosa.onset.onset_detect(
                y=audio, sr=sr, units='time', 
                delta=self.major_onset_threshold, 
                wait=self.major_min_spacing,
                backtrack=True  # More precise timing
            )
            
            medium_onsets = librosa.onset.onset_detect(
                y=audio, sr=sr, units='time',
                delta=self.medium_onset_threshold,
                wait=self.medium_min_spacing,
                backtrack=True
            )
            
            # Method 2: Spectral flux onset detection (good for impacts)
            stft = librosa.stft(audio, hop_length=512)
            spectral_flux = np.sum(np.maximum(0, np.diff(np.abs(stft), axis=1)), axis=0)
            spectral_flux = spectral_flux / (np.max(spectral_flux) + 1e-8)
            
            # Find peaks in spectral flux
            flux_peaks, _ = signal.find_peaks(
                spectral_flux, 
                prominence=0.3 - (self.sensitivity * 0.15),
                distance=int(self.quick_min_spacing * sr / 512)
            )
            flux_onsets = flux_peaks * 512 / sr
            
            # Method 3: Energy-based detection (good for sudden impacts)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hops
            
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames**2, axis=0)
            energy_diff = np.diff(energy)
            
            # Normalize energy difference
            energy_diff = energy_diff / (np.max(np.abs(energy_diff)) + 1e-8)
            
            # Find energy spikes
            energy_threshold = 0.4 - (self.sensitivity * 0.2)
            energy_peaks, _ = signal.find_peaks(
                energy_diff, 
                height=energy_threshold,
                distance=int(self.quick_min_spacing * sr / hop_length)
            )
            energy_onsets = energy_peaks * hop_length / sr
            
            # Combine all methods for quick onsets
            all_quick_onsets = np.concatenate([
                medium_onsets,
                flux_onsets,
                energy_onsets
            ])
            
            # Remove duplicates and sort
            quick_onsets = np.unique(np.round(all_quick_onsets, 2))
            
            return {
                'major': major_onsets.tolist(), 
                'medium': medium_onsets.tolist(), 
                'quick': quick_onsets.tolist()
            }
            
        except Exception as e:
            self.log_func(f"Enhanced onset detection failed, using basic: {e}")
            basic_onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
            return {'major': basic_onsets.tolist(), 'medium': [], 'quick': []}

    def merge_onset_tiers(self, major_onsets: List[float], medium_onsets: List[float], quick_onsets: List[float]) -> List[Dict]:
        """Enhanced merging with better priority handling."""
        events = []
        
        # Add all onsets with their tiers
        for onset in major_onsets: 
            events.append({
                'time': onset, 
                'tier': 'major', 
                'priority': 3, 
                'min_spacing': self.major_min_spacing
            })
        for onset in medium_onsets: 
            events.append({
                'time': onset, 
                'tier': 'medium', 
                'priority': 2, 
                'min_spacing': self.medium_min_spacing
            })
        for onset in quick_onsets: 
            events.append({
                'time': onset, 
                'tier': 'quick', 
                'priority': 1, 
                'min_spacing': self.quick_min_spacing
            })
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        if not events:
            return []
        
        # Enhanced merging logic
        filtered_events = [events[0]]
        
        for current_event in events[1:]:
            should_add = True
            current_time = current_event['time']
            
            # Check against recent events (not just the last one)
            for i in range(len(filtered_events) - 1, max(-1, len(filtered_events) - 3), -1):
                recent_event = filtered_events[i]
                time_diff = abs(current_time - recent_event['time'])
                required_spacing = max(current_event['min_spacing'], recent_event['min_spacing'])
                
                if time_diff < required_spacing:
                    # Choose which event to keep based on priority and recency
                    if current_event['priority'] > recent_event['priority']:
                        # Replace the recent event
                        filtered_events[i] = current_event
                        should_add = False
                        break
                    else:
                        # Keep the recent event, skip current
                        should_add = False
                        break
            
            if should_add:
                filtered_events.append(current_event)
        
        return filtered_events

    def find_precise_peak_time(self, audio: np.ndarray, onset_time: float, sr: int) -> float:
        """Find the precise peak time for sharp transients like gunshots - LOOK BACKWARDS."""
        # For sharp sounds, the actual attack often happens BEFORE the detected onset
        search_window_s = 0.5  # 500ms window - look further back
        search_start = max(0, int((onset_time - search_window_s * 0.8) * sr))  # Look 80% backwards
        search_end = min(len(audio), int((onset_time + search_window_s * 0.2) * sr))  # Look 20% forwards
        
        search_region = audio[search_start:search_end]
        
        if len(search_region) < 100:
            return onset_time
        
        # Method 1: Find steepest positive derivative (sharpest attack) - LOOK BACKWARDS
        derivatives = np.diff(search_region)
        positive_derivs = np.where(derivatives > 0, derivatives, 0)
        
        # Find all significant peaks in the derivatives
        peak_threshold = np.max(positive_derivs) * 0.7  # 70% of max derivative
        significant_peaks = np.where(positive_derivs > peak_threshold)[0]
        
        if len(significant_peaks) > 0:
            # Choose the EARLIEST significant peak (attack transient)
            earliest_peak_idx = significant_peaks[0]
            precise_peak = (search_start + earliest_peak_idx) / sr
        else:
            # Fallback: maximum absolute value in the early part of the window  
            early_region_end = int(len(search_region) * 0.6)  # First 60% of window
            early_region = search_region[:early_region_end]
            if len(early_region) > 0:
                early_peak_idx = np.argmax(np.abs(early_region))
                precise_peak = (search_start + early_peak_idx) / sr
            else:
                precise_peak = onset_time
        
        # Ensure we don't go too far back in time
        earliest_allowed = onset_time - 0.4  # Max 400ms before detection
        precise_peak = max(earliest_allowed, precise_peak)
        
        return precise_peak

    def calculate_onset_properties(self, audio: np.ndarray, onset_time: float, sr: int) -> tuple[float, float, float]:
        """Enhanced property calculation with PRECISE peak timing for synchronization."""
        window_start_s = max(0, onset_time - 0.1)
        window_end_s = onset_time + 0.5
        
        start_sample = int(window_start_s * sr)
        end_sample = int(window_end_s * sr)
        
        window = audio[start_sample:end_sample]
        if len(window) < 1024:
            return 0.0, onset_time, 0.0

        # Energy calculation
        energy = float(np.sqrt(np.mean(window**2)))
        
        # PRECISE peak detection - this is key for gunshots
        precise_peak = self.find_precise_peak_time(audio, onset_time, sr)
        
        # Enhanced spectral flux calculation
        try:
            S = librosa.stft(window, n_fft=2048, hop_length=512)
            magnitude = np.abs(S)
            flux = np.sum(np.maximum(0, np.diff(magnitude, axis=1)), axis=0)
            spectral_flux = float(np.mean(flux) / (np.max(flux) + 1e-8))
            
            # Boost flux score for sharp transients
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            hf_mask = freqs > 2000
            hf_energy = np.mean(magnitude[hf_mask, :])
            total_energy = np.mean(magnitude) + 1e-8
            hf_ratio = float(hf_energy / total_energy)
            
            # Sharp sounds get higher flux scores
            if hf_ratio > 0.3:  # High frequency content
                spectral_flux *= (1.0 + hf_ratio)
                
        except Exception:
            spectral_flux = 0.0
        
        return energy, precise_peak, spectral_flux

    def classify_onset_type(self, audio: np.ndarray, onset_time: float, sr: int) -> str:
        """Enhanced onset classification with better frequency analysis."""
        window_start = max(0, int((onset_time - 0.15) * sr))
        window_end = min(len(audio), int((onset_time + 0.35) * sr))
        window = audio[window_start:window_end]
        
        if len(window) < 1024: 
            return "UNKNOWN"
        
        try:
            # More detailed spectral analysis
            stft = librosa.stft(window, n_fft=2048)
            magnitude = np.abs(stft)
            
            # Calculate spectral centroid and other features
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0, 0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0, 0]
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0, 0]
            
            # Zero crossing rate for percussive content
            zcr = librosa.feature.zero_crossing_rate(window)[0, 0]
            
            # Enhanced classification logic
            if spectral_centroid < 400 and spectral_rolloff < 1000:
                return "LOW_FREQ"  # Bass, explosions, impacts
            elif spectral_centroid > 3000 or zcr > 0.15:
                return "HIGH_FREQ"  # Gunshots, metal, glass
            elif spectral_bandwidth > 2000:
                return "BROADBAND"  # Complex impacts, crashes
            else:
                return "GENERAL"
                
        except Exception:
            return "GENERAL"

    def detect_gaming_onsets(self, audio_path: str) -> List[Dict]:
        """Enhanced main onset detection pipeline."""
        try:
            self.log_func(f"\n🚀 Enhanced gaming onset detection: {os.path.basename(audio_path)}")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            self.log_func(f"📊 Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Pre-process audio for better detection
            audio = self._preprocess_audio(audio, sr)
            
            onset_tiers = self.detect_multi_tier_onsets(audio, sr)
            merged_onsets = self.merge_onset_tiers(
                onset_tiers['major'], 
                onset_tiers['medium'], 
                onset_tiers['quick']
            )
            
            self.log_func(f"🎤 Found {len(merged_onsets)} potential onsets before energy filtering.")
            
            events = []
            for onset_event in merged_onsets:
                onset_time = onset_event['time']
                
                energy, peak_time, spectral_flux = self.calculate_onset_properties(audio, onset_time, sr)
                
                # Enhanced energy filtering
                effective_threshold = self._calculate_adaptive_threshold(audio, onset_time, sr)
                
                if energy < effective_threshold:
                    self.log_func(f"  -> Filtered quiet event at {onset_time:.2f}s "
                                f"(Energy: {energy:.4f} < {effective_threshold:.4f})")
                    continue

                # Additional quality checks
                onset_type = self.classify_onset_type(audio, onset_time, sr)
                
                # Calculate confidence score
                confidence = self._calculate_onset_confidence(energy, spectral_flux, onset_event['tier'])
                
                event = {
                    'time': onset_time,
                    'peak_time': peak_time,
                    'tier': onset_event['tier'],
                    'energy': energy,
                    'spectral_flux': spectral_flux,
                    'onset_type': onset_type,
                    'confidence': confidence
                }
                events.append(event)
            
            # Post-processing: remove very close duplicates
            events = self._remove_close_duplicates(events)
            
            self.log_func(f"🎉 Enhanced gaming onset detection complete: {len(events)} events passed all filters.")
            return events
            
        except Exception as e:
            self.log_func(f"💥 Enhanced gaming onset detection failed: {e}")
            import traceback
            self.log_func(traceback.format_exc())
            return []

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio for better onset detection."""
        # Light high-pass filter to remove low-frequency rumble
        from scipy.signal import butter, filtfilt
        
        try:
            nyquist = sr / 2
            low_cutoff = 40 / nyquist  # Remove very low frequencies
            b, a = butter(2, low_cutoff, btype='high')
            audio_filtered = filtfilt(b, a, audio)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_filtered))
            if max_val > 0:
                audio_filtered = audio_filtered / max_val * 0.95
                
            return audio_filtered
        except Exception:
            return audio  # Return original if filtering fails

    def _calculate_adaptive_threshold(self, audio: np.ndarray, onset_time: float, sr: int) -> float:
        """Calculate adaptive energy threshold based on local audio characteristics."""
        # Analyze energy in surrounding context
        context_start = max(0, int((onset_time - 5.0) * sr))
        context_end = min(len(audio), int((onset_time + 5.0) * sr))
        context_audio = audio[context_start:context_end]
        
        if len(context_audio) < sr:  # Less than 1 second of context
            return self.min_energy_threshold
        
        # Calculate RMS energy in frames
        frame_size = int(0.1 * sr)  # 100ms frames
        frame_energies = []
        
        for i in range(0, len(context_audio) - frame_size, frame_size // 2):
            frame = context_audio[i:i + frame_size]
            frame_energy = np.sqrt(np.mean(frame**2))
            frame_energies.append(frame_energy)
        
        if not frame_energies:
            return self.min_energy_threshold
        
        # Use percentile-based threshold
        background_level = np.percentile(frame_energies, 50)  # Median
        adaptive_threshold = max(
            self.min_energy_threshold,
            background_level * 1.5  # Must be 50% above background
        )
        
        return adaptive_threshold

    def _calculate_onset_confidence(self, energy: float, spectral_flux: float, tier: str) -> float:
        """Calculate confidence score for an onset."""
        # Base confidence from energy
        energy_confidence = min(1.0, energy / 0.1)  # Normalize to 0.1 as "high energy"
        
        # Confidence from spectral flux (transient sharpness)
        flux_confidence = min(1.0, spectral_flux * 2.0)
        
        # Tier-based confidence
        tier_confidences = {'major': 1.0, 'medium': 0.8, 'quick': 0.6}
        tier_confidence = tier_confidences.get(tier, 0.5)
        
        # Combined confidence
        overall_confidence = (energy_confidence * 0.4 + 
                            flux_confidence * 0.4 + 
                            tier_confidence * 0.2)
        
        return min(1.0, overall_confidence)

    def _remove_close_duplicates(self, events: List[Dict], min_separation: float = 0.05) -> List[Dict]:
        """Remove events that are very close together (likely duplicates)."""
        if len(events) <= 1:
            return events
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        filtered_events = [events[0]]
        
        for current_event in events[1:]:
            last_event = filtered_events[-1]
            time_diff = current_event['time'] - last_event['time']
            
            if time_diff < min_separation:
                # Keep the event with higher confidence
                if current_event['confidence'] > last_event['confidence']:
                    filtered_events[-1] = current_event
            else:
                filtered_events.append(current_event)
        
        return filtered_events