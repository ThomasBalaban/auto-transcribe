"""
Enhanced onset detection system for gaming content.
Replaces chunking with intelligent sound event detection.
"""

import numpy as np
import librosa
import random
from typing import List, Dict, Tuple


class GamingOnsetDetector:
    """Multi-tier onset detection optimized for gaming content"""
    
    def __init__(self, sensitivity: float = 0.5, log_func=None):
        self.sensitivity = sensitivity
        self.log_func = log_func or print
        
        # Gaming-optimized parameters
        self.sample_rate = 48000
        self.min_energy_threshold = 0.001  # Slightly higher than before
        
        # Multi-tier onset detection thresholds
        self.major_onset_threshold = 0.7 - (sensitivity * 0.2)     # 0.5-0.7
        self.medium_onset_threshold = 0.4 - (sensitivity * 0.15)   # 0.25-0.4  
        self.quick_onset_threshold = 0.2 - (sensitivity * 0.1)     # 0.1-0.2
        
        # Timing constraints for gaming
        self.major_min_spacing = 1.0    # 1 second between major events
        self.medium_min_spacing = 0.3   # 300ms between medium events
        self.quick_min_spacing = 0.15   # 150ms between quick events
        
        self.log_func(f"🎯 Gaming onset detector initialized")
        self.log_func(f"   Sensitivity: {sensitivity}")
        self.log_func(f"   Thresholds: Major={self.major_onset_threshold:.2f}, Medium={self.medium_onset_threshold:.2f}, Quick={self.quick_onset_threshold:.2f}")

    def detect_multi_tier_onsets(self, audio: np.ndarray, sr: int) -> Dict[str, List[float]]:
        """Detect onsets at multiple sensitivity levels"""
        
        # Tier 1: Major events (explosions, big impacts)
        major_onsets = librosa.onset.onset_detect(
            y=audio, 
            sr=sr,
            units='time',
            threshold=self.major_onset_threshold,
            pre_max=0.05,   # Must be peak for 50ms
            post_max=0.05,
            wait=self.major_min_spacing
        )
        
        # Tier 2: Medium events (gunshots, crashes)  
        medium_onsets = librosa.onset.onset_detect(
            y=audio,
            sr=sr, 
            units='time',
            threshold=self.medium_onset_threshold,
            pre_max=0.03,   # Must be peak for 30ms
            post_max=0.03,
            wait=self.medium_min_spacing
        )
        
        # Tier 3: Quick events (UI sounds, small impacts)
        quick_onsets = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            units='time', 
            threshold=self.quick_onset_threshold,
            pre_max=0.02,   # Must be peak for 20ms
            post_max=0.02,
            wait=self.quick_min_spacing
        )
        
        return {
            'major': major_onsets.tolist(),
            'medium': medium_onsets.tolist(), 
            'quick': quick_onsets.tolist()
        }

    def detect_rapid_sequences(self, onsets: List[float], max_gap: float = 0.4) -> List[Tuple[float, float, int]]:
        """Detect rapid-fire sequences (machine gun, explosions)"""
        if not onsets:
            return []
            
        sequences = []
        current_seq = []
        
        for onset in sorted(onsets):
            if current_seq and (onset - current_seq[-1]) > max_gap:
                # End current sequence, start new one
                if len(current_seq) >= 3:  # 3+ rapid sounds = sequence
                    sequences.append((current_seq[0], current_seq[-1], len(current_seq)))
                current_seq = [onset]
            else:
                current_seq.append(onset)
        
        # Handle final sequence
        if len(current_seq) >= 3:
            sequences.append((current_seq[0], current_seq[-1], len(current_seq)))
        
        return sequences

    def filter_rapid_sequences(self, onsets: List[float]) -> List[float]:
        """Replace rapid sequences with first/last onsets only"""
        if not onsets:
            return []
            
        sequences = self.detect_rapid_sequences(onsets)
        filtered_onsets = []
        sequence_ranges = [(seq[0], seq[1]) for seq in sequences]
        
        for onset in onsets:
            # Check if this onset is part of a sequence
            in_sequence = False
            for seq_start, seq_end in sequence_ranges:
                if seq_start <= onset <= seq_end:
                    # Only keep first and last of sequence
                    if abs(onset - seq_start) < 0.01 or abs(onset - seq_end) < 0.01:
                        filtered_onsets.append(onset)
                    in_sequence = True
                    break
            
            if not in_sequence:
                # Keep individual onsets
                filtered_onsets.append(onset)
        
        return sorted(list(set(filtered_onsets)))  # Remove duplicates

    def merge_onset_tiers(self, major_onsets: List[float], medium_onsets: List[float], 
                         quick_onsets: List[float]) -> List[Dict]:
        """Intelligently merge onset tiers with priorities"""
        
        # Filter rapid sequences in each tier
        major_filtered = self.filter_rapid_sequences(major_onsets)
        medium_filtered = self.filter_rapid_sequences(medium_onsets)
        quick_filtered = self.filter_rapid_sequences(quick_onsets)
        
        self.log_func(f"🔍 Raw onsets: Major={len(major_onsets)}, Medium={len(medium_onsets)}, Quick={len(quick_onsets)}")
        self.log_func(f"🔧 After sequence filtering: Major={len(major_filtered)}, Medium={len(medium_filtered)}, Quick={len(quick_filtered)}")
        
        # Create onset events with priorities
        events = []
        
        # Major events (highest priority)
        for onset in major_filtered:
            events.append({
                'time': onset,
                'tier': 'major',
                'priority': 3,
                'min_spacing': self.major_min_spacing
            })
        
        # Medium events
        for onset in medium_filtered:
            events.append({
                'time': onset,
                'tier': 'medium', 
                'priority': 2,
                'min_spacing': self.medium_min_spacing
            })
        
        # Quick events (lowest priority)
        for onset in quick_filtered:
            events.append({
                'time': onset,
                'tier': 'quick',
                'priority': 1,
                'min_spacing': self.quick_min_spacing
            })
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        # Remove conflicts (higher priority wins)
        filtered_events = []
        for event in events:
            # Check for conflicts with existing events
            conflict = False
            for existing in filtered_events:
                time_diff = abs(event['time'] - existing['time'])
                min_spacing = max(event['min_spacing'], existing['min_spacing'])
                
                if time_diff < min_spacing:
                    # Conflict detected - keep higher priority
                    if event['priority'] > existing['priority']:
                        # Remove existing, add current
                        filtered_events.remove(existing)
                        break
                    else:
                        # Skip current event
                        conflict = True
                        break
            
            if not conflict:
                filtered_events.append(event)
        
        self.log_func(f"✅ Final merged onsets: {len(filtered_events)} events")
        return filtered_events

    def calculate_onset_energy(self, audio: np.ndarray, onset_time: float, sr: int) -> float:
        """Calculate energy around onset for effect sizing"""
        window_start = max(0, int((onset_time - 0.1) * sr))
        window_end = min(len(audio), int((onset_time + 0.3) * sr))
        window = audio[window_start:window_end]
        
        if len(window) == 0:
            return 0.0
            
        return float(np.sqrt(np.mean(window**2)))

    def classify_onset_type(self, audio: np.ndarray, onset_time: float, sr: int) -> str:
        """Classify onset based on spectral characteristics"""
        window_start = max(0, int((onset_time - 0.1) * sr))
        window_end = min(len(audio), int((onset_time + 0.3) * sr))
        window = audio[window_start:window_end]
        
        if len(window) == 0:
            return "UNKNOWN"
        
        # Calculate spectral features
        stft = librosa.stft(window)
        magnitude = np.abs(stft)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0, 0]
        
        # Spectral rolloff (frequency distribution)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0, 0]
        
        # Energy distribution
        low_energy = np.sum(magnitude[:magnitude.shape[0]//3])  # Low frequencies
        high_energy = np.sum(magnitude[2*magnitude.shape[0]//3:])  # High frequencies
        
        # Classification logic
        if spectral_centroid < 500:
            return "LOW_FREQ"    # Explosions, bass impacts
        elif spectral_centroid > 2000:
            return "HIGH_FREQ"   # Gunshots, metal impacts  
        elif high_energy > low_energy * 2:
            return "SHARP"       # Crisp impacts, clicks
        else:
            return "GENERAL"     # General impacts

    def detect_gaming_onsets(self, audio_path: str) -> List[Dict]:
        """Main onset detection pipeline for gaming content"""
        try:
            self.log_func(f"\n🚀 Gaming onset detection: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_duration = len(audio) / sr
            
            self.log_func(f"📊 Audio loaded: {audio_duration:.2f}s at {sr}Hz")
            
            # Multi-tier onset detection
            onset_tiers = self.detect_multi_tier_onsets(audio, sr)
            
            # Merge and filter onsets
            merged_onsets = self.merge_onset_tiers(
                onset_tiers['major'],
                onset_tiers['medium'], 
                onset_tiers['quick']
            )
            
            # Create full event objects
            events = []
            for onset_event in merged_onsets:
                onset_time = onset_event['time']
                
                # Calculate onset characteristics
                energy = self.calculate_onset_energy(audio, onset_time, sr)
                onset_type = self.classify_onset_type(audio, onset_time, sr)
                
                # Skip very quiet onsets
                if energy < self.min_energy_threshold:
                    continue
                
                event = {
                    'time': onset_time,
                    'tier': onset_event['tier'],
                    'energy': energy,
                    'onset_type': onset_type,
                    'confidence': 0.8,  # Will be updated by CLAP analysis
                    'priority': onset_event['priority']
                }
                
                events.append(event)
                self.log_func(f"   🎯 {onset_time:.2f}s: {onset_event['tier'].upper()} {onset_type} (energy: {energy:.4f})")
            
            self.log_func(f"🎉 Gaming onset detection complete: {len(events)} events")
            return events
            
        except Exception as e:
            self.log_func(f"💥 Gaming onset detection failed: {e}")
            return []


def test_gaming_onset_detection():
    """Test the gaming onset detection system"""
    print("Testing gaming onset detection...")
    
    detector = GamingOnsetDetector(sensitivity=0.5, log_func=print)
    
    # Test with synthetic gaming-like audio
    duration = 10  # 10 seconds
    sr = 48000
    t = np.linspace(0, duration, duration * sr)
    
    # Create synthetic gaming audio
    audio = np.zeros_like(t)
    
    # Add some "events"
    # Explosion at 2s
    explosion_start = int(2 * sr)
    explosion_end = int(2.5 * sr)
    audio[explosion_start:explosion_end] = 0.8 * np.sin(2 * np.pi * 100 * t[explosion_start:explosion_end])
    
    # Gunshots at 4s, 4.2s, 4.4s (rapid sequence)
    for shot_time in [4.0, 4.2, 4.4]:
        shot_idx = int(shot_time * sr)
        audio[shot_idx:shot_idx+1000] += 0.6 * np.random.random(1000)
    
    # Impact at 7s
    impact_idx = int(7 * sr)
    audio[impact_idx:impact_idx+5000] += 0.5 * np.sin(2 * np.pi * 200 * t[impact_idx:impact_idx+5000])
    
    # Save test audio
    import soundfile as sf
    sf.write("test_gaming_audio.wav", audio, sr)
    
    # Test detection
    events = detector.detect_gaming_onsets("test_gaming_audio.wav")
    
    print(f"\nDetected {len(events)} gaming events:")
    for event in events:
        print(f"  {event['time']:.2f}s: {event['tier']} {event['onset_type']} (energy: {event['energy']:.4f})")
    
    return events


if __name__ == "__main__":
    test_gaming_onset_detection()