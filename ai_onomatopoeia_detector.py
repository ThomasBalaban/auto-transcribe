"""
Enhanced Onomatopoeia detection with AI-determined duration.
The AI determines duration based on sound characteristics without external filters.
Updated with overlapping chunks and peak detection for better timing accuracy.
"""

import os
import random
import numpy as np # type: ignore
import librosa # type: ignore
import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore
from sound_mappings import SOUND_MAPPINGS
from yamnet_mappings import YAMNET_CLASS_MAPPINGS
from sound_duration_profiles import sound_duration_profiles
from ai_thresholds import AI_THRESHOLDS

class AIOnomatopoeiaDetector:
    """Enhanced detector where AI determines onomatopoeia duration naturally"""
    
    def __init__(self, ai_sensitivity=0.5, log_func=None):
        """
        Initialize the AI-driven onomatopoeia detector.
        
        Args:
            ai_sensitivity (float): AI decision sensitivity (0.1-0.9, higher = more selective)
            log_func: Logging function
        """
        self.ai_sensitivity = ai_sensitivity
        self.log_func = log_func or print
        self.yamnet_model = None
        self.class_names = None
        self.recent_words = []
        self.max_recent_words = 5
        
        # Sound duration characteristics imported from separate file
        self.sound_duration_profiles = sound_duration_profiles
        
        self._load_yamnet_model()
    
    def _load_yamnet_model(self):
        """Load YAMNet model (same as original implementation)"""
        try:
            self.log_func("Loading YAMNet model for AI-driven onomatopoeia detection...")
            
            # Use the same loading logic as the original
            self._fix_ssl_certificates()
            self._setup_cache_directory()
            
            model_loaded = False
            try:
                self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
                model_loaded = True
                self.log_func("✓ YAMNet loaded successfully")
            except Exception as e:
                self.log_func(f"YAMNet load failed: {e}")
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                try:
                    self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
                    model_loaded = True
                    self.log_func("✓ YAMNet loaded with SSL fix")
                except Exception as e2:
                    self.log_func(f"All loading methods failed: {e2}")
            
            if model_loaded:
                # Load class names
                try:
                    class_map_path = tf.keras.utils.get_file(
                        'yamnet_class_map.csv',
                        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
                    )
                    with open(class_map_path) as f:
                        self.class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]
                    self.log_func(f"AI Onomatopoeia system ready with {len(self.class_names)} sound classes")
                except Exception as e:
                    self.log_func(f"Class names load failed: {e}")
                    self._create_fallback_class_names()
            else:
                raise Exception("Failed to load YAMNet model")
                
        except Exception as e:
            self.log_func(f"AI Onomatopoeia system initialization failed: {e}")
            self.yamnet_model = None
    
    def _fix_ssl_certificates(self):
        """Fix SSL certificates (same as original)"""
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
    
    def _setup_cache_directory(self):
        """Setup cache directory (same as original)"""
        try:
            cache_dir = os.path.expanduser("~/.cache/tensorflow_hub")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
        except Exception:
            pass
    
    def _create_fallback_class_names(self):
        """Create fallback class names (same as original)"""
        self.class_names = [
            'Silence', 'Speech', 'Music', 'Explosion', 'Gunshot, gunfire', 
            'Breaking', 'Glass', 'Slam', 'Thud', 'Bell', 'Buzzer', 
            'Beep, bleep', 'Click', 'Alarm', 'Whoosh, swoosh, swish', 
            'Pop', 'Whistle', 'Siren', 'Splash, splatter', 'Thunder', 
            'Rain', 'Applause', 'Footsteps', 'Knock', 'Car'
        ]
    
    def ai_determine_duration(self, sound_class, confidence_score, audio_context=None):
        """
        AI determines the natural duration for an onomatopoeia based on sound characteristics.
        
        Args:
            sound_class (str): The detected sound class
            confidence_score (float): YAMNet confidence (0.0 to 1.0)
            audio_context (dict): Optional audio analysis context
            
        Returns:
            float: Determined duration in seconds
        """
        # Get the sound profile
        profile = self.sound_duration_profiles.get(sound_class, {
            'base_duration': 0.8,
            'variability': 0.3, 
            'decay_type': 'medium'
        })
        
        base_duration = profile['base_duration']
        variability = profile['variability']
        decay_type = profile['decay_type']
        
        # AI decision factors:
        
        # 1. Confidence affects certainty - higher confidence = more typical duration
        confidence_factor = 0.7 + (confidence_score * 0.3)  # 0.7 to 1.0 range
        
        # 2. Sound decay type affects duration consistency
        decay_modifiers = {
            'instant': 0.8,    # Quick sounds are more predictable
            'sharp': 0.9,      # Sharp sounds have consistent duration
            'medium': 1.0,     # Standard variation
            'gradual': 1.2,    # Gradual sounds can vary more
            'sustained': 1.5,  # Sustained sounds have high variation
            'soft': 0.95       # Soft sounds are fairly consistent
        }
        
        decay_modifier = decay_modifiers.get(decay_type, 1.0)
        
        # 3. Add natural randomness (AI's "intuitive" variation)
        # Higher confidence = less random variation
        random_variation = random.uniform(-variability, variability) * (2.0 - confidence_factor)
        
        # 4. Apply decay type modifier to base duration
        adjusted_duration = base_duration * decay_modifier
        
        # 5. Add the random variation
        final_duration = adjusted_duration + random_variation
        
        # 6. Ensure reasonable bounds (AI won't go to extremes)
        min_duration = 0.1  # Never shorter than 100ms
        max_duration = 5.0  # Never longer than 5 seconds
        final_duration = max(min_duration, min(max_duration, final_duration))
        
        # 7. AI rounds to reasonable precision (100ms increments)
        final_duration = round(final_duration * 10) / 10
        
        self.log_func(f"AI determined duration for {sound_class}: {final_duration:.1f}s "
                     f"(base: {base_duration:.1f}s, confidence: {confidence_score:.2f}, "
                     f"decay: {decay_type})")
        
        return final_duration
    
    def get_onomatopoeia_word(self, sound_class):
        """Get onomatopoeia word (same as original)"""
        if sound_class not in SOUND_MAPPINGS:
            return None
        
        available_words = SOUND_MAPPINGS[sound_class]
        
        if self.recent_words:
            unused_words = [w for w in available_words if w not in self.recent_words]
            if unused_words:
                available_words = unused_words
        
        selected_word = random.choice(available_words)
        
        self.recent_words.append(selected_word)
        if len(self.recent_words) > self.max_recent_words:
            self.recent_words.pop(0)
        
        return selected_word
    
    def find_sound_peak_in_chunk(self, audio_chunk, chunk_start_time, sample_rate=16000):
        """
        Find the timing of the actual sound peak within the audio chunk.
        
        Args:
            audio_chunk (np.array): Audio data for the chunk
            chunk_start_time (float): Start time of this chunk
            sample_rate (int): Audio sample rate
            
        Returns:
            float: Absolute timestamp of the sound peak
        """
        try:
            # Calculate the RMS energy in small windows to find the peak
            window_size = int(0.1 * sample_rate)  # 100ms windows
            chunk_length = len(audio_chunk)
            
            if chunk_length < window_size:
                # Chunk too short, just use the middle
                return chunk_start_time + (chunk_length / sample_rate) / 2
            
            max_energy = 0
            peak_sample = chunk_length // 2  # Default to middle if no clear peak
            
            # Slide window through chunk to find peak energy
            for i in range(0, chunk_length - window_size, window_size // 2):
                window = audio_chunk[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))  # RMS energy
                
                if energy > max_energy:
                    max_energy = energy
                    peak_sample = i + window_size // 2  # Middle of the window
            
            # Convert sample position to absolute time
            peak_time_offset = peak_sample / sample_rate
            absolute_peak_time = chunk_start_time + peak_time_offset
            
            return absolute_peak_time
            
        except Exception as e:
            self.log_func(f"Error finding peak timing: {e}")
            # Fallback to chunk middle
            chunk_duration = len(audio_chunk) / sample_rate
            return chunk_start_time + chunk_duration / 2
    
    def detect_sounds_in_chunk_with_timing(self, audio_chunk, chunk_start_time, sample_rate=16000):
        """
        Detect sounds with improved timing by finding the actual peak within the chunk.
        
        Args:
            audio_chunk (np.array): Audio data
            chunk_start_time (float): Start time of this chunk
            sample_rate (int): Audio sample rate
            
        Returns:
            list: List of detected onomatopoeia events with accurate timing
        """
        if self.yamnet_model is None or len(audio_chunk) == 0:
            return []
        
        detected_events = []
        
        try:
            # Ensure audio is normalized
            audio_chunk = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Run YAMNet inference
            scores, embeddings, spectrogram = self.yamnet_model(audio_chunk)
            mean_scores = tf.reduce_mean(scores, axis=0)
            
            # AI decides: Look at top 5 predictions with higher selectivity
            top_indices = tf.nn.top_k(mean_scores, k=5).indices
                        
            for i, idx in enumerate(top_indices):
                confidence = float(mean_scores[idx])
                class_name = self.class_names[idx]
                
                # Much higher minimum confidence threshold
                if confidence < 0.20:  # Was 0.01, now 0.20
                    continue
                
                # Check if this class maps to our onomatopoeia
                sound_class = None
                for yamnet_class, mapped_class in YAMNET_CLASS_MAPPINGS.items():
                    if yamnet_class.lower() in class_name.lower():
                        sound_class = mapped_class
                        break
                
                # Check for partial matches
                if not sound_class:
                    explosion_terms = ['explosion', 'blast', 'boom', 'bang', 'gunshot', 'crash']
                    for term in explosion_terms:
                        if term in class_name.lower():
                            sound_class = 'explosion'
                            break
                    
                    if not sound_class:
                        impact_terms = ['thud', 'slam', 'break', 'crash', 'smash']
                        for term in impact_terms:
                            if term in class_name.lower():
                                sound_class = 'crash'
                                break
                
                if sound_class:                   
                    base_threshold = AI_THRESHOLDS.get(sound_class, 0.40)  # Default was 0.15, now 0.40
                    
                    # Steeper ranking penalty - position matters much more
                    ai_decision_score = confidence * (1.0 - (i * 0.20))  # Was 0.05, now 0.20
                    
                    # Apply sensitivity with more aggressive scaling
                    ai_threshold = base_threshold * (0.5 + (self.ai_sensitivity * 0.5))
                    
                    if ai_decision_score >= ai_threshold:
                        # Find the actual peak timing within this chunk
                        peak_time = self.find_sound_peak_in_chunk(audio_chunk, chunk_start_time, sample_rate)
                        
                        # AI decides to create onomatopoeia
                        onomatopoeia = self.get_onomatopoeia_word(sound_class)
                        
                        if onomatopoeia:
                            # AI determines duration naturally
                            ai_duration = self.ai_determine_duration(sound_class, confidence)
                            
                            event = {
                                'word': onomatopoeia,
                                'start_time': peak_time,  # Use detected peak time instead of chunk start
                                'end_time': peak_time + ai_duration,
                                'confidence': confidence,
                                'energy': confidence,
                                'detected_class': class_name,
                                'ai_duration': ai_duration,
                                'ai_decision_score': ai_decision_score,
                                'chunk_start': chunk_start_time  # Keep for debugging
                            }
                            detected_events.append(event)
                            
                            self.log_func(f"AI CREATED ONOMATOPOEIA: {onomatopoeia} "
                                         f"(peak at {peak_time:.1f}s, duration: {ai_duration:.1f}s, "
                                         f"confidence: {confidence:.3f}, chunk: {chunk_start_time:.1f}s)")
                            break  # Only one onomatopoeia per chunk
            
        except Exception as e:
            self.log_func(f"Error in AI sound detection: {e}")
        
        return detected_events
    
    def analyze_audio_file(self, audio_path, chunk_duration=3.0, step_size=1.0):
        """
        Main method - now uses overlapping chunks by default.
        """
        return self.analyze_audio_file_with_overlapping_chunks(audio_path, chunk_duration, step_size)
    
    def analyze_audio_file_with_overlapping_chunks(self, audio_path, chunk_duration=3.0, step_size=1.0):
        """
        Analyze entire audio file with overlapping chunks for better sound detection.
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration (float): Duration of each analysis chunk (3 seconds)
            step_size (float): Time between chunk starts (1 second)
            
        Returns:
            list: All detected onomatopoeia events with accurate timing
        """
        if self.yamnet_model is None:
            self.log_func("AI Onomatopoeia system not available")
            return []
        
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"AI analyzing audio with overlapping chunks: {audio_path}")
            self.log_func(f"Chunk duration: {chunk_duration}s, Step size: {step_size}s")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if len(audio) == 0:
                self.log_func("Audio file is empty")
                return []
            
            audio_duration = len(audio) / sr
            self.log_func(f"Audio loaded: {audio_duration:.2f} seconds")
            
            # Calculate overlapping chunks
            chunk_samples = int(chunk_duration * sr)
            step_samples = int(step_size * sr)
            
            all_events = []
            chunk_count = 0
            
            # Process overlapping chunks
            for start_sample in range(0, len(audio) - chunk_samples + 1, step_samples):
                chunk = audio[start_sample:start_sample + chunk_samples]
                start_time = start_sample / sr
                
                # Skip if we've gone past the audio
                if start_time >= audio_duration:
                    break
                
                chunk_count += 1
                
                # AI analyzes this chunk with improved timing
                events = self.detect_sounds_in_chunk_with_timing(chunk, start_time, sr)
                all_events.extend(events)
            
            self.log_func(f"AI onomatopoeia analysis complete: {len(all_events)} events from {chunk_count} overlapping chunks")
            
            # Show AI's timing improvements
            if all_events:
                total_duration = sum(event['ai_duration'] for event in all_events)
                avg_duration = total_duration / len(all_events)
                self.log_func(f"AI duration statistics: avg={avg_duration:.1f}s, total={total_duration:.1f}s")
                
                # Show timing accuracy info
                timing_offsets = []
                for event in all_events:
                    if 'chunk_start' in event:
                        offset = event['start_time'] - event['chunk_start']
                        timing_offsets.append(offset)
                
                if timing_offsets:
                    avg_offset = sum(timing_offsets) / len(timing_offsets)
                    self.log_func(f"Timing accuracy: avg peak offset from chunk start = {avg_offset:.2f}s")
            
            return all_events
            
        except Exception as e:
            self.log_func(f"Error in AI audio analysis: {e}")
            return []

    def generate_srt_content(self, events):
        """
        Generate SRT subtitle content from onomatopoeia events.
        Uses the exact same format as the desktop subtitles.
        
        Args:
            events (list): List of onomatopoeia events
            
        Returns:
            str: SRT formatted content
        """
        if not events:
            return ""
        
        srt_content = []
        
        for i, event in enumerate(events, 1):
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']
            
            # Format timestamps for SRT (same format as subtitle_converter.py)
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            # Use exact same format as other SRT files
            srt_content.append(f"{i}")
            srt_content.append(f"{start_formatted} --> {end_formatted}")
            srt_content.append(f"{word}")
            srt_content.append("")  # Empty line between entries
        
        return "\n".join(srt_content)
    
    def _format_srt_time(self, seconds):
        """Format time in seconds to SRT format: HH:MM:SS,mmm (same as subtitle_converter.py)"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# Integration function to replace the original detector
def create_ai_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, use_animation=True, animation_setting="Random"):
    """
    Create onomatopoeia subtitle file using AI-determined durations with overlapping chunks.
    
    Args:
        audio_path (str): Path to audio file
        output_srt_path (str): Output subtitle file path
        log_func: Logging function
        use_animation (bool): Whether to use animations
        animation_setting (str): Animation type
        
    Returns:
        tuple: (success: bool, events: list)
    """
    try:
        if log_func:
            log_func("Using AI-determined onomatopoeia with overlapping chunks for better detection")
        
        # Use the AI detector with overlapping chunks
        ai_detector = AIOnomatopoeiaDetector(log_func=log_func)
        events = ai_detector.analyze_audio_file(audio_path)  # Now uses 3s chunks with 1s steps by default
        
        if not events:
            if log_func:
                log_func("AI found no suitable onomatopoeia events")
            return False, []
        
        if use_animation:
            try:
                from animations import create_animated_onomatopoeia_ass
                # Change to .ass for animated version
                ass_path = os.path.splitext(output_srt_path)[0] + '.ass'
                
                # Create ASS file with AI-determined durations and timing
                success = create_animated_ass_from_events(events, ass_path, animation_setting, log_func)
                return success, events
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, using static SRT")
                use_animation = False
        
        # Create static SRT with AI durations and improved timing
        if not use_animation:
            srt_content = ai_detector.generate_srt_content(events)
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            if log_func:
                log_func(f"AI onomatopoeia SRT created: {len(events)} events with improved timing")
            return True, events
            
    except Exception as e:
        if log_func:
            log_func(f"Error creating AI onomatopoeia: {e}")
        return False, []


def create_animated_ass_from_events(events, output_path, animation_setting, log_func):
    """Create animated ASS file from AI-determined events"""
    try:
        from animations.core import OnomatopoeiaAnimator
        
        animator = OnomatopoeiaAnimator()
        animated_content = animator.generate_animated_ass_content(events, animation_setting)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(animated_content)
        
        if log_func:
            log_func(f"AI animated ASS created: {output_path} with {len(events)} precisely-timed events")
        
        return True
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating animated ASS: {e}")
        return False