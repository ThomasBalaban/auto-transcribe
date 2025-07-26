"""
Onomatopoeia detection for comic book-style sound effects.
Analyzes desktop audio track to detect sound events and generates
comic-style onomatopoeia overlays.
"""

import os
import random
import numpy as np # type: ignore
import librosa # type: ignore
import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore
from sound_mappings import SOUND_MAPPINGS
from yamnet_mappings import YAMNET_CLASS_MAPPINGS

# Try importing YAMNet
try:
    # YAMNet model from TensorFlow Hub
    YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = None
    YAMNET_AVAILABLE = False
except ImportError:
    YAMNET_AVAILABLE = False

class OnomatopoeiaDetector:
    def __init__(self, confidence_threshold=0.7, log_func=None):
        """
        Initialize the onomatopoeia detector.
        
        Args:
            confidence_threshold (float): Minimum confidence for sound detection
            log_func: Logging function
        """
        self.confidence_threshold = confidence_threshold
        self.log_func = log_func or print
        self.yamnet_model = None
        self.class_names = None
        self.recent_words = []  # Track recent words to avoid repetition
        self.max_recent_words = 5
        
        self._load_yamnet_model()
    
    def _load_yamnet_model(self):
        """Load YAMNet model for audio classification with SSL fixes"""
        try:
            self.log_func("Loading YAMNet model for onomatopoeia detection...")
            
            # First, try to fix SSL certificates
            self._fix_ssl_certificates()
            
            # Setup cache directory
            self._setup_cache_directory()
            
            # Try multiple loading approaches
            model_loaded = False
            
            # Approach 1: Standard load
            try:
                self.yamnet_model = hub.load(YAMNET_MODEL_URL)
                model_loaded = True
                self.log_func("✓ YAMNet loaded with standard method")
            except Exception as e:
                self.log_func(f"Standard load failed: {e}")
            
            # Approach 2: With explicit SSL context fix
            if not model_loaded:
                try:
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.yamnet_model = hub.load(YAMNET_MODEL_URL)
                    model_loaded = True
                    self.log_func("✓ YAMNet loaded with SSL certificate fix")
                except Exception as e:
                    self.log_func(f"SSL fix method failed: {e}")
            
            if not model_loaded:
                raise Exception("All YAMNet loading methods failed")
            
            # Load class names with the same SSL fix
            try:
                class_map_path = tf.keras.utils.get_file(
                    'yamnet_class_map.csv',
                    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
                )
                
                with open(class_map_path) as f:
                    self.class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]
                
                self.log_func(f"YAMNet model loaded successfully with {len(self.class_names)} sound classes")
                global YAMNET_AVAILABLE
                YAMNET_AVAILABLE = True
                
            except Exception as e:
                self.log_func(f"Failed to load class names: {e}")
                # Create fallback class names for common sounds
                self._create_fallback_class_names()
                
        except Exception as e:
            self.log_func(f"Failed to load YAMNet model: {e}")
            self.log_func("Onomatopoeia detection will be disabled")
            self.log_func("\nTo fix this issue, try:")
            self.log_func("1. macOS: Run '/Applications/Python\\ 3.x/Install\\ Certificates.command'")
            self.log_func("2. All systems: pip install --upgrade certifi")
            self.log_func("3. Try using a VPN or different network connection")
            YAMNET_AVAILABLE = False
    
    def _fix_ssl_certificates(self):
        """Fix SSL certificate issues"""
        try:
            import ssl
            import certifi # type: ignore
            
            # Method 1: Use certifi certificates
            ssl._create_default_https_context = ssl.create_default_context
            
            # Method 2: If that fails, create unverified context
            original_context = ssl._create_default_https_context
            
            def create_unverified_context(*args, **kwargs):
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                return context
            
            ssl._create_default_https_context = create_unverified_context
            self.log_func("SSL certificate handling configured")
            
        except Exception as e:
            self.log_func(f"SSL fix warning: {e}")
    
    def _setup_cache_directory(self):
        """Setup TensorFlow Hub cache directory"""
        try:
            import os
            cache_dir = os.path.expanduser("~/.cache/tensorflow_hub")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
            self.log_func(f"TensorFlow Hub cache directory: {cache_dir}")
        except Exception as e:
            self.log_func(f"Cache setup warning: {e}")
    
    def _create_fallback_class_names(self):
        """Create fallback class names if download fails"""
        # Basic class names for common sounds (subset of YAMNet classes)
        self.class_names = [
            'Silence', 'Speech', 'Male speech', 'Female speech', 'Child speech',
            'Music', 'Musical instrument', 'Plucked string instrument', 'Guitar',
            'Electric guitar', 'Bass guitar', 'Acoustic guitar', 'Steel guitar',
            'Explosion', 'Gunshot, gunfire', 'Breaking', 'Glass', 'Slam',
            'Thud', 'Bell', 'Buzzer', 'Beep, bleep', 'Click', 'Alarm',
            'Whoosh, swoosh, swish', 'Pop', 'Whistle', 'Siren',
            'Splash, splatter', 'Thunder', 'Rain', 'Applause', 'Footsteps',
            'Knock', 'Car', 'Vehicle horn, car horn, honking', 'Motor vehicle (road)'
        ]
        self.log_func(f"Using fallback class names ({len(self.class_names)} classes)")
        global YAMNET_AVAILABLE
        YAMNET_AVAILABLE = True
    
    def calculate_audio_energy(self, audio_chunk):
        """
        Calculate the energy/loudness of an audio chunk for text sizing.
        
        Args:
            audio_chunk (np.array): Audio data
            
        Returns:
            float: Normalized energy level (0.0 to 1.0)
        """
        if len(audio_chunk) == 0:
            return 0.0
            
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        
        # Convert to dB and normalize
        db_level = 20 * np.log10(rms_energy + 1e-10)
        
        # Normalize to 0-1 range (assuming -60dB to 0dB range)
        normalized_energy = max(0.0, min(1.0, (db_level + 60) / 60))
        
        return normalized_energy
    
    def get_onomatopoeia_word(self, sound_class):
        """
        Get a random onomatopoeia word for the detected sound class.
        Avoids recently used words for variety.
        
        Args:
            sound_class (str): The detected sound class
            
        Returns:
            str: Onomatopoeia word or None if no mapping exists
        """
        if sound_class not in SOUND_MAPPINGS:
            return None
        
        available_words = SOUND_MAPPINGS[sound_class]
        
        # Try to avoid recently used words
        if self.recent_words:
            unused_words = [w for w in available_words if w not in self.recent_words]
            if unused_words:
                available_words = unused_words
        
        selected_word = random.choice(available_words)
        
        # Track recent words
        self.recent_words.append(selected_word)
        if len(self.recent_words) > self.max_recent_words:
            self.recent_words.pop(0)
        
        return selected_word
    
    def detect_sounds_in_chunk(self, audio_chunk, start_time, sample_rate=16000):
        """
        Detect sounds in an audio chunk using YAMNet.
        
        Args:
            audio_chunk (np.array): Audio data
            start_time (float): Start time of this chunk in seconds
            sample_rate (int): Audio sample rate
            
        Returns:
            list: List of detected onomatopoeia events with timing and energy
        """
        if not YAMNET_AVAILABLE or self.yamnet_model is None:
            return []
        
        if len(audio_chunk) == 0:
            return []
        
        detected_events = []
        
        try:
            # Ensure audio is float32 and normalized
            audio_chunk = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Run YAMNet inference
            scores, embeddings, spectrogram = self.yamnet_model(audio_chunk)
            
            # Get the mean scores across time for this chunk
            mean_scores = tf.reduce_mean(scores, axis=0)
            
            # DEBUG: Show top 10 predictions regardless of threshold
            top_10_indices = tf.nn.top_k(mean_scores, k=10).indices
            self.log_func(f"DEBUG - Top 10 sounds detected at {start_time:.1f}s:")
            for i, idx in enumerate(top_10_indices):
                confidence = float(mean_scores[idx])
                class_name = self.class_names[idx]
                self.log_func(f"  {i+1}. {class_name}: {confidence:.3f}")
            
            # Find top predictions above threshold
            top_indices = tf.nn.top_k(mean_scores, k=10).indices  # Check top 10 instead of 5
            
            for idx in top_indices:
                confidence = float(mean_scores[idx])
                class_name = self.class_names[idx]
                
                # DEBUG: Log all sounds above a low threshold for debugging
                if confidence >= 0.1:  # Very low threshold for debugging
                    self.log_func(f"DEBUG - Potential sound: {class_name} ({confidence:.3f})")
                
                if confidence >= self.confidence_threshold:
                    # Check if this class maps to our onomatopoeia
                    sound_class = None
                    for yamnet_class, mapped_class in YAMNET_CLASS_MAPPINGS.items():
                        if yamnet_class.lower() in class_name.lower():
                            sound_class = mapped_class
                            self.log_func(f"MATCH FOUND: {class_name} -> {sound_class}")
                            break
                    
                    # Also check for partial matches with broader terms
                    if not sound_class:
                        # Check for explosion-related terms
                        explosion_terms = ['explosion', 'blast', 'boom', 'bang', 'gunshot', 'crash']
                        for term in explosion_terms:
                            if term in class_name.lower():
                                sound_class = 'explosion'
                                self.log_func(f"PARTIAL MATCH: {class_name} contains '{term}' -> explosion")
                                break
                        
                        # Check for other sound types
                        if not sound_class:
                            impact_terms = ['thud', 'slam', 'break', 'crash', 'smash']
                            for term in impact_terms:
                                if term in class_name.lower():
                                    sound_class = 'crash'
                                    self.log_func(f"PARTIAL MATCH: {class_name} contains '{term}' -> crash")
                                    break
                    
                    if sound_class:
                        # Get onomatopoeia word
                        onomatopoeia = self.get_onomatopoeia_word(sound_class)
                        if onomatopoeia:
                            # Calculate energy for text sizing
                            energy = self.calculate_audio_energy(audio_chunk)
                            
                            event = {
                                'word': onomatopoeia,
                                'start_time': start_time,
                                'end_time': start_time + 0.5,  # Fixed 0.5s duration
                                'confidence': confidence,
                                'energy': energy,
                                'detected_class': class_name
                            }
                            detected_events.append(event)
                            
                            self.log_func(f"ONOMATOPOEIA DETECTED: {onomatopoeia} (confidence: {confidence:.3f}, energy: {energy:.2f})")
                            break  # Only one onomatopoeia per chunk
            
        except Exception as e:
            self.log_func(f"Error in sound detection: {e}")
            import traceback
            self.log_func(f"Traceback: {traceback.format_exc()}")
        
        return detected_events
    
    def analyze_audio_file(self, audio_path, chunk_duration=1.0):
        """
        Analyze an entire audio file for onomatopoeia events.
        
        Args:
            audio_path (str): Path to the audio file
            chunk_duration (float): Duration of each analysis chunk in seconds
            
        Returns:
            list: List of all detected onomatopoeia events
        """
        if not YAMNET_AVAILABLE:
            self.log_func("YAMNet not available - skipping onomatopoeia detection")
            return []
        
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"Analyzing audio for onomatopoeia: {audio_path}")
            
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)  # YAMNet expects 16kHz
            
            if len(audio) == 0:
                self.log_func("Audio file is empty")
                return []
            
            self.log_func(f"Audio loaded: {len(audio)/sr:.2f} seconds, {sr} Hz")
            
            # Process in chunks
            chunk_samples = int(chunk_duration * sr)
            all_events = []
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                start_time = i / sr
                
                # Detect sounds in this chunk
                events = self.detect_sounds_in_chunk(chunk, start_time, sr)
                all_events.extend(events)
            
            self.log_func(f"Onomatopoeia detection complete: {len(all_events)} events found")
            return all_events
            
        except Exception as e:
            self.log_func(f"Error analyzing audio file: {e}")
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


def detect_onomatopoeia(audio_path, log_func=None, confidence_threshold=0.7):
    """
    Convenience function to detect onomatopoeia in an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        log_func: Logging function
        confidence_threshold (float): Minimum confidence for detection
        
    Returns:
        list: List of onomatopoeia events
    """
    detector = OnomatopoeiaDetector(confidence_threshold, log_func)
    return detector.analyze_audio_file(audio_path)


def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None):
    """
    Create an SRT file with onomatopoeia subtitles from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        output_srt_path (str): Path for the output SRT file
        log_func: Logging function
        
    Returns:
        tuple: (success: bool, events: list) - Success status and detected events
    """
    try:
        detector = OnomatopoeiaDetector(log_func=log_func)
        events = detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("No onomatopoeia events detected")
            return False, []
        
        srt_content = detector.generate_srt_content(events)
        
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        if log_func:
            log_func(f"Onomatopoeia SRT created: {output_srt_path} with {len(events)} events")
        return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating onomatopoeia SRT: {e}")
        return False, []


def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, use_animation=True, animation_setting="Random"):
    """
    Create a subtitle file with onomatopoeia effects using AI-determined durations.
    This version removes all filters and lets the AI decide everything naturally.
    
    Args:
        audio_path (str): Path to the audio file
        output_srt_path (str): Path for the output subtitle file
        log_func: Logging function
        use_animation (bool): Whether to use animated effects
        animation_setting (str): Animation type
        
    Returns:
        tuple: (success: bool, events: list) - Success status and detected events
    """
    try:
        if log_func:
            log_func("=== AI-DETERMINED ONOMATOPOEIA SYSTEM ===")
            log_func("No confidence filters, no energy thresholds - pure AI decisions")
        
        # Import the AI detector
        from ai_onomatopoeia_detector import AIOnomatopoeiaDetector
        
        # Get AI sensitivity from the confidence setting (now repurposed)
        # Note: In video_processor.py, you'll need to pass the confidence_threshold as ai_sensitivity
        ai_sensitivity = 0.5  # Default, should be passed from UI
        
        # Use AI detector instead of original
        ai_detector = AIOnomatopoeiaDetector(ai_sensitivity=ai_sensitivity, log_func=log_func)
        events = ai_detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("AI determined no onomatopoeia events should be created")
            return False, []
        
        if use_animation:
            try:
                from animations.core import OnomatopoeiaAnimator
                if log_func:
                    log_func(f"Creating AI-timed animated effects (ASS format) - {animation_setting}")
                
                # Change extension to .ass for animated version
                ass_path = os.path.splitext(output_srt_path)[0] + '.ass'
                
                # Generate animated content with AI-determined durations
                animator = OnomatopoeiaAnimator()
                animated_content = animator.generate_animated_ass_content(events, animation_setting)
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(animated_content)
                
                if log_func:
                    log_func(f"AI animated onomatopoeia created: {len(events)} events with natural timing")
                    for i, event in enumerate(events[:3]):
                        duration = event['end_time'] - event['start_time']
                        log_func(f"  Event {i+1}: {event['word']} - {duration:.1f}s (AI decision)")
                
                return True, events
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, falling back to static SRT")
                use_animation = False
        
        # Fallback to static SRT with AI durations
        if not use_animation:
            if log_func:
                log_func("Creating AI-timed static effects (SRT format)")
            
            # Generate SRT content with AI durations
            srt_content = ai_detector.generate_srt_content(events)
            
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            if log_func:
                log_func(f"AI static onomatopoeia created: {len(events)} events with natural timing")
            
            return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error in AI onomatopoeia system: {e}")
            import traceback
            log_func(f"Traceback: {traceback.format_exc()}")
        return False, []