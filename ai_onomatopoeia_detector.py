"""
Enhanced Onomatopoeia detection with AI-determined duration.
The AI determines duration based on sound characteristics without external filters.
"""

import os
import random
import numpy as np # type: ignore
import librosa # type: ignore
import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore
from onomatopoeia_detector import SOUND_MAPPINGS, YAMNET_CLASS_MAPPINGS

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
        
        # Sound duration characteristics - AI determines these based on sound type
        self.sound_duration_profiles = {
            # Impact sounds - brief and sharp
            'explosion': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'sharp'},
            'crash': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'sharp'},
            'glass': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'sharp'},
            'gunshot': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
            'slam': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
            'thud': {'base_duration': 0.6, 'variability': 0.3, 'decay_type': 'medium'},
            
            # Enhanced thud variants
            'heavy_thud': {'base_duration': 0.8, 'variability': 0.2, 'decay_type': 'medium'},
            'soft_thud': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'soft'},
            
            # Crunch sounds - variable duration
            'crunch': {'base_duration': 0.7, 'variability': 0.4, 'decay_type': 'medium'},
            'crackle': {'base_duration': 1.2, 'variability': 0.5, 'decay_type': 'gradual'},
            'break': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
            
            # Combat/Fighting sounds - quick and punchy
            'punch': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
            'hit': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'sharp'},
            'kick': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'sharp'},
            'sword': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
            
            # Gun/Weapon loading sounds - mechanical precision
            'gun_load': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
            'reload': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'instant'},
            'magazine': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
            'bolt': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
            
            # Electronic/Mechanical - varies by type
            'bell': {'base_duration': 1.5, 'variability': 0.5, 'decay_type': 'gradual'},
            'buzz': {'base_duration': 1.0, 'variability': 0.3, 'decay_type': 'sustained'},
            'beep': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
            'click': {'base_duration': 0.1, 'variability': 0.05, 'decay_type': 'instant'},
            'alarm': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'sustained'},
            
            # Movement/Air sounds - flowing
            'whoosh': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'gradual'},
            'pop': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
            'whistle': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
            'siren': {'base_duration': 3.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'wind': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            
            # Water/Liquid sounds - varies by intensity
            'splash': {'base_duration': 0.8, 'variability': 0.4, 'decay_type': 'medium'},
            'drip': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'soft'},
            'pour': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'bubble': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'gradual'},
            
            # Nature sounds - environmental
            'thunder': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'gradual'},
            'rain': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
            'fire': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'sustained'},
            
            # Animal sounds - characteristic durations
            'dog': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'sharp'},
            'cat': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'medium'},
            'bird': {'base_duration': 0.5, 'variability': 0.3, 'decay_type': 'medium'},
            'horse': {'base_duration': 1.2, 'variability': 0.4, 'decay_type': 'medium'},
            'cow': {'base_duration': 1.5, 'variability': 0.5, 'decay_type': 'gradual'},
            'pig': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'medium'},
            'sheep': {'base_duration': 1.0, 'variability': 0.3, 'decay_type': 'medium'},
            'lion': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'gradual'},
            'bear': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
            'wolf': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'gradual'},
            'snake': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'sustained'},
            'insect': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'frog': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
            
            # Human sounds
            'applause': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
            'footsteps': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'sharp'},
            'knock': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
            'sneeze': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
            'cough': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'medium'},
            'laugh': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'gasp': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
            'whisper': {'base_duration': 1.5, 'variability': 0.8, 'decay_type': 'soft'},
            
            # Vehicle sounds
            'car_horn': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'sustained'},
            'engine': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'brakes': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
            'tire': {'base_duration': 1.2, 'variability': 0.5, 'decay_type': 'gradual'},
            'motorcycle': {'base_duration': 1.8, 'variability': 0.8, 'decay_type': 'sustained'},
            'truck': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'sustained'},
            
            # Food/Eating sounds
            'chew': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'sustained'},
            'bite': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
            'slurp': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'gradual'},
            'sizzle': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            'boil': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
            
            # Technology sounds
            'computer': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
            'phone': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'sustained'},
            'camera': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
            'printer': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
            
            # Miscellaneous
            'zipper': {'base_duration': 0.6, 'variability': 0.3, 'decay_type': 'gradual'},
            'paper': {'base_duration': 0.5, 'variability': 0.3, 'decay_type': 'soft'},
            'fabric': {'base_duration': 0.8, 'variability': 0.4, 'decay_type': 'soft'},
            'door': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'gradual'},
            'spring': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
            'rubber': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'medium'}
        }
        
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
    
    def detect_sounds_in_chunk(self, audio_chunk, start_time, sample_rate=16000):
        """
        Detect sounds with AI-determined duration (no filters or confidence thresholds).
        
        Args:
            audio_chunk (np.array): Audio data
            start_time (float): Start time of this chunk
            sample_rate (int): Audio sample rate
            
        Returns:
            list: List of detected onomatopoeia events with AI-determined duration
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
            
            # AI decides: Look at ALL predictions, not just high-confidence ones
            # Get top 20 predictions to give AI more options
            top_indices = tf.nn.top_k(mean_scores, k=20).indices
            
            self.log_func(f"AI analyzing {len(top_indices)} sound possibilities at {start_time:.1f}s:")
            
            for i, idx in enumerate(top_indices):
                confidence = float(mean_scores[idx])
                class_name = self.class_names[idx]
                
                # AI decision: Consider ANY detected sound, regardless of confidence
                # The AI will determine if it's worth creating an onomatopoeia
                if confidence > 0.01:  # Only filter out completely irrelevant sounds
                    
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
                        # AI determines if this sound should generate an onomatopoeia
                        # Factors: sound type, confidence, position in ranking
                        
                        ai_decision_score = confidence * (1.0 - (i * 0.05))  # Lower rank = lower score
                        
                        # AI threshold: Dynamic based on sound type and sensitivity
                        ai_thresholds = {
                            'explosion': 0.05,  # AI loves explosions
                            'crash': 0.08,
                            'glass': 0.10,
                            'gunshot': 0.05,
                            'slam': 0.12,
                            'thud': 0.15,
                            'punch': 0.08,
                            'bell': 0.20,
                            'alarm': 0.25,
                        }
                        
                        base_threshold = ai_thresholds.get(sound_class, 0.15)
                        
                        # Apply sensitivity: lower sensitivity = lower threshold (more permissive)
                        # Higher sensitivity = higher threshold (more selective)
                        ai_threshold = base_threshold * self.ai_sensitivity
                        
                        if ai_decision_score >= ai_threshold:
                            # AI decides to create onomatopoeia
                            onomatopoeia = self.get_onomatopoeia_word(sound_class)
                            
                            if onomatopoeia:
                                # AI determines duration naturally
                                ai_duration = self.ai_determine_duration(sound_class, confidence)
                                
                                event = {
                                    'word': onomatopoeia,
                                    'start_time': start_time,
                                    'end_time': start_time + ai_duration,  # AI-determined duration
                                    'confidence': confidence,
                                    'energy': confidence,  # Use confidence as energy for sizing
                                    'detected_class': class_name,
                                    'ai_duration': ai_duration,
                                    'ai_decision_score': ai_decision_score
                                }
                                detected_events.append(event)
                                
                                self.log_func(f"AI CREATED ONOMATOPOEIA: {onomatopoeia} "
                                             f"(duration: {ai_duration:.1f}s, confidence: {confidence:.3f}, "
                                             f"decision_score: {ai_decision_score:.3f})")
                                break  # Only one onomatopoeia per chunk
            
        except Exception as e:
            self.log_func(f"Error in AI sound detection: {e}")
        
        return detected_events
    
    def analyze_audio_file(self, audio_path, chunk_duration=1.0):
        """
        Analyze entire audio file with AI-determined durations.
        
        Args:
            audio_path (str): Path to audio file
            chunk_duration (float): Analysis chunk duration
            
        Returns:
            list: All detected onomatopoeia events with AI durations
        """
        if self.yamnet_model is None:
            self.log_func("AI Onomatopoeia system not available")
            return []
        
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"AI analyzing audio for natural onomatopoeia durations: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if len(audio) == 0:
                self.log_func("Audio file is empty")
                return []
            
            self.log_func(f"Audio loaded: {len(audio)/sr:.2f} seconds - AI will determine all durations naturally")
            
            # Process in chunks
            chunk_samples = int(chunk_duration * sr)
            all_events = []
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                start_time = i / sr
                
                # AI analyzes this chunk
                events = self.detect_sounds_in_chunk(chunk, start_time, sr)
                all_events.extend(events)
            
            self.log_func(f"AI onomatopoeia analysis complete: {len(all_events)} events with natural durations")
            
            # Show AI's duration decisions
            if all_events:
                total_duration = sum(event['ai_duration'] for event in all_events)
                avg_duration = total_duration / len(all_events)
                self.log_func(f"AI duration statistics: avg={avg_duration:.1f}s, total={total_duration:.1f}s")
            
            return all_events
            
        except Exception as e:
            self.log_func(f"Error in AI audio analysis: {e}")
            return []


# Integration function to replace the original detector
def create_ai_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, use_animation=True, animation_setting="Random"):
    """
    Create onomatopoeia subtitle file using AI-determined durations.
    
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
            log_func("Using AI-determined onomatopoeia durations (no filters or thresholds)")
        
        # Use the AI detector instead of the original
        ai_detector = AIOnomatopoeiaDetector(log_func=log_func)
        events = ai_detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("AI found no suitable onomatopoeia events")
            return False, []
        
        if use_animation:
            try:
                from animations import create_animated_onomatopoeia_ass
                # Change to .ass for animated version
                ass_path = os.path.splitext(output_srt_path)[0] + '.ass'
                
                # Create ASS file with AI-determined durations
                success = create_animated_ass_from_events(events, ass_path, animation_setting, log_func)
                return success, events
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, using static SRT")
                use_animation = False
        
        # Create static SRT with AI durations
        if not use_animation:
            srt_content = ai_detector.generate_srt_content(events)
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            if log_func:
                log_func(f"AI onomatopoeia SRT created: {len(events)} events with natural durations")
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
            log_func(f"AI animated ASS created: {output_path} with {len(events)} naturally-timed events")
        
        return True
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating animated ASS: {e}")
        return False