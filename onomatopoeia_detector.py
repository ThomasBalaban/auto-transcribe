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

# Try importing YAMNet
try:
    # YAMNet model from TensorFlow Hub
    YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = None
    YAMNET_AVAILABLE = False
except ImportError:
    YAMNET_AVAILABLE = False

# Sound class to onomatopoeia mapping (no punctuation)
SOUND_MAPPINGS = {
    # Impact sounds
    'explosion': ['BOOM', 'KABOOM', 'BLAST', 'BLAM', 'WHAM'],
    'crash': ['CRASH', 'SMASH', 'BANG', 'CLANG', 'CLANK'],
    'glass': ['SHATTER', 'CRASH', 'SMASH', 'TINKLE', 'CLINK'],
    'gunshot': ['BANG', 'POW', 'BLAST', 'CRACK', 'POP'],
    'slam': ['SLAM', 'BANG', 'THUD', 'WHAM', 'CLANG'],
    'thud': ['THUD', 'THUMP', 'BUMP', 'PLOP', 'BONK'],
    
    # Enhanced thud variants
    'heavy_thud': ['THUNK', 'CLUNK', 'PLUNK', 'BONK', 'WHUMP'],
    'soft_thud': ['PLOP', 'PLUNK', 'FLOP', 'SQUISH', 'SPLAT'],
    
    # Crunch sounds
    'crunch': ['CRUNCH', 'CRACKLE', 'SNAP', 'CRACK', 'CHOMP'],
    'crackle': ['CRACKLE', 'SNAP', 'POP', 'FIZZ', 'SIZZLE'],
    'break': ['CRACK', 'SNAP', 'BREAK', 'CRUNCH', 'SPLIT'],
    
    # Combat/Fighting sounds
    'punch': ['POW', 'PUNCH', 'SOCK', 'BAM', 'WHACK'],
    'hit': ['SMACK', 'WHAP', 'SLAP', 'BONK', 'CLONK'],
    'kick': ['THWACK', 'KICK', 'BOOT', 'PUNT', 'WHAM'],
    'sword': ['CLANG', 'SLASH', 'SWISH', 'CLANK', 'ZING'],
    
    # Gun/Weapon loading sounds
    'gun_load': ['CLICK', 'CLACK', 'SNAP', 'CLANK', 'COCK'],
    'reload': ['CLACK', 'SNAP', 'CLICK', 'CHUNK', 'SLIDE'],
    'magazine': ['CLICK', 'SNAP', 'CLUNK', 'SLIDE', 'LOCK'],
    'bolt': ['CLACK', 'SNAP', 'SLIDE', 'CHUNK', 'RACK'],
    
    # Electronic/Mechanical
    'bell': ['DING', 'RING', 'CHIME', 'CLANG', 'BONG'],
    'buzz': ['BUZZ', 'ZZZ', 'BZZT', 'HUM', 'DRONE'],
    'beep': ['BEEP', 'BOOP', 'PING', 'BLIP', 'CHIRP'],
    'click': ['CLICK', 'TAP', 'SNAP', 'TICK', 'CLACK'],
    'alarm': ['BLARE', 'WAIL', 'RING', 'CLANG', 'DING'],
    
    # Movement/Air sounds
    'whoosh': ['WHOOSH', 'SWOOSH', 'ZOOM', 'WHIP', 'SWISH'],
    'pop': ['POP', 'SNAP', 'CRACK', 'BURST', 'PING'],
    'whistle': ['TWEET', 'WHEE', 'PEEP', 'TRILL', 'CHIRP'],
    'siren': ['WAIL', 'BLARE', 'WHOOP', 'HOWL', 'SCREECH'],
    'wind': ['WHOOSH', 'HOWL', 'WHISTLE', 'MOAN', 'RUSH'],
    
    # Water/Liquid sounds
    'splash': ['SPLASH', 'PLOP', 'DRIP', 'SPLAT', 'GUSH'],
    'drip': ['DRIP', 'DROP', 'PLIP', 'PLINK', 'PLOP'],
    'pour': ['GLUG', 'GURGLE', 'SPLASH', 'FLOW', 'RUSH'],
    'bubble': ['BLUB', 'GURGLE', 'POP', 'FIZZ', 'BUBBLE'],
    
    # Nature sounds
    'thunder': ['RUMBLE', 'CRASH', 'BOOM', 'ROAR', 'GROWL'],
    'rain': ['PATTER', 'DRIP', 'DROP', 'SPLASH', 'PING'],
    'fire': ['CRACKLE', 'POP', 'HISS', 'ROAR', 'WHOOSH'],
    
    # Animal sounds
    'dog': ['WOOF', 'BARK', 'ARF', 'YIP', 'GROWL'],
    'cat': ['MEOW', 'HISS', 'PURR', 'YOWL', 'SCREECH'],
    'bird': ['CHIRP', 'TWEET', 'SQUAWK', 'CAW', 'SCREECH'],
    'horse': ['NEIGH', 'WHINNY', 'SNORT', 'CLOP', 'GALLOP'],
    'cow': ['MOO', 'LOW', 'BELLOW', 'SNORT', 'HUFF'],
    'pig': ['OINK', 'SNORT', 'GRUNT', 'SQUEAL', 'SNUFF'],
    'sheep': ['BAA', 'BLEAT', 'MAA', 'BELLOW', 'GRUNT'],
    'lion': ['ROAR', 'GROWL', 'SNARL', 'RUMBLE', 'GROAN'],
    'bear': ['GROWL', 'ROAR', 'GRUNT', 'SNUFF', 'HUFF'],
    'wolf': ['HOWL', 'GROWL', 'SNARL', 'BARK', 'YIP'],
    'snake': ['HISS', 'RATTLE', 'SLITHER', 'SSSS', 'STRIKE'],
    'insect': ['BUZZ', 'HUM', 'DRONE', 'CHIRP', 'CLICK'],
    'frog': ['CROAK', 'RIBBIT', 'PLOP', 'SPLASH', 'GULP'],
    
    # Human sounds
    'applause': ['CLAP', 'CHEER', 'ROAR', 'WHOOP', 'HOORAY'],
    'footsteps': ['STOMP', 'THUD', 'STEP', 'CLOP', 'PATTER'],
    'knock': ['KNOCK', 'RAP', 'TAP', 'BANG', 'THUD'],
    'sneeze': ['ACHOO', 'SNEEZE', 'CHOO', 'GESUNDHEIT', 'ATISHOO'],
    'cough': ['COUGH', 'HACK', 'AHEM', 'CHOKE', 'WHEEZE'],
    'laugh': ['HAHA', 'GIGGLE', 'CHUCKLE', 'SNORT', 'CHORTLE'],
    'gasp': ['GASP', 'GULP', 'WHEEZE', 'PANT', 'HUFF'],
    'whisper': ['PSST', 'WHISPER', 'SHUSH', 'HUSH', 'MURMUR'],
    
    # Vehicle sounds
    'car_horn': ['HONK', 'BEEP', 'TOOT', 'BLARE', 'BLAST'],
    'engine': ['VROOM', 'PURR', 'ROAR', 'RUMBLE', 'REV'],
    'brakes': ['SCREECH', 'SQUEAL', 'SKID', 'SQUEAK', 'GRIND'],
    'tire': ['SCREECH', 'SQUEAL', 'SKID', 'BURN', 'SPIN'],
    'motorcycle': ['VROOM', 'ROAR', 'REV', 'RUMBLE', 'ZOOM'],
    'truck': ['RUMBLE', 'ROAR', 'DIESEL', 'CHUG', 'GRUNT'],
    
    # Food/Eating sounds
    'chew': ['MUNCH', 'CRUNCH', 'CHOMP', 'CHEW', 'GNAW'],
    'bite': ['CHOMP', 'BITE', 'SNAP', 'CRUNCH', 'MUNCH'],
    'slurp': ['SLURP', 'SIP', 'GULP', 'GLUG', 'SUCK'],
    'sizzle': ['SIZZLE', 'FRY', 'CRACKLE', 'POP', 'HISS'],
    'boil': ['BUBBLE', 'GURGLE', 'BOIL', 'STEAM', 'HISS'],
    
    # Technology sounds
    'computer': ['BEEP', 'BOOP', 'PING', 'CLICK', 'WHIR'],
    'phone': ['RING', 'BUZZ', 'CHIRP', 'DING', 'PING'],
    'camera': ['CLICK', 'SNAP', 'WHIR', 'ZOOM', 'FLASH'],
    'printer': ['WHIR', 'BUZZ', 'CLICK', 'CHUG', 'BEEP'],
    
    # Miscellaneous
    'zipper': ['ZIP', 'UNZIP', 'ZZZIP', 'SLIDE', 'PULL'],
    'paper': ['RUSTLE', 'CRINKLE', 'RIP', 'TEAR', 'CRUMPLE'],
    'fabric': ['RUSTLE', 'SWISH', 'FLUTTER', 'FLAP', 'WHOOSH'],
    'door': ['CREAK', 'SLAM', 'CLICK', 'BANG', 'SQUEAK'],
    'spring': ['BOING', 'BOUNCE', 'SPRING', 'POP', 'TWANG'],
    'rubber': ['SQUEAK', 'BOUNCE', 'POP', 'STRETCH', 'SNAP']
}

# YAMNet class names that map to our onomatopoeia categories
YAMNET_CLASS_MAPPINGS = {
    # Impact/Explosive - Expanded for better detection
    'Explosion': 'explosion',
    'Gunshot, gunfire': 'explosion',  # Often similar to explosions
    'Burst, pop': 'explosion',
    'Breaking': 'crash',
    'Glass': 'glass',
    'Slam': 'slam',
    'Thud': 'thud',
    'Bang': 'explosion',  # Generic bang -> explosion
    'Boom': 'explosion',  # Direct boom sound
    'Artillery fire': 'explosion',
    'Cap gun': 'explosion',
    'Gunshot': 'explosion',
    
    # Thud variants
    'Thunk': 'heavy_thud',
    'Clunk': 'heavy_thud',
    'Plop': 'soft_thud',
    'Splat': 'soft_thud',
    
    # Crunch/Break sounds
    'Crushing': 'crunch',
    'Crumpling, crinkling': 'crunch',
    'Tearing': 'break',
    'Rip': 'break',
    'Crack': 'crackle',
    
    # Combat sounds
    'Slap, smack': 'punch',
    'Whip': 'hit',
    'Whack, thwack': 'hit',
    
    # Gun/Weapon sounds
    'Mechanisms': 'gun_load',
    'Clicking': 'gun_load',
    'Sliding door': 'reload',
    'Latch': 'gun_load',
    
    # Electronic
    'Bell': 'bell',
    'Buzzer': 'buzz',
    'Beep, bleep': 'beep',
    'Click': 'click',
    'Alarm': 'alarm',
    'Ding': 'bell',
    'Telephone bell ringing': 'phone',
    'Ringtone': 'phone',
    'Computer keyboard': 'computer',
    'Typing': 'computer',
    'Printer': 'printer',
    
    # Movement/Air
    'Whoosh, swoosh, swish': 'whoosh',
    'Pop': 'pop',
    'Whistle': 'whistle',
    'Siren': 'siren',
    'Wind noise (microphone)': 'wind',
    'Whoosh': 'whoosh',
    'Swoosh': 'whoosh',
    'Wind': 'wind',
    
    # Water/Nature
    'Splash, splatter': 'splash',
    'Thunder': 'explosion',  # Thunder is like explosion
    'Thunderstorm': 'explosion',
    'Rain': 'rain',
    'Water': 'splash',
    'Drip': 'drip',
    'Pour': 'pour',
    'Gurgling': 'bubble',
    'Bubble': 'bubble',
    'Fire': 'fire',
    'Crackle': 'fire',
    
    # Animal sounds - Dogs
    'Dog': 'dog',
    'Bark': 'dog',
    'Bow-wow': 'dog',
    'Growling': 'dog',
    'Whimper': 'dog',
    
    # Animal sounds - Cats
    'Cat': 'cat',
    'Meow': 'cat',
    'Purr': 'cat',
    'Hiss': 'snake',  # Can be cat or snake
    'Yowl': 'cat',
    
    # Animal sounds - Birds
    'Bird': 'bird',
    'Chirp, tweet': 'bird',
    'Squawk': 'bird',
    'Crow': 'bird',
    'Rooster': 'bird',
    'Owl': 'bird',
    
    # Animal sounds - Farm animals
    'Cattle, bovine': 'cow',
    'Moo': 'cow',
    'Pig': 'pig',
    'Oink': 'pig',
    'Horse': 'horse',
    'Neigh, whinny': 'horse',
    'Sheep': 'sheep',
    'Bleat': 'sheep',
    
    # Animal sounds - Wild animals
    'Lion': 'lion',
    'Roar': 'lion',  # Could be lion, bear, or engine
    'Bear': 'bear',
    'Wolf': 'wolf',
    'Howl': 'wolf',
    'Snake': 'snake',
    'Rattle': 'snake',
    'Frog': 'frog',
    'Croak': 'frog',
    'Insect': 'insect',
    'Bee, wasp, etc.': 'insect',
    'Cricket': 'insect',
    'Mosquito': 'insect',
    
    # Human sounds
    'Applause': 'applause',
    'Clapping': 'applause',
    'Footsteps': 'footsteps',
    'Walk, footsteps': 'footsteps',
    'Run': 'footsteps',
    'Knock': 'knock',
    'Tap': 'knock',
    'Sneeze': 'sneeze',
    'Sneezing': 'sneeze',
    'Cough': 'cough',
    'Laughter': 'laugh',
    'Giggle': 'laugh',
    'Chuckle, chortle': 'laugh',
    'Gasp': 'gasp',
    'Sigh': 'gasp',
    'Breathing': 'gasp',
    'Whispering': 'whisper',
    
    # Vehicles
    'Car': 'car_horn',
    'Vehicle horn, car horn, honking': 'car_horn',
    'Motor vehicle (road)': 'engine',
    'Truck': 'truck',
    'Motorcycle': 'motorcycle',
    'Bus': 'engine',
    'Accelerating, revving, vroom': 'engine',
    'Tire squeal': 'brakes',
    'Skidding': 'brakes',
    'Brake': 'brakes',
    
    # Food/Eating
    'Chewing, mastication': 'chew',
    'Biting': 'bite',
    'Crunch': 'crunch',
    'Munching': 'chew',
    'Slurp': 'slurp',
    'Sizzle': 'sizzle',
    'Frying (food)': 'sizzle',
    'Boiling': 'boil',
    
    # Technology
    'Camera': 'camera',
    'Phone': 'phone',
    'Smartphone': 'phone',
    
    # Miscellaneous
    'Zipper (clothing)': 'zipper',
    'Rustle': 'paper',
    'Crumpling, crinkling': 'paper',
    'Squeak': 'rubber',
    'Bounce': 'spring',
    'Boing': 'spring',
    'Door': 'door',
    'Creak': 'door',
    'Wood': 'thud',
    'Metal': 'crash',
    'Scissors': 'click',
    'Scrape': 'scratch'
}

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

def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, use_animation=True):
    """
    Create a subtitle file with onomatopoeia effects from an audio file.
    Now supports animated effects using ASS format.
    
    Args:
        audio_path (str): Path to the audio file
        output_srt_path (str): Path for the output subtitle file (will use .ass for animated)
        log_func: Logging function
        use_animation (bool): Whether to use animated effects (default True)
        
    Returns:
        tuple: (success: bool, events: list) - Success status and detected events
    """
    try:
        if use_animation:
            # Try to use animated ASS version
            try:
                from onomatopoeia_animator import create_animated_onomatopoeia_ass
                if log_func:
                    log_func("Using animated onomatopoeia effects (ASS format)...")
                
                # Change extension to .ass for animated version
                import os
                ass_path = os.path.splitext(output_srt_path)[0] + '.ass'
                success, events = create_animated_onomatopoeia_ass(audio_path, ass_path, log_func)
                
                # Update the output path reference for the caller
                if success and hasattr(create_onomatopoeia_srt, '_last_output_path'):
                    create_onomatopoeia_srt._last_output_path = ass_path
                
                return success, events
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, falling back to static SRT...")
                use_animation = False
        
        # Fallback to static SRT version
        if log_func:
            log_func("Creating static onomatopoeia effects (SRT format)...")
            
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
            log_func(f"Static onomatopoeia SRT created: {output_srt_path} with {len(events)} events")
        return True, events
        
    except Exception as e:
        if log_func:
            log_func(f"Error creating onomatopoeia subtitle file: {e}")
        return False, []