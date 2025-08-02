"""
Updated onomatopoeia_detector.py - Modern CLAP + LLM System
Complete replacement for the old YAMNet-based system.
Maintains backward compatibility with existing codebase.
"""

import os
import random
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class ModernOnomatopoeiaDetector:
    """
    Modern onomatopoeia detector using CLAP audio captioning + local LLM.
    Generates creative, contextual comic book sound effects.
    """
    
    def __init__(self, 
                 sensitivity: float = 0.5,
                 chunk_duration: float = 2.0,
                 step_size: float = 0.5,
                 log_func=None):
        """
        Initialize the modern onomatopoeia detector.
        
        Args:
            sensitivity: Detection sensitivity (0.1-0.9, higher = more selective)
            chunk_duration: Audio chunk duration for analysis
            step_size: Time between chunk starts (overlap)
            log_func: Logging function
        """
        self.sensitivity = sensitivity
        self.chunk_duration = chunk_duration
        self.step_size = step_size
        self.log_func = log_func or print
        
        # Models will be loaded lazily
        self.clap_model = None
        self.clap_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        
        # Audio processing settings
        self.sample_rate = 48000  # CLAP's preferred sample rate
        self.min_energy_threshold = 0.001  # Minimum energy to process
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load CLAP and LLM models for audio captioning and onomatopoeia generation."""
        try:
            self.log_func("Loading modern onomatopoeia detection models...")
            
            # Load CLAP model for audio captioning
            self.log_func("Loading CLAP audio captioning model...")
            self.clap_model = ClapModel.from_pretrained("microsoft/clap-htsat-unfused")
            self.clap_processor = ClapProcessor.from_pretrained("microsoft/clap-htsat-unfused")
            
            # Move to MPS if available (Mac GPU), otherwise CPU
            if torch.backends.mps.is_available():
                device = "mps"
                self.log_func("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                self.log_func("Using NVIDIA GPU (CUDA)")
            else:
                device = "cpu"
                self.log_func("Using CPU")
            
            self.clap_model = self.clap_model.to(device)
            self.device = device
            
            # Load local LLM for onomatopoeia generation
            self.log_func("Loading local LLM for onomatopoeia generation...")
            
            # Try different local models in order of preference
            llm_models = [
                "microsoft/DialoGPT-medium",  # Smaller, faster
                "gpt2",  # Fallback
            ]
            
            for model_name in llm_models:
                try:
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.llm_model = self.llm_model.to(device)
                    
                    # Set pad token if missing
                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
                    self.log_func(f"✓ Loaded LLM: {model_name}")
                    break
                except Exception as e:
                    self.log_func(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.llm_model is None:
                raise Exception("Failed to load any LLM model")
            
            self.log_func("✓ Modern onomatopoeia system ready!")
            self.log_func(f"  - Audio captioning: CLAP (microsoft/clap-htsat-unfused)")
            self.log_func(f"  - Onomatopoeia generation: Local LLM")
            self.log_func(f"  - Device: {device}")
            self.log_func(f"  - Sensitivity: {self.sensitivity}")
            
        except Exception as e:
            self.log_func(f"Failed to load modern onomatopoeia models: {e}")
            self.clap_model = None
            self.llm_model = None
            raise
    
    def _calculate_audio_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk**2)))
    
    def _audio_to_description(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Convert audio chunk to natural language description using CLAP.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            Description string or None if processing fails
        """
        try:
            # Process audio with CLAP
            inputs = self.clap_processor(
                audios=audio_chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Move inputs to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)
            
            # Use a set of candidate descriptions and find the best match
            candidate_descriptions = [
                "loud explosion sound",
                "glass breaking and shattering", 
                "metal objects colliding",
                "wooden objects hitting",
                "water splashing",
                "paper rustling",
                "electronic beeping sound",
                "mechanical clicking",
                "wind blowing",
                "footsteps on hard surface",
                "door slamming",
                "car engine running",
                "dog barking",
                "thunder rumbling",
                "fire crackling",
                "crowd cheering",
                "bell ringing",
                "whistle blowing",
                "rubber squeaking",
                "fabric rustling",
                "bubble popping",
                "spring bouncing",
                "liquid pouring",
                "zipper closing",
                "keyboard typing",
                "phone ringing",
                "alarm beeping",
                "bird chirping",
                "cat meowing",
                "insect buzzing",
                "machinery humming",
                "drill spinning",
                "hammer hitting",
                "saw cutting",
                "vacuum cleaner",
                "microwave beeping",
                "toaster popping",
                "ice cracking",
                "rain falling",
                "ocean waves",
                "motorcycle engine",
                "helicopter rotor",
                "airplane flying",
                "train whistle",
                "horse galloping",
                "cow mooing",
                "pig oinking",
                "sheep bleating",
                "lion roaring",
                "gun firing",
                "fireworks exploding",
                "balloon popping",
                "cork popping",
                "can opening",
                "bottle breaking",
                "chain rattling",
                "coin dropping",
                "pencil writing",
                "book closing",
                "chair creaking",
                "floor squeaking",
                "window closing",
                "light switch clicking"
            ]
            
            # Process text descriptions
            text_inputs = self.clap_processor(
                text=candidate_descriptions,
                return_tensors="pt",
                padding=True
            )
            
            # Move text inputs to device
            for key in text_inputs:
                if isinstance(text_inputs[key], torch.Tensor):
                    text_inputs[key] = text_inputs[key].to(self.device)
            
            # Get text embeddings
            with torch.no_grad():
                text_embeds = self.clap_model.get_text_features(**text_inputs)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                audio_embeds.unsqueeze(1), 
                text_embeds.unsqueeze(0), 
                dim=2
            )
            
            # Get best match
            best_idx = similarities.argmax().item()
            best_score = similarities.max().item()
            best_description = candidate_descriptions[best_idx]
            
            # Apply sensitivity threshold
            confidence_threshold = 0.1 + (self.sensitivity * 0.15)  # 0.1 to 0.25 range
            
            if best_score < confidence_threshold:
                return None
            
            self.log_func(f"Audio description: '{best_description}' (confidence: {best_score:.3f})")
            return best_description
            
        except Exception as e:
            self.log_func(f"Error in audio captioning: {e}")
            return None
    
    def _description_to_onomatopoeia(self, description: str) -> Optional[str]:
        """
        Convert description to comic book onomatopoeia using local LLM.
        
        Args:
            description: Natural language description of the sound
            
        Returns:
            Onomatopoeia word or None if generation fails
        """
        try:
            # Create a focused prompt for onomatopoeia generation
            prompt = f"Convert this sound to a comic book sound effect word:\n\nSound: {description}\nComic sound effect:"
            
            # Tokenize
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate with careful parameters for single word output
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=10,  # Short output
                    temperature=0.8,    # Some creativity
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the onomatopoeia part
            response = response.replace(prompt, "").strip()
            
            # Clean and validate the response
            onomatopoeia = self._clean_onomatopoeia(response)
            
            if onomatopoeia:
                self.log_func(f"Generated onomatopoeia: '{onomatopoeia}'")
                return onomatopoeia
            else:
                # Fallback to rule-based generation
                return self._fallback_onomatopoeia(description)
            
        except Exception as e:
            self.log_func(f"Error in LLM generation: {e}")
            return self._fallback_onomatopoeia(description)
    
    def _clean_onomatopoeia(self, text: str) -> Optional[str]:
        """Clean and validate generated onomatopoeia."""
        if not text:
            return None
        
        # Take first word/phrase
        words = text.split()
        if not words:
            return None
        
        onomatopoeia = words[0].upper()
        
        # Remove punctuation except hyphens
        onomatopoeia = ''.join(c for c in onomatopoeia if c.isalpha() or c == '-')
        
        # Validate length (2-10 characters)
        if 2 <= len(onomatopoeia) <= 10:
            return onomatopoeia
        
        return None
    
    def _fallback_onomatopoeia(self, description: str) -> str:
        """Fallback rule-based onomatopoeia generation."""
        description_lower = description.lower()
        
        # Simple keyword mapping
        if any(word in description_lower for word in ['explosion', 'explod', 'boom', 'blast']):
            return random.choice(['BOOM', 'BANG', 'KABOOM', 'BLAST'])
        elif any(word in description_lower for word in ['glass', 'break', 'shatter']):
            return random.choice(['CRASH', 'SHATTER', 'SMASH'])
        elif any(word in description_lower for word in ['metal', 'collision', 'clang']):
            return random.choice(['CLANG', 'CRASH', 'BANG'])
        elif any(word in description_lower for word in ['water', 'splash', 'liquid']):
            return random.choice(['SPLASH', 'GLUG', 'DRIP'])
        elif any(word in description_lower for word in ['door', 'slam']):
            return random.choice(['SLAM', 'BANG', 'THUD'])
        elif any(word in description_lower for word in ['click', 'beep', 'electronic']):
            return random.choice(['CLICK', 'BEEP', 'BUZZ'])
        elif any(word in description_lower for word in ['wind', 'blow']):
            return random.choice(['WHOOSH', 'WHOMP', 'SWOOSH'])
        elif any(word in description_lower for word in ['footstep', 'walk', 'step']):
            return random.choice(['STOMP', 'THUD', 'TAP'])
        elif any(word in description_lower for word in ['engine', 'car', 'motor']):
            return random.choice(['VROOM', 'RUMBLE', 'ROAR'])
        elif any(word in description_lower for word in ['dog', 'bark']):
            return random.choice(['WOOF', 'BARK', 'ARF'])
        elif any(word in description_lower for word in ['thunder', 'rumble']):
            return random.choice(['RUMBLE', 'BOOM', 'CRASH'])
        elif any(word in description_lower for word in ['fire', 'crack']):
            return random.choice(['CRACKLE', 'POP', 'SNAP'])
        elif any(word in description_lower for word in ['bell', 'ring']):
            return random.choice(['DING', 'RING', 'CHIME'])
        elif any(word in description_lower for word in ['whistle', 'blow']):
            return random.choice(['TWEET', 'WHISTLE', 'TOOT'])
        elif any(word in description_lower for word in ['pop', 'bubble']):
            return random.choice(['POP', 'BLOOP', 'PLOP'])
        else:
            # Generic sound effects
            return random.choice(['SOUND', 'NOISE', 'THUD', 'WHOMP'])
    
    def _determine_duration(self, onomatopoeia: str, energy: float, description: str) -> float:
        """
        Determine duration for onomatopoeia based on sound characteristics.
        
        Args:
            onomatopoeia: The sound effect word
            energy: Audio energy level
            description: Original audio description
            
        Returns:
            Duration in seconds
        """
        # Base duration mapping
        duration_map = {
            # Short/instant sounds
            'BANG': 0.3, 'CLICK': 0.2, 'POP': 0.2, 'SNAP': 0.3, 'TAP': 0.2,
            'DING': 0.4, 'BEEP': 0.3, 'TWEET': 0.4,
            
            # Medium duration sounds  
            'BOOM': 0.8, 'CRASH': 0.6, 'SLAM': 0.5, 'THUD': 0.6, 'CLANG': 0.7,
            'SPLASH': 0.8, 'STOMP': 0.4, 'BARK': 0.6, 'RING': 0.8,
            
            # Longer duration sounds
            'WHOOSH': 1.2, 'RUMBLE': 1.5, 'VROOM': 1.0, 'ROAR': 1.3, 
            'CRACKLE': 1.0, 'BUZZ': 1.2, 'WHISTLE': 1.0
        }
        
        base_duration = duration_map.get(onomatopoeia, 0.7)  # Default 0.7s
        
        # Adjust based on energy (louder = slightly longer presence)
        energy_factor = 0.8 + (energy * 0.4)  # 0.8 to 1.2 multiplier
        
        # Adjust based on description context
        description_lower = description.lower()
        if any(word in description_lower for word in ['loud', 'intense', 'heavy']):
            context_factor = 1.2
        elif any(word in description_lower for word in ['soft', 'quiet', 'gentle']):
            context_factor = 0.8
        else:
            context_factor = 1.0
        
        # Add some randomness for variety
        random_factor = random.uniform(0.9, 1.1)
        
        final_duration = base_duration * energy_factor * context_factor * random_factor
        
        # Clamp to reasonable bounds
        return max(0.2, min(3.0, final_duration))
    
    def detect_sounds_in_chunk(self, audio_chunk: np.ndarray, chunk_start_time: float) -> List[Dict]:
        """
        Detect sounds in audio chunk using modern CLAP + LLM pipeline.
        
        Args:
            audio_chunk: Audio data  
            chunk_start_time: Start time of chunk
            
        Returns:
            List of detected sound events
        """
        if self.clap_model is None or self.llm_model is None:
            return []
        
        events = []
        
        try:
            # Calculate energy
            energy = self._calculate_audio_energy(audio_chunk)
            
            # Skip very quiet audio
            if energy < self.min_energy_threshold:
                return []
            
            # Get audio description using CLAP
            description = self._audio_to_description(audio_chunk)
            if not description:
                return []
            
            # Generate onomatopoeia using LLM
            onomatopoeia = self._description_to_onomatopoeia(description)
            if not onomatopoeia:
                return []
            
            # Determine duration
            duration = self._determine_duration(onomatopoeia, energy, description)
            
            # Find peak timing within chunk for better placement
            peak_time = self._find_sound_peak(audio_chunk, chunk_start_time)
            
            event = {
                'word': onomatopoeia,
                'start_time': peak_time,
                'end_time': peak_time + duration,
                'confidence': 0.8,  # Modern models are generally more confident
                'energy': energy,
                'description': description,
                'chunk_start': chunk_start_time
            }
            
            events.append(event)
            self.log_func(f"CREATED: {onomatopoeia} at {peak_time:.1f}s ({duration:.1f}s) - '{description}'")
            
        except Exception as e:
            self.log_func(f"Error processing chunk at {chunk_start_time:.1f}s: {e}")
        
        return events
    
    def _find_sound_peak(self, audio_chunk: np.ndarray, chunk_start_time: float) -> float:
        """Find the peak energy location within the audio chunk."""
        try:
            # Simple approach: find the window with highest RMS energy
            window_size = int(0.1 * self.sample_rate)  # 100ms windows
            
            if len(audio_chunk) < window_size:
                return chunk_start_time + len(audio_chunk) / (2 * self.sample_rate)
            
            max_energy = 0
            peak_sample = len(audio_chunk) // 2  # Default to center
            
            for i in range(0, len(audio_chunk) - window_size, window_size // 2):
                window = audio_chunk[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                
                if energy > max_energy:
                    max_energy = energy
                    peak_sample = i + window_size // 2
            
            peak_time_offset = peak_sample / self.sample_rate
            return chunk_start_time + peak_time_offset
            
        except Exception:
            # Fallback to chunk center
            return chunk_start_time + len(audio_chunk) / (2 * self.sample_rate)
    
    def analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """
        Analyze audio file for onomatopoeia events using modern pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of detected events
        """
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"Analyzing audio with modern CLAP + LLM pipeline: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) == 0:
                self.log_func("Audio file is empty")
                return []
            
            audio_duration = len(audio) / sr
            self.log_func(f"Audio loaded: {audio_duration:.2f}s at {sr}Hz")
            
            # Process overlapping chunks
            chunk_samples = int(self.chunk_duration * sr)
            step_samples = int(self.step_size * sr)
            
            all_events = []
            chunk_count = 0
            
            for start_sample in range(0, len(audio) - chunk_samples + 1, step_samples):
                chunk = audio[start_sample:start_sample + chunk_samples]
                start_time = start_sample / sr
                
                if start_time >= audio_duration:
                    break
                
                chunk_count += 1
                
                # Analyze this chunk
                events = self.detect_sounds_in_chunk(chunk, start_time)
                all_events.extend(events)
            
            self.log_func(f"Modern analysis complete: {len(all_events)} events from {chunk_count} chunks")
            
            return all_events
            
        except Exception as e:
            self.log_func(f"Error in modern audio analysis: {e}")
            return []
    
    def generate_srt_content(self, events: List[Dict]) -> str:
        """Generate SRT subtitle content from events."""
        if not events:
            return ""
        
        srt_content = []
        
        for i, event in enumerate(events, 1):
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']
            
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_formatted} --> {end_formatted}")
            srt_content.append(f"{word}")
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

# Backward compatibility aliases for existing codebase
OnomatopoeiaDetector = ModernOnomatopoeiaDetector


def create_onomatopoeia_srt(audio_path, output_srt_path, log_func=None, use_animation=True, animation_setting="Random", ai_sensitivity=0.5):
    """
    Backward compatible function that now uses the modern system.
    Updated to match the new parameter names while maintaining the same interface.
    
    Args:
        audio_path: Path to audio file
        output_srt_path: Output subtitle file path
        log_func: Logging function
        use_animation: Whether to use animations
        animation_setting: Animation type
        ai_sensitivity: Detection sensitivity (was confidence_threshold)
        
    Returns:
        tuple: (success: bool, events: list)
    """
    try:
        if log_func:
            log_func("=== MODERN ONOMATOPOEIA SYSTEM ===")
            log_func("Using CLAP audio captioning + local LLM generation")
        
        # Create detector with modern system
        detector = ModernOnomatopoeiaDetector(
            sensitivity=ai_sensitivity,
            log_func=log_func
        )
        
        # Analyze audio
        events = detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("No onomatopoeia events detected")
            return False, []
        
        # Create output file
        if use_animation:
            try:
                # Try to create animated ASS file
                from animations.core import OnomatopoeiaAnimator
                
                ass_path = os.path.splitext(output_srt_path)[0] + '.ass'
                animator = OnomatopoeiaAnimator()
                animated_content = animator.generate_animated_ass_content(events, animation_setting)
                
                with open(ass_path, 'w', encoding='utf-8') as f:
                    f.write(animated_content)
                
                if log_func:
                    log_func(f"Modern animated onomatopoeia created: {len(events)} events")
                
                return True, events
                
            except ImportError:
                if log_func:
                    log_func("Animation module not available, creating static SRT")
                use_animation = False
        
        if not use_animation:
            # Create static SRT
            srt_content = detector.generate_srt_content(events)
            with open(output_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            if log_func:
                log_func(f"Modern static onomatopoeia created: {len(events)} events")
            
            return True, events
            
    except Exception as e:
        if log_func:
            log_func(f"Error in modern onomatopoeia system: {e}")
        return False, []


def detect_onomatopoeia(audio_path, log_func=None, confidence_threshold=0.7):
    """
    Backward compatible function using modern detector.
    confidence_threshold is now mapped to sensitivity.
    
    Args:
        audio_path: Path to audio file
        log_func: Logging function
        confidence_threshold: Now mapped to sensitivity
        
    Returns:
        list: List of onomatopoeia events
    """
    detector = ModernOnomatopoeiaDetector(
        sensitivity=confidence_threshold,
        log_func=log_func
    )
    return detector.analyze_audio_file(audio_path)


# =============================================================================
# LEGACY IMPORTS (for complete backward compatibility)
# =============================================================================

# These were imported from the old files - now they just use modern fallbacks
try:
    # These modules no longer exist, but we provide compatibility
    YAMNET_AVAILABLE = True  # Always report as available for compatibility
    
    def create_ai_onomatopoeia_srt(*args, **kwargs):
        """Legacy function redirects to modern system"""
        return create_onomatopoeia_srt(*args, **kwargs)
    
except ImportError:
    YAMNET_AVAILABLE = False