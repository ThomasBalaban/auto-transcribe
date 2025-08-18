"""
Enhanced Modern Onomatopoeia Detector with Ollama LLM integration.
Replaces DistilGPT2 with Ollama's mistral-nemo for better generation.
Fixes word truncation issues (footsteps -> foot problem).
"""

import os
import random
import numpy as np
import librosa # type: ignore
import torch # type: ignore
import requests
import json
from typing import List, Dict, Optional, Tuple
from transformers import ClapModel, ClapProcessor # type: ignore
import warnings
warnings.filterwarnings("ignore")


class OllamaLLM:
    """
    Ollama integration for onomatopoeia generation using mistral-nemo.
    Much better than DistilGPT2 for following instructions.
    """
    
    def __init__(self, model_name="mistral-nemo:latest", base_url="http://localhost:11434", log_func=None):
        self.model_name = model_name
        self.base_url = base_url
        self.log_func = log_func or print
        self.available = self._check_availability()
        
    def _check_availability(self):
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.log_func(f"❌ Ollama not accessible at {self.base_url}")
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                self.log_func(f"⚠️  Model {self.model_name} not found. Available models: {model_names}")
                self.log_func(f"💡 Run: ollama pull {self.model_name}")
                return False
            
            self.log_func(f"✅ Ollama + {self.model_name} ready!")
            return True
            
        except requests.exceptions.RequestException as e:
            self.log_func(f"❌ Ollama connection failed: {e}")
            self.log_func("💡 Make sure Ollama is running: ollama serve")
            return False
    
    def generate_onomatopoeia(self, description: str) -> Optional[str]:
        """
        Generate onomatopoeia using Ollama with improved prompting.
        """
        if not self.available:
            return None
            
        try:
            # Enhanced prompt - much clearer instructions
            prompt = f"""Convert this audio description into ONE comic book sound effect word.

Audio description: "{description}"

Rules:
- Give me ONLY the sound effect word (like BANG, CRASH, SPLASH)
- Use ALL CAPS
- Keep it short (1-2 words max)
- Make it punchy and comic book style
- Don't explain, just give the word

Sound effect:"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temperature for consistency
                    "top_p": 0.9,
                    "max_tokens": 10,    # Force short responses
                    "stop": ["\n", ".", "!", "?"]  # Stop at punctuation
                }
            }
            
            self.log_func(f"🤖 Ollama prompt: '{description}' → ?")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_output = result.get('response', '').strip()
                
                self.log_func(f"🤖 Ollama raw: '{raw_output}'")
                
                # Clean the output
                cleaned = self._clean_ollama_output(raw_output)
                
                if cleaned:
                    self.log_func(f"✨ Ollama generated: '{cleaned}'")
                    return cleaned
                else:
                    self.log_func(f"🔄 Ollama output unusable, using fallback")
                    return None
            else:
                self.log_func(f"❌ Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.log_func(f"💥 Ollama generation error: {e}")
            return None
    
    def _clean_ollama_output(self, text: str) -> Optional[str]:
        """
        Clean Ollama output with improved logic to preserve compound words.
        FIXES the footsteps -> foot problem!
        """
        if not text:
            return None
        
        # Remove common unwanted prefixes/suffixes
        text = text.upper().strip()
        
        # Remove quotes if they wrap the whole thing
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove punctuation but keep hyphens for compound words
        import re
        text = re.sub(r'[^\w\s\-]', '', text)
        
        # Split into words
        words = text.split()
        
        # Filter out unwanted words
        unwanted = {'THE', 'A', 'AN', 'OF', 'IN', 'ON', 'AT', 'IS', 'ARE', 'WAS', 'WERE', 
                   'SOUND', 'EFFECT', 'NOISE', 'AUDIO'}
        
        # Find the best word(s) - IMPROVED LOGIC
        good_words = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in unwanted and 
                not word.startswith('SOUND') and
                word.isalpha() or '-' in word):  # Allow hyphenated words
                good_words.append(word)
        
        # Smart word selection - prefer compound onomatopoeia
        if good_words:
            # If we have multiple words, try to combine them intelligently
            if len(good_words) >= 2:
                # Check if it's a compound onomatopoeia (like "TICK TOCK", "DING DONG")
                combined = ' '.join(good_words[:2])  # Take first two words
                if len(combined) <= 15:  # Reasonable length
                    return combined
            
            # Single word - just return the first good one
            return good_words[0]
        
        # Fallback - take first word if it's reasonable
        if words and len(words[0]) >= 2:
            return words[0]
        
        return None


class ModernOnomatopoeiaDetector:
    """
    Enhanced onomatopoeia detector using CLAP + Ollama.
    Fixed word truncation and improved generation quality.
    """
    
    def __init__(self, 
                 sensitivity: float = 0.5,
                 chunk_duration: float = 2.0,
                 step_size: float = 0.5,
                 log_func=None):
        """
        Initialize the enhanced detector with Ollama integration.
        """
        self.sensitivity = sensitivity
        self.chunk_duration = chunk_duration
        self.step_size = step_size
        self.log_func = log_func or print
        
        # Models will be loaded lazily
        self.clap_model = None
        self.clap_processor = None
        self.ollama_llm = None
        
        # Audio processing settings - ADJUSTED threshold
        self.sample_rate = 48000
        self.min_energy_threshold = 0.0005  # Slightly higher to reduce noise
        
        self.log_func(f"🎚️  Energy threshold: {self.min_energy_threshold}")
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load CLAP and Ollama models."""
        try:
            self.log_func("Loading enhanced onomatopoeia detection models...")
            
            # Load CLAP model for audio captioning
            self.log_func("Loading CLAP audio captioning model...")
            
            clap_models_to_try = [
                "laion/clap-htsat-unfused",
                "laion/clap-htsat-fused",
            ]
            
            clap_loaded = False
            for model_name in clap_models_to_try:
                try:
                    self.log_func(f"Trying CLAP model: {model_name}")
                    self.clap_model = ClapModel.from_pretrained(model_name)
                    self.clap_processor = ClapProcessor.from_pretrained(model_name)
                    self.log_func(f"✓ Successfully loaded CLAP model: {model_name}")
                    clap_loaded = True
                    break
                except Exception as e:
                    self.log_func(f"Failed to load {model_name}: {e}")
                    continue
            
            if not clap_loaded:
                raise Exception("Failed to load any CLAP model")
            
            # Force CPU for compatibility
            self.log_func("🔧 Using CPU mode for CLAP (MPS compatibility)")
            self.device = "cpu"
            self.clap_model = self.clap_model.to("cpu")
            
            # Initialize Ollama LLM
            self.log_func("Initializing Ollama LLM integration...")
            self.ollama_llm = OllamaLLM(log_func=self.log_func)
            
            if self.ollama_llm.available:
                self.log_func("✓ Enhanced system ready with Ollama!")
                self.log_func(f"  - CLAP model: CPU")
                self.log_func(f"  - LLM: Ollama mistral-nemo")
                self.log_func(f"  - Sensitivity: {self.sensitivity}")
            else:
                self.log_func("⚠️  Ollama not available, will use fallback generation")
            
        except Exception as e:
            self.log_func(f"Failed to load enhanced models: {e}")
            self.clap_model = None
            self.ollama_llm = None
            raise
    
    def _calculate_audio_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        if len(audio_chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk**2)))
    
    def _audio_to_description(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Convert audio chunk to natural language description using CLAP."""
        try:
            # Process audio with CLAP
            inputs = self.clap_processor(
                audios=audio_chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Force all inputs to CPU
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to("cpu")
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)
            
            # EXPANDED candidate descriptions for better variety
            candidate_descriptions = [
                "loud explosion sound",
                "glass breaking and shattering", 
                "metal objects colliding",
                "wooden objects hitting",
                "water splashing loudly",
                "paper rustling",
                "electronic beeping sound",
                "mechanical clicking",
                "wind blowing strongly",
                "footsteps on hard surface",  # THIS should become STOMP, not FOOT
                "door slamming shut",
                "car engine running",
                "dog barking loudly",
                "thunder rumbling",
                "fire crackling",
                "crowd cheering",
                "bell ringing clearly",
                "whistle blowing",
                "rubber squeaking",
                "fabric rustling",
                "bubble popping",
                "spring bouncing",
                "liquid pouring",
                "zipper closing",
                "keyboard typing rapidly",
                "phone ringing",
                "alarm beeping",
                "bird chirping",
                "cat meowing",
                "insect buzzing",
                "hammer hitting nail",
                "saw cutting wood",
                "drill boring hole",
                "vacuum cleaner running",
                "microwave beeping",
                "coins jingling",
                "chain rattling",
                "sword clashing",
                "arrow whistling",
                "rocket launching"
            ]
            
            # Process text descriptions
            text_inputs = self.clap_processor(
                text=candidate_descriptions,
                return_tensors="pt",
                padding=True
            )
            
            # Force all text inputs to CPU
            for key in text_inputs:
                if isinstance(text_inputs[key], torch.Tensor):
                    text_inputs[key] = text_inputs[key].to("cpu")
            
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
            confidence_threshold = 0.05 + (self.sensitivity * 0.1)
            
            if best_score < confidence_threshold:
                self.log_func(f"🔍 CLAP: Best match '{best_description}' score {best_score:.3f} below threshold {confidence_threshold:.3f}")
                return None
            
            self.log_func(f"🎯 CLAP description: '{best_description}' (confidence: {best_score:.3f})")
            return best_description
            
        except Exception as e:
            self.log_func(f"Error in audio captioning: {e}")
            return None
    
    def _description_to_onomatopoeia(self, description: str) -> Optional[str]:
        """
        Convert description to onomatopoeia using Ollama or fallback.
        FIXED to preserve compound words like "footsteps" → "STOMP" not "FOOT".
        """
        # Try Ollama first
        if self.ollama_llm and self.ollama_llm.available:
            onomatopoeia = self.ollama_llm.generate_onomatopoeia(description)
            if onomatopoeia and len(onomatopoeia) >= 3:
                return onomatopoeia
        
        # Fallback to improved rule-based generation
        fallback = self._enhanced_fallback_onomatopoeia(description)
        self.log_func(f"🔄 Using enhanced fallback: '{fallback}'")
        return fallback
    
    def _enhanced_fallback_onomatopoeia(self, description: str) -> str:
        """
        ENHANCED fallback rule-based onomatopoeia generation.
        FIXES footsteps → STOMP (not FOOT).
        """
        description_lower = description.lower()
        
        # More specific and comprehensive keyword mapping
        if any(word in description_lower for word in ['explosion', 'exploding', 'boom', 'blast']):
            return random.choice(['BOOM', 'BANG', 'KABOOM', 'BLAST', 'WHAM'])
        elif any(word in description_lower for word in ['glass', 'breaking', 'shatter']):
            return random.choice(['CRASH', 'SHATTER', 'SMASH', 'CRACK'])
        elif any(word in description_lower for word in ['metal', 'collision', 'colliding']):
            return random.choice(['CLANG', 'CRASH', 'BANG', 'CLANK'])
        elif any(word in description_lower for word in ['water', 'splash', 'liquid']):
            return random.choice(['SPLASH', 'GLUG', 'DRIP', 'WHOOSH'])
        elif any(word in description_lower for word in ['door', 'slam']):
            return random.choice(['SLAM', 'BANG', 'THUD', 'WHAM'])
        elif any(word in description_lower for word in ['click', 'beep', 'electronic']):
            return random.choice(['CLICK', 'BEEP', 'BUZZ', 'BLEEP'])
        elif any(word in description_lower for word in ['wind', 'blow']):
            return random.choice(['WHOOSH', 'SWOOSH', 'WOOSH'])
        # FIXED: footsteps now properly becomes STOMP instead of FOOT
        elif any(word in description_lower for word in ['footstep', 'walk', 'step', 'walking']):
            return random.choice(['STOMP', 'THUD', 'CLUNK', 'STEP'])
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
        elif any(word in description_lower for word in ['whistle']):
            return random.choice(['TWEET', 'WHISTLE', 'TOOT'])
        elif any(word in description_lower for word in ['pop', 'bubble']):
            return random.choice(['POP', 'BLOOP', 'PLOP'])
        elif any(word in description_lower for word in ['hammer', 'hitting']):
            return random.choice(['BANG', 'WHACK', 'THWACK'])
        elif any(word in description_lower for word in ['saw', 'cutting']):
            return random.choice(['BZZZZ', 'RRRRR', 'WHIRR'])
        elif any(word in description_lower for word in ['drill', 'boring']):
            return random.choice(['WHIRR', 'BZZZZ', 'DRILL'])
        else:
            # Generic sound effects
            return random.choice(['THUD', 'WHOMP', 'BUMP'])

    def detect_sounds_in_chunk(self, audio_chunk: np.ndarray, chunk_start_time: float) -> List[Dict]:
        """Detect sounds in audio chunk using enhanced pipeline."""
        if self.clap_model is None:
            return []
        
        events = []
        
        try:
            self.log_func(f"\n🔍 ENHANCED SYSTEM analyzing chunk at {chunk_start_time:.1f}s")
            
            # Calculate energy
            energy = self._calculate_audio_energy(audio_chunk)
            self.log_func(f"📊 Audio energy: {energy:.4f} (threshold: {self.min_energy_threshold})")
            
            # Energy check
            if energy < self.min_energy_threshold:
                self.log_func(f"🔇 Skipping quiet audio (energy {energy:.4f} < {self.min_energy_threshold})")
                return []
            
            # Get audio description using CLAP
            description = self._audio_to_description(audio_chunk)
            if not description:
                self.log_func("❌ CLAP: No description generated")
                return []
            
            # Generate onomatopoeia using Ollama/fallback
            onomatopoeia = self._description_to_onomatopoeia(description)
            if not onomatopoeia:
                self.log_func("❌ LLM: No onomatopoeia generated")
                return []
            
            # Determine duration
            duration = random.uniform(0.5, 1.5)
            
            # Find peak timing within chunk
            peak_time = chunk_start_time + (len(audio_chunk) / (2 * self.sample_rate))
            
            event = {
                'word': onomatopoeia,
                'start_time': peak_time,
                'end_time': peak_time + duration,
                'confidence': 0.8,
                'energy': energy,
                'description': description,
                'chunk_start': chunk_start_time
            }
            
            events.append(event)
            self.log_func(f"✨ ENHANCED EVENT: '{onomatopoeia}' at {peak_time:.1f}s ({duration:.1f}s)")
            self.log_func(f"   Source: '{description}' (energy: {energy:.4f})")
            
        except Exception as e:
            self.log_func(f"💥 Error in enhanced pipeline: {e}")
        
        return events

    def analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Analyze audio file with enhanced detection and deduplication."""
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"\n🚀 ENHANCED SYSTEM analyzing: {audio_path}")
            
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
            
            self.log_func(f"\n📊 RAW ANALYSIS: {len(all_events)} events from {chunk_count} chunks")
            
            # Deduplicate and filter events
            filtered_events = self._deduplicate_and_filter_events(all_events)
            
            self.log_func(f"🎯 FINAL ANALYSIS: {len(filtered_events)} events after deduplication")
            
            return filtered_events
            
        except Exception as e:
            self.log_func(f"Error in enhanced audio analysis: {e}")
            return []
    
    def _deduplicate_and_filter_events(self, events: List[Dict]) -> List[Dict]:
        """Remove duplicate events and filter out low-quality detections."""
        if not events:
            return []
        
        self.log_func(f"🔍 Deduplicating {len(events)} events...")
        
        # Sort events by start time
        events.sort(key=lambda x: x['start_time'])
        
        filtered_events = []
        
        for event in events:
            # Skip events with generic/bad words
            word = event['word']
            if word in ['SOUND', 'THE', 'A', 'AN', 'OF', 'IS']:
                self.log_func(f"   ❌ Skipping generic word: '{word}'")
                continue
            
            # Skip if too close to previous event (within 1 second)
            if filtered_events:
                last_event = filtered_events[-1]
                time_diff = event['start_time'] - last_event['start_time']
                
                if time_diff < 1.0:  # Less than 1 second apart
                    # Keep the one with higher energy
                    if event['energy'] > last_event['energy']:
                        filtered_events[-1] = event  # Replace last event
                        self.log_func(f"   🔄 Replaced '{last_event['word']}' with '{word}' (higher energy)")
                    else:
                        self.log_func(f"   ❌ Skipping '{word}' too close to '{last_event['word']}'")
                    continue
            
            # Add event
            filtered_events.append(event)
            self.log_func(f"   ✅ Kept '{word}' at {event['start_time']:.1f}s")
        
        return filtered_events

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


# Quick test function for the enhanced system
def test_enhanced_system():
    """Test the enhanced Ollama-based system."""
    print("Testing enhanced onomatopoeia system with Ollama...")
    try:
        detector = ModernOnomatopoeiaDetector(log_func=print)
        
        if detector.ollama_llm and detector.ollama_llm.available:
            print("✅ Enhanced system with Ollama loaded successfully!")
            
            # Test the word cleaning fix
            test_descriptions = [
                "footsteps on hard surface",
                "loud explosion sound", 
                "glass breaking and shattering"
            ]
            
            print("\n🧪 Testing generation:")
            for desc in test_descriptions:
                result = detector._description_to_onomatopoeia(desc)
                print(f"  '{desc}' → '{result}'")
            
            return True
        else:
            print("⚠️  System loaded but Ollama not available")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced system failed: {e}")
        return False


def create_enhanced_onomatopoeia_srt(audio_path: str, 
                                   output_srt_path: str, 
                                   sensitivity: float = 0.5,
                                   animation_setting: str = "Random",
                                   log_func=None,
                                   use_animation: bool = True) -> Tuple[bool, List[Dict]]:
    """
    Create onomatopoeia subtitle file using enhanced Ollama pipeline.
    """
    try:
        if log_func:
            log_func("=== ENHANCED ONOMATOPOEIA SYSTEM ===")
            log_func("Using CLAP audio captioning + Ollama mistral-nemo")
        
        # Create enhanced detector
        detector = ModernOnomatopoeiaDetector(
            sensitivity=sensitivity,
            log_func=log_func
        )
        
        # Analyze audio
        events = detector.analyze_audio_file(audio_path)
        
        if not events:
            if log_func:
                log_func("No onomatopoeia events detected")
            return False, []
        
        # Create output
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
                    log_func(f"Enhanced animated onomatopoeia created: {len(events)} events")
                
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
                log_func(f"Enhanced static onomatopoeia created: {len(events)} events")
            
            return True, events
            
    except Exception as e:
        if log_func:
            log_func(f"Error in enhanced onomatopoeia system: {e}")
        return False, []


if __name__ == "__main__":
    test_enhanced_system()