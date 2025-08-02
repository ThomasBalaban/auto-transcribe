"""
Fixed Modern Onomatopoeia Detector with correct model identifiers
The issue was using an incorrect CLAP model name.
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
    FIXED with correct model identifiers.
    """
    
    def __init__(self, 
             sensitivity: float = 0.5,
             chunk_duration: float = 2.0,
             step_size: float = 0.5,
             log_func=None):
        """
        Initialize the modern onomatopoeia detector.
        UPDATED with much lower energy threshold for quiet audio.
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
        
        # Audio processing settings - LOWERED THRESHOLD
        self.sample_rate = 48000
        self.min_energy_threshold = 0.0001  # CHANGED: was 0.001, now 0.0001 (10x more sensitive)
        
        self.log_func(f"🎚️  Energy threshold set to {self.min_energy_threshold} (lower = more sensitive)")
        
        # Initialize models
        self._load_models()
        

    def _load_models(self):
        """Load CLAP and LLM models - FORCED CPU for CLAP compatibility."""
        try:
            self.log_func("Loading modern onomatopoeia detection models...")
            
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
            
            # FORCE CPU FOR EVERYTHING TO AVOID MPS ISSUES
            self.log_func("🔧 Forcing CPU mode for all models (MPS compatibility)")
            self.device = "cpu"
            self.clap_device = "cpu"
            
            # Move CLAP to CPU
            self.clap_model = self.clap_model.to("cpu")
            
            # Load LLM and also put it on CPU
            self.log_func("Loading local LLM for onomatopoeia generation...")
            
            llm_models = [
                "distilgpt2",
                "gpt2",
                "microsoft/DialoGPT-small",
            ]
            
            llm_loaded = False
            for model_name in llm_models:
                try:
                    self.log_func(f"Trying LLM model: {model_name}")
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.llm_model = self.llm_model.to("cpu")  # Force CPU
                    
                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
                    self.log_func(f"✓ Successfully loaded LLM: {model_name}")
                    llm_loaded = True
                    break
                except Exception as e:
                    self.log_func(f"Failed to load {model_name}: {e}")
                    continue
            
            if not llm_loaded:
                raise Exception("Failed to load any LLM model")
            
            self.log_func("✓ Modern onomatopoeia system ready!")
            self.log_func(f"  - CLAP model: CPU (forced for compatibility)")
            self.log_func(f"  - LLM model: CPU (forced for compatibility)")
            self.log_func(f"  - Sensitivity: {self.sensitivity}")
            self.log_func("  - Note: Using CPU for stability, may be slower but more reliable")
            
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
        """Convert audio chunk to natural language description using CLAP - FORCED CPU."""
        try:
            # Process audio with CLAP
            inputs = self.clap_processor(
                audios=audio_chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # FORCE ALL INPUTS TO CPU
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to("cpu")
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)
            
            # Candidate descriptions
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
                "insect buzzing"
            ]
            
            # Process text descriptions
            text_inputs = self.clap_processor(
                text=candidate_descriptions,
                return_tensors="pt",
                padding=True
            )
            
            # FORCE ALL TEXT INPUTS TO CPU
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
            confidence_threshold = 0.05 + (self.sensitivity * 0.1)  # LOWERED: was 0.1 + 0.15, now 0.05 + 0.1
            
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
        Convert description to comic book onomatopoeia using local LLM.
        IMPROVED with better prompting and fallback handling.
        """
        try:
            # IMPROVED PROMPT - More direct and with examples
            prompt = f"Convert to ONE comic book sound effect word:\n\n{description} → "
            
            self.log_func(f"🤖 LLM prompt: '{prompt}'")
            
            # Tokenize and force to CPU
            inputs = self.llm_tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to("cpu")
            
            # Generate with parameters optimized for single words
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=3,     # Even shorter - just 1-2 words max
                    temperature=0.5,      # Less random for more predictable output
                    do_sample=False,      # Use greedy decoding for consistency
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            self.log_func(f"🤖 LLM raw output: '{response}'")
            
            # IMPROVED cleaning and validation
            onomatopoeia = self._clean_onomatopoeia_improved(response)
            
            if onomatopoeia and len(onomatopoeia) >= 3:  # Must be at least 3 characters
                self.log_func(f"✨ LLM generated onomatopoeia: '{onomatopoeia}'")
                return onomatopoeia
            else:
                # Fallback to rule-based generation
                fallback = self._fallback_onomatopoeia(description)
                self.log_func(f"🔄 LLM failed, using fallback: '{fallback}' (LLM gave: '{response}')")
                return fallback
            
        except Exception as e:
            self.log_func(f"Error in LLM generation: {e}")
            fallback = self._fallback_onomatopoeia(description)
            self.log_func(f"🔄 Error fallback: '{fallback}'")
            return fallback  
    
    
    def _clean_onomatopoeia_improved(self, text: str) -> Optional[str]:
        """IMPROVED cleaning and validation of generated onomatopoeia."""
        if not text:
            return None
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            "sound effect", "sound", "noise", "the", "a", "an", "of", "in", "on", "at", "is", "are", "was", "were"
        ]
        
        # Take first meaningful word
        words = text.split()
        if not words:
            return None
        
        # Find first word that's not in unwanted phrases
        for word in words:
            clean_word = word.upper().strip('.,!?"():')
            clean_word = ''.join(c for c in clean_word if c.isalpha() or c == '-')
            
            if (len(clean_word) >= 3 and 
                clean_word.lower() not in unwanted_phrases and
                not clean_word.lower().startswith('sound')):
                return clean_word
        
        # If no good word found, try the first word anyway
        if words:
            first_word = words[0].upper().strip('.,!?"():')
            first_word = ''.join(c for c in first_word if c.isalpha() or c == '-')
            if len(first_word) >= 2:
                return first_word
        
        return None


    def _fallback_onomatopoeia(self, description: str) -> str:
        """IMPROVED fallback rule-based onomatopoeia generation."""
        description_lower = description.lower()
        
        # More specific keyword mapping
        if any(word in description_lower for word in ['explosion', 'exploding', 'boom', 'blast']):
            return random.choice(['BOOM', 'BANG', 'KABOOM', 'BLAST', 'WHAM'])
        elif any(word in description_lower for word in ['glass', 'breaking', 'shatter']):
            return random.choice(['CRASH', 'SHATTER', 'SMASH', 'CRACK'])
        elif any(word in description_lower for word in ['metal', 'collision', 'colliding']):
            return random.choice(['CLANG', 'CRASH', 'BANG', 'CLANK'])
        elif any(word in description_lower for word in ['water', 'splash', 'liquid']):
            return random.choice(['SPLASH', 'GLUG', 'DRIP', 'SPLASH'])
        elif any(word in description_lower for word in ['door', 'slam']):
            return random.choice(['SLAM', 'BANG', 'THUD', 'WHAM'])
        elif any(word in description_lower for word in ['click', 'beep', 'electronic']):
            return random.choice(['CLICK', 'BEEP', 'BUZZ', 'BLEEP'])
        elif any(word in description_lower for word in ['wind', 'blow']):
            return random.choice(['WHOOSH', 'SWOOSH', 'WOOSH'])
        elif any(word in description_lower for word in ['footstep', 'walk', 'step']):
            return random.choice(['STOMP', 'THUD', 'TAP', 'STEP'])
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
        else:
            # Generic sound effects
            return random.choice(['THUD', 'WHOMP', 'BUMP'])

    def detect_sounds_in_chunk(self, audio_chunk: np.ndarray, chunk_start_time: float) -> List[Dict]:
        """
        Detect sounds in audio chunk - UPDATED for better quiet audio handling.
        """
        if self.clap_model is None or self.llm_model is None:
            return []
        
        events = []
        
        try:
            self.log_func(f"\n🔍 MODERN SYSTEM analyzing chunk at {chunk_start_time:.1f}s")
            
            # Calculate energy
            energy = self._calculate_audio_energy(audio_chunk)
            self.log_func(f"📊 Audio energy: {energy:.4f} (threshold: {self.min_energy_threshold})")
            
            # UPDATED: More permissive energy check
            if energy < self.min_energy_threshold:
                self.log_func(f"🔇 Skipping quiet audio (energy {energy:.4f} < {self.min_energy_threshold})")
                return []
            elif energy < 0.01:  # Still low but above threshold
                self.log_func(f"🔉 Processing quiet audio (energy: {energy:.4f})")
            else:
                self.log_func(f"🔊 Processing normal audio (energy: {energy:.4f})")
            
            # Get audio description using CLAP
            description = self._audio_to_description(audio_chunk)
            if not description:
                self.log_func("❌ CLAP: No description generated")
                return []
            
            # Generate onomatopoeia using LLM
            onomatopoeia = self._description_to_onomatopoeia(description)
            if not onomatopoeia:
                self.log_func("❌ LLM: No onomatopoeia generated")
                return []
            
            # Determine duration (simplified for now)
            duration = random.uniform(0.5, 1.5)
            
            # Find peak timing within chunk for better placement
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
            self.log_func(f"✨ MODERN EVENT CREATED: '{onomatopoeia}' at {peak_time:.1f}s ({duration:.1f}s)")
            self.log_func(f"   Source: '{description}' (energy: {energy:.4f})")
            
        except Exception as e:
            self.log_func(f"💥 Error in modern pipeline: {e}")
        
        return events

    def analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """
        Analyze audio file with IMPROVED event deduplication and filtering.
        """
        if not os.path.exists(audio_path):
            self.log_func(f"Audio file not found: {audio_path}")
            return []
        
        try:
            self.log_func(f"\n🚀 MODERN SYSTEM analyzing: {audio_path}")
            
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
            
            # IMPROVED: Deduplicate and filter events
            filtered_events = self._deduplicate_and_filter_events(all_events)
            
            self.log_func(f"🎯 FILTERED ANALYSIS: {len(filtered_events)} events after deduplication")
            
            return filtered_events
            
        except Exception as e:
            self.log_func(f"Error in modern audio analysis: {e}")
            return []
        
    def _deduplicate_and_filter_events(self, events: List[Dict]) -> List[Dict]:
        """
        Remove duplicate events and filter out low-quality detections.
        """
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


# Quick test function
def test_modern_system():
    """Quick test to see if the modern system loads."""
    print("Testing modern onomatopoeia system...")
    try:
        detector = ModernOnomatopoeiaDetector(log_func=print)
        print("✅ Modern system loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Modern system failed: {e}")
        return False


if __name__ == "__main__":
    test_modern_system()