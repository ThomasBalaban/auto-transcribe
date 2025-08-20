"""
Audio Enhancement Module for Onomatopoeia Detection.
Handles CLAP processing and audio event enhancement.
"""

import random
import numpy as np
import librosa
import torch
from typing import List, Dict, Optional
from ollama_integration import OllamaLLM


class AudioEnhancer:
    """
    Handles audio enhancement using CLAP model and onomatopoeia generation.
    """
    
    def __init__(self, log_func=None):
        self.log_func = log_func or print
        self.clap_model = None
        self.clap_processor = None
        self.ollama_llm = None
        self._initialize_clap_ollama()
    
    def _initialize_clap_ollama(self):
        """Initialize CLAP and Ollama systems"""
        try:
            from transformers import ClapModel, ClapProcessor
            
            # Load CLAP model
            self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.clap_model = self.clap_model.to("cpu")  # CPU for compatibility
            
            # Initialize Ollama
            self.ollama_llm = OllamaLLM(log_func=self.log_func)
            
            self.log_func("✅ CLAP + Ollama pipeline ready")
            
        except Exception as e:
            self.log_func(f"Warning: CLAP/Ollama initialization failed: {e}")
            self.clap_model = None
            self.ollama_llm = None
    
    def enhance_audio_events(self, audio_events: List[Dict], audio_path: str) -> List[Dict]:
        """Enhance audio events with CLAP descriptions and Ollama generation"""
        if not self.clap_model or not audio_events:
            return audio_events
        
        enhanced_events = []
        audio, sr = librosa.load(audio_path, sr=48000)
        
        for event in audio_events:
            try:
                # Extract audio window around event
                onset_time = event['time']
                window_start = max(0, int((onset_time - 1.0) * sr))
                window_end = min(len(audio), int((onset_time + 1.0) * sr))
                audio_chunk = audio[window_start:window_end]
                
                # Get CLAP description
                description = self._get_clap_description(audio_chunk, event)
                
                # Generate onomatopoeia
                if description:
                    onomatopoeia = self._generate_onomatopoeia(description, event)
                    if onomatopoeia:
                        event['clap_description'] = description
                        event['generated_word'] = onomatopoeia
                        enhanced_events.append(event)
                
            except Exception as e:
                self.log_func(f"Error enhancing event at {event['time']:.1f}s: {e}")
                enhanced_events.append(event)  # Keep original event
        
        return enhanced_events
    
    def _get_clap_description(self, audio_chunk: np.ndarray, event: Dict) -> Optional[str]:
        """Get CLAP description for audio chunk"""
        try:
            # Process audio with CLAP
            inputs = self.clap_processor(
                audios=audio_chunk,
                sampling_rate=48000,
                return_tensors="pt"
            )
            
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to("cpu")
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)
            
            # Gaming-focused descriptions based on onset type
            onset_type = event.get('onset_type', 'GENERAL')
            
            if onset_type == 'LOW_FREQ':
                descriptions = [
                    "loud explosion sound", "thunder rumbling", "bass impact",
                    "building collapse", "heavy machinery"
                ]
            elif onset_type == 'HIGH_FREQ':
                descriptions = [
                    "gunshot firing", "metal collision", "glass breaking",
                    "electronic beeping", "sharp impact"
                ]
            else:
                descriptions = [
                    "impact sound", "collision", "mechanical sound",
                    "footsteps", "object hitting"
                ]
            
            # Process descriptions
            text_inputs = self.clap_processor(
                text=descriptions,
                return_tensors="pt",
                padding=True
            )
            
            for key in text_inputs:
                if hasattr(text_inputs[key], 'to'):
                    text_inputs[key] = text_inputs[key].to("cpu")
            
            with torch.no_grad():
                text_embeds = self.clap_model.get_text_features(**text_inputs)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                audio_embeds.unsqueeze(1), 
                text_embeds.unsqueeze(0), 
                dim=2
            )
            
            best_idx = similarities.argmax().item()
            best_score = similarities.max().item()
            
            if best_score > 0.1:  # Confidence threshold
                return descriptions[best_idx]
            
            return None
            
        except Exception as e:
            self.log_func(f"CLAP description error: {e}")
            return None
    
    def _generate_onomatopoeia(self, description: str, event: Dict) -> Optional[str]:
        """Generate onomatopoeia using Ollama or fallback"""
        # Try Ollama first
        if self.ollama_llm and self.ollama_llm.available:
            result = self.ollama_llm.generate_onomatopoeia(description)
            if result:
                return result
        
        # Fallback generation
        return self._fallback_onomatopoeia(description, event)
    
    def _fallback_onomatopoeia(self, description: str, event: Dict) -> str:
        """Fallback onomatopoeia generation"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['explosion', 'blast']):
            return random.choice(['BOOM!', 'KABOOM!', 'BLAST!'])
        elif any(word in desc_lower for word in ['gun', 'shot']):
            return random.choice(['BANG!', 'SHOT!', 'FIRE!'])
        elif any(word in desc_lower for word in ['metal', 'collision']):
            return random.choice(['CLANG!', 'CRASH!', 'CLANK!'])
        elif any(word in desc_lower for word in ['glass', 'break']):
            return random.choice(['SHATTER!', 'CRASH!', 'SMASH!'])
        elif any(word in desc_lower for word in ['impact', 'hit']):
            return random.choice(['THUD!', 'WHACK!', 'SLAM!'])
        else:
            return random.choice(['BANG!', 'CRASH!', 'THUD!'])