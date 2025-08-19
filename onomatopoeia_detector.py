"""
Multimodal Onomatopoeia Detection System.
Complete 4-phase pipeline: Onset Detection + Video Analysis + Multimodal Fusion + Gaming Optimization.
"""

import os
import random
import tempfile
import subprocess
import numpy as np
import librosa # type: ignore
import torch
import requests
import json
from typing import List, Dict, Optional, Tuple

# Import all our phase components
from onset_detector import GamingOnsetDetector
from video_analyzer import VideoAnalyzer
from multimodal_fusion import MultimodalFusionEngine


class OllamaLLM:
    """
    Ollama integration for onomatopoeia generation using mistral-nemo.
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
        """Generate onomatopoeia using Ollama with improved prompting."""
        if not self.available:
            return None
            
        try:
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
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 10,
                    "stop": ["\n", ".", "!", "?"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_output = result.get('response', '').strip()
                cleaned = self._clean_ollama_output(raw_output)
                
                if cleaned:
                    self.log_func(f"✨ Ollama generated: '{cleaned}'")
                    return cleaned
                    
            return None
                
        except Exception as e:
            self.log_func(f"💥 Ollama generation error: {e}")
            return None
    
    def _clean_ollama_output(self, text: str) -> Optional[str]:
        """Clean Ollama output with improved logic to preserve compound words."""
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
        
        # Find the best word(s)
        good_words = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in unwanted and 
                not word.startswith('SOUND') and
                word.isalpha() or '-' in word):
                good_words.append(word)
        
        if good_words:
            if len(good_words) >= 2:
                combined = ' '.join(good_words[:2])
                if len(combined) <= 15:
                    return combined
            return good_words[0]
        
        if words and len(words[0]) >= 2:
            return words[0]
        
        return None


class OnomatopoeiaDetector:
    """
    Main onomatopoeia detection system.
    Handles both video files (multimodal) and audio files (audio-only).
    """
    
    def __init__(self, sensitivity: float = 0.5, device: str = "mps", log_func=None):
        """Initialize the detection system."""
        self.sensitivity = sensitivity
        self.device = device
        self.log_func = log_func or print
        
        # Gaming optimization parameters
        self.max_effects_per_minute = 12
        self.min_effect_spacing = 0.8
        
        self.log_func("🚀 Initializing Multimodal Onomatopoeia Detector...")
        self._initialize_components()
        self.log_func("✅ Multimodal system ready!")

    def _initialize_components(self):
        """Initialize all detection components"""
        try:
            # Initialize audio onset detector
            self.log_func("Loading onset detection system...")
            self.onset_detector = GamingOnsetDetector(
                sensitivity=self.sensitivity,
                log_func=self.log_func
            )
            
            # Initialize video analyzer
            self.log_func("Loading video analysis system...")
            self.video_analyzer = VideoAnalyzer(
                device=self.device,
                log_func=self.log_func
            )
            
            # Initialize fusion engine
            self.log_func("Loading multimodal fusion engine...")
            self.fusion_engine = MultimodalFusionEngine(
                log_func=self.log_func
            )
            
            # Initialize CLAP and Ollama
            self.log_func("Loading CLAP + Ollama pipeline...")
            self._initialize_clap_ollama()
            
        except Exception as e:
            self.log_func(f"Failed to initialize components: {e}")
            raise

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

    def analyze_file(self, input_path: str) -> List[Dict]:
        """
        Main analysis method - automatically detects file type and uses appropriate pipeline.
        
        Args:
            input_path: Path to video or audio file
            
        Returns:
            List of detected onomatopoeia events
        """
        # Determine file type
        file_ext = os.path.splitext(input_path)[1].lower()
        is_video = file_ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        is_audio = file_ext in ['.wav', '.mp3', '.flac', '.m4a']
        
        if is_video:
            return self._analyze_video_file(input_path)
        elif is_audio:
            return self._analyze_audio_file(input_path)
        else:
            self.log_func(f"Unsupported file type: {file_ext}")
            return []

    def _analyze_video_file(self, video_path: str) -> List[Dict]:
        """Full multimodal analysis for video files."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"MULTIMODAL ANALYSIS: {os.path.basename(video_path)}")
            self.log_func(f"{'='*60}")
            
            # Extract audio from video
            audio_path = self._extract_audio_from_video(video_path)
            
            # Phase 1: Audio onset detection
            self.log_func(f"\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            if not audio_events:
                self.log_func("No audio events detected")
                return []
            
            self.log_func(f"✅ Detected {len(audio_events)} audio onset events")
            
            # Phase 2: Video analysis at onset timestamps
            self.log_func(f"\n🎬 PHASE 2: Video Analysis at Onset Timestamps")
            onset_timestamps = [event['time'] for event in audio_events]
            video_analyses = self.video_analyzer.analyze_video_at_timestamps(
                video_path, onset_timestamps, window_duration=2.0
            )
            self.log_func(f"✅ Completed video analysis for {len(video_analyses)} timestamps")
            
            # Phase 3: Enhanced audio analysis with CLAP
            self.log_func(f"\n🎯 PHASE 3: Enhanced Audio Analysis")
            enhanced_audio_events = self._enhance_audio_events(audio_events, audio_path)
            
            # Phase 4: Multimodal fusion
            self.log_func(f"\n🔄 PHASE 4: Multimodal Fusion")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, video_analyses
            )
            
            # Phase 5: Gaming-specific optimizations
            self.log_func(f"\n🎮 PHASE 5: Gaming Content Optimization")
            optimized_effects = self._apply_gaming_optimizations(final_effects)
            
            self.log_func(f"\n🎉 MULTIMODAL ANALYSIS COMPLETE!")
            self.log_func(f"   - Audio events: {len(audio_events)}")
            self.log_func(f"   - Video analyses: {len(video_analyses)}")
            self.log_func(f"   - Final effects: {len(optimized_effects)}")
            
            # Cleanup temp audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            return optimized_effects
            
        except Exception as e:
            self.log_func(f"Error in multimodal analysis: {e}")
            return []

    def _analyze_audio_file(self, audio_path: str) -> List[Dict]:
        """Audio-only analysis for audio files."""
        try:
            self.log_func(f"\n{'='*60}")
            self.log_func(f"AUDIO-ONLY ANALYSIS: {os.path.basename(audio_path)}")
            self.log_func(f"{'='*60}")
            
            # Phase 1: Audio onset detection
            self.log_func(f"\n📊 PHASE 1: Audio Onset Detection")
            audio_events = self.onset_detector.detect_gaming_onsets(audio_path)
            
            if not audio_events:
                self.log_func("No audio events detected")
                return []
            
            # Phase 3: Enhanced audio analysis with CLAP
            self.log_func(f"\n🎯 PHASE 3: Enhanced Audio Analysis")
            enhanced_audio_events = self._enhance_audio_events(audio_events, audio_path)
            
            # Phase 4: Audio-only processing (no video data)
            self.log_func(f"\n🔄 PHASE 4: Audio-Only Processing")
            final_effects = self.fusion_engine.process_multimodal_events(
                enhanced_audio_events, []  # Empty video analyses
            )
            
            # Phase 5: Gaming optimizations
            self.log_func(f"\n🎮 PHASE 5: Gaming Content Optimization")
            optimized_effects = self._apply_gaming_optimizations(final_effects)
            
            self.log_func(f"\n🎉 AUDIO-ONLY ANALYSIS COMPLETE!")
            self.log_func(f"   - Audio events: {len(audio_events)}")
            self.log_func(f"   - Final effects: {len(optimized_effects)}")
            
            return optimized_effects
            
        except Exception as e:
            self.log_func(f"Error in audio-only analysis: {e}")
            return []

    def create_subtitle_file(self, input_path: str, output_path: str, 
                           animation_type: str = "Random") -> Tuple[bool, List[Dict]]:
        """
        Create subtitle file with onomatopoeia effects.
        
        Args:
            input_path: Input video/audio file
            output_path: Output subtitle file (.srt or .ass)
            animation_type: Animation type for effects
            
        Returns:
            Tuple of (success, events_list)
        """
        try:
            # Analyze file
            events = self.analyze_file(input_path)
            
            if not events:
                self.log_func("No onomatopoeia events detected")
                return False, []
            
            # Create output file
            file_ext = os.path.splitext(output_path)[1].lower()
            
            if file_ext == '.ass' or animation_type != "Static":
                # Create animated ASS file
                ass_path = os.path.splitext(output_path)[0] + '.ass'
                success = self._create_animated_subtitle_file(events, ass_path, animation_type)
                return success, events
            else:
                # Create static SRT file
                success = self._create_static_subtitle_file(events, output_path)
                return success, events
                
        except Exception as e:
            self.log_func(f"Error creating subtitle file: {e}")
            return False, []

    def _create_animated_subtitle_file(self, events: List[Dict], output_path: str, 
                                     animation_type: str) -> bool:
        """Create animated ASS subtitle file."""
        try:
            from animations.core import OnomatopoeiaAnimator
            
            animator = OnomatopoeiaAnimator()
            animated_content = animator.generate_animated_ass_content(events, animation_type)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(animated_content)
            
            self.log_func(f"✅ Animated subtitle file created: {output_path}")
            return True
            
        except Exception as e:
            self.log_func(f"Error creating animated subtitle file: {e}")
            return False

    def _create_static_subtitle_file(self, events: List[Dict], output_path: str) -> bool:
        """Create static SRT subtitle file."""
        try:
            srt_content = self._generate_srt_content(events)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            self.log_func(f"✅ Static subtitle file created: {output_path}")
            return True
            
        except Exception as e:
            self.log_func(f"Error creating static subtitle file: {e}")
            return False

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video for analysis"""
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"extracted_audio_{random.randint(1000,9999)}.wav")
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '48000',
                '-ac', '1',  # Mono
                audio_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.log_func(f"✅ Audio extracted: {audio_path}")
            return audio_path
            
        except Exception as e:
            self.log_func(f"Failed to extract audio: {e}")
            raise

    def _enhance_audio_events(self, audio_events: List[Dict], audio_path: str) -> List[Dict]:
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

    def _apply_gaming_optimizations(self, effects: List[Dict]) -> List[Dict]:
        """Apply gaming-specific optimizations to effect list"""
        if not effects:
            return effects
        
        self.log_func(f"🎮 Applying gaming optimizations to {len(effects)} effects...")
        
        # Sort by time
        effects.sort(key=lambda x: x['start_time'])
        
        # Apply density management
        optimized_effects = self._manage_effect_density(effects)
        
        # Apply minimum spacing
        spaced_effects = self._apply_minimum_spacing(optimized_effects)
        
        # Final priority filtering
        final_effects = self._priority_filtering(spaced_effects)
        
        self.log_func(f"✅ Gaming optimization complete: {len(final_effects)} final effects")
        return final_effects

    def _manage_effect_density(self, effects: List[Dict]) -> List[Dict]:
        """Manage effect density - max effects per time window"""
        if not effects:
            return effects
        
        # Group effects into 1-minute buckets
        buckets = {}
        for effect in effects:
            bucket = int(effect['start_time'] // 60)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(effect)
        
        # Keep top N effects per bucket based on confidence
        filtered_effects = []
        for bucket_effects in buckets.values():
            # Sort by confidence
            bucket_effects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            # Keep top N
            filtered_effects.extend(bucket_effects[:self.max_effects_per_minute])
        
        return sorted(filtered_effects, key=lambda x: x['start_time'])

    def _apply_minimum_spacing(self, effects: List[Dict]) -> List[Dict]:
        """Ensure minimum spacing between effects"""
        if not effects:
            return effects
        
        spaced_effects = [effects[0]]  # Keep first effect
        
        for effect in effects[1:]:
            last_effect = spaced_effects[-1]
            time_diff = effect['start_time'] - last_effect['start_time']
            
            if time_diff >= self.min_effect_spacing:
                spaced_effects.append(effect)
            else:
                # Too close - keep higher confidence effect
                if effect.get('confidence', 0) > last_effect.get('confidence', 0):
                    spaced_effects[-1] = effect
        
        return spaced_effects

    def _priority_filtering(self, effects: List[Dict]) -> List[Dict]:
        """Final priority filtering based on gaming content importance"""
        priority_effects = []
        
        for effect in effects:
            context = effect.get('context', '')
            confidence = effect.get('confidence', 0)
            
            # Boost certain contexts
            if any(key in context.lower() for key in ['explosion', 'attack', 'damage', 'monster']):
                confidence *= 1.2
            elif 'quiet' in context.lower():
                confidence *= 0.5
            
            # Apply boosted confidence
            effect['final_confidence'] = min(confidence, 1.0)
            
            # Keep effects above threshold
            if effect['final_confidence'] > 0.4:
                priority_effects.append(effect)
        
        return priority_effects

    def _generate_srt_content(self, events: List[Dict]) -> str:
        """Generate SRT content from events"""
        if not events:
            return ""
        
        srt_lines = []
        for i, event in enumerate(events, 1):
            start_time = event['start_time']
            end_time = event['end_time']
            word = event['word']
            
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            srt_lines.extend([
                str(i),
                f"{start_formatted} --> {end_formatted}",
                word,
                ""
            ])
        
        return "\n".join(srt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# Convenient function for backward compatibility in existing code
def create_onomatopoeia_srt(input_path: str, output_path: str, 
                           sensitivity: float = 0.5, animation_type: str = "Random",
                           log_func=None) -> Tuple[bool, List[Dict]]:
    """
    Convenience function to create onomatopoeia subtitle file.
    """
    detector = OnomatopoeiaDetector(sensitivity=sensitivity, log_func=log_func)
    return detector.create_subtitle_file(input_path, output_path, animation_type)


# Backward compatibility alias
CompleteMultimodalDetector = OnomatopoeiaDetector


def test_detector():
    """Test the cleaned detector system"""
    print("Testing cleaned onomatopoeia detector...")
    
    try:
        detector = OnomatopoeiaDetector(log_func=print)
        print("✅ Detector initialized successfully!")
        print("✅ Ready for multimodal onomatopoeia detection!")
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False


if __name__ == "__main__":
    test_detector()