"""
Ollama LLM Integration for Onomatopoeia Generation.
Handles connection to Ollama API and text generation for sound effects.
"""

import requests
import json
from typing import Optional, Set


class OllamaLLM:
    """
    Ollama integration for onomatopoeia generation using mistral-nemo.
    """
    
    def __init__(self, model_name="llama3:70b", base_url="http://localhost:11434", log_func=None):
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
    
    def generate_onomatopoeia(self, video_caption: str, audio_context: str, scene_context: Set[str]) -> Optional[str]:
        """Generate onomatopoeia using Ollama with improved prompting."""
        if not self.available:
            return None
            
        try:
            # Build the context string
            full_context = f"**Audio Context:** {audio_context}\n"
            full_context += f"**Video Caption:** \"{video_caption}\"\n"
            if scene_context:
                full_context += f"**Scene Environment:** {', '.join(scene_context)}"

            prompt = f"""You are a master comic book artist. Create the perfect onomatopoeia based on the following context.

**CONTEXT:**
{full_context}

**INSTRUCTIONS:**
1.  **Analyze the Scene:** How does the environment (e.g., underwater) change the sound?
2.  **Determine the Action:** What is the primary action causing the sound?
3.  **Listen to the Audio:** Is the sound sharp, dull, deep, or metallic?
4.  **Create the Word:** Generate a single, powerful onomatopoeia in ALL CAPS.
5.  **Output ONLY the word.**

**Example:**
-   *CONTEXT:* Audio: Sharp, high-frequency. Video: "a soldier fires a rifle". Scene: underwater.
-   *Onomatopoeia:* **BLUB!**

---

**CONTEXT:**
{full_context}
**Onomatopoeia:**"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 10,
                    "stop": ["\n", ".", "!", "?"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_output = result.get('response', '').strip()
                cleaned = self._clean_ollama_output(raw_output)
                
                if cleaned:
                    self.log_func(f"✨ Ollama generated: '{cleaned}' from context: {full_context}")
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
        text = re.sub(r'[^\w\s\-!]', '', text)
        
        # Split into words
        words = text.split()
        
        # Filter out unwanted words
        unwanted = {'THE', 'A', 'AN', 'OF', 'IN', 'ON', 'AT', 'IS', 'ARE', 'WAS', 'WERE', 
                   'SOUND', 'EFFECT', 'NOISE', 'AUDIO', 'ONOMATOPOEIA:'}
        
        # Find the best word(s)
        good_words = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in unwanted and 
                not word.startswith('SOUND') and
                (word.isalpha() or '-' in word or '!' in word)):
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