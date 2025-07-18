#!/usr/bin/env python3
"""
YAMNet Model Downloader
This script downloads and caches the YAMNet model for offline use in SimpleAutoSubs.
Run this script once with internet connection to enable onomatopoeia detection.
"""

import datetime
import os
import sys
import ssl
import json
import urllib.request

def install_certificates():
    """Install SSL certificates (especially important on macOS)"""
    print("Installing/updating SSL certificates...")
    
    try:
        # Method 1: Update certifi
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])
        print("✓ Updated certifi package")
        
        # Method 2: Try to install certificates on macOS
        if sys.platform == "darwin":  # macOS
            cert_command = "/Applications/Python\\ 3.*/Install\\ Certificates.command"
            print(f"On macOS, you may also need to run: {cert_command}")
        
        return True
    except Exception as e:
        print(f"Certificate installation failed: {e}")
        return False

def fix_ssl_context():
    """Fix SSL context for downloading"""
    try:
        # Create an unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✓ SSL context configured for downloading")
        return True
    except Exception as e:
        print(f"SSL context fix failed: {e}")
        return False

def setup_tensorflow_environment():
    """Setup TensorFlow and TensorFlow Hub environment"""
    try:
        # Set cache directories
        cache_dir = os.path.expanduser("~/.cache/tensorflow_hub")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        keras_cache = os.path.expanduser("~/.keras")
        os.makedirs(keras_cache, exist_ok=True)
        
        print(f"✓ TensorFlow Hub cache: {cache_dir}")
        print(f"✓ Keras cache: {keras_cache}")
        return True
    except Exception as e:
        print(f"Environment setup failed: {e}")
        return False

def download_yamnet():
    """Download YAMNet model and class map"""
    try:
        print("Importing TensorFlow (this may take a moment)...")
        import tensorflow as tf # type: ignore
        import tensorflow_hub as hub # type: ignore
        
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Download class map first (smaller file)
        print("Downloading YAMNet class map...")
        class_map_path = tf.keras.utils.get_file(
            'yamnet_class_map.csv',
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        print(f"✓ Class map downloaded: {class_map_path}")
        
        # Download the main model
        print("Downloading YAMNet model (this may take several minutes)...")
        print("Please be patient - the model is approximately 13MB...")
        
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("✓ YAMNet model downloaded and cached successfully!")
        
        # Test the model
        print("Testing model...")
        import numpy as np # type: ignore
        
        # Create dummy audio data for testing
        dummy_audio = np.random.random(16000).astype(np.float32)  # 1 second of random audio
        scores, embeddings, spectrogram = model(dummy_audio)
        print(f"✓ Model test successful! Output shape: {scores.shape}")
        
        return True, class_map_path
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install tensorflow tensorflow-hub")
        return False, None
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False, None

def create_config_file(class_map_path):
    """Create configuration file for SimpleAutoSubs"""
    try:
        config = {
            "yamnet_available": True,
            "model_cached": True,
            "class_map_path": class_map_path,
            "cache_directory": os.environ.get('TFHUB_CACHE_DIR'),
            "download_date": str(datetime.now()),
            "status": "ready"
        }
        
        # Save config file in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "yamnet_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved: {config_path}")
        return True
    except Exception as e:
        print(f"Config file creation failed: {e}")
        return False

def main():
    """Main download process"""
    print("="*60)
    print("YAMNET MODEL DOWNLOADER FOR SIMPLEAUTOSUBS")
    print("="*60)
    print("This script will download YAMNet for comic book sound effects.")
    print("You only need to run this once with internet connection.")
    print()
    
    # Step 1: Install certificates
    print("Step 1: Setting up SSL certificates...")
    if not install_certificates():
        print("⚠ Certificate setup had issues, but continuing...")
    
    # Step 2: Fix SSL context
    print("\nStep 2: Configuring SSL context...")
    fix_ssl_context()
    
    # Step 3: Setup environment
    print("\nStep 3: Setting up TensorFlow environment...")
    if not setup_tensorflow_environment():
        print("✗ Environment setup failed")
        return False
    
    # Step 4: Download model
    print("\nStep 4: Downloading YAMNet model...")
    success, class_map_path = download_yamnet()
    
    if success:
        print("\nStep 5: Creating configuration...")
        try:
            from datetime import datetime
            create_config_file(class_map_path)
        except ImportError:
            from datetime import datetime
            create_config_file(class_map_path)
        
        print("\n🎉 SUCCESS! YAMNet is now ready for SimpleAutoSubs!")
        print("\nYou can now:")
        print("1. Enable 'Comic Book Sound Effects' in SimpleAutoSubs")
        print("2. Process videos with onomatopoeia detection")
        print("3. Enjoy comic book style sound effects in your videos!")
        
        return True
    else:
        print("\n❌ DOWNLOAD FAILED")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Try using a VPN or different network")
        print("3. Run: pip install --upgrade certifi")
        if sys.platform == "darwin":
            print("4. macOS: Run /Applications/Python\\ 3.*/Install\\ Certificates.command")
        print("5. Try running this script as administrator/root")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            input("\nPress Enter to exit...")
        else:
            input("\nPress Enter to exit...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)