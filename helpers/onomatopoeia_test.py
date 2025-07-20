#!/usr/bin/env python3
"""
Simple test script to verify onomatopoeia SRT files work
"""

import os
import subprocess
import tempfile

def create_test_srt():
    """Create a simple test SRT file"""
    srt_content = """1
00:00:02,000 --> 00:00:03,000
BOOM

2
00:00:05,000 --> 00:00:06,000
CRASH

3
00:00:08,000 --> 00:00:09,000
WHUMP
"""
    
    # Create temp SRT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(srt_content)
        return f.name

def test_subtitle_positions(input_video):
    """Test different subtitle positions"""
    srt_file = create_test_srt()
    
    positions_to_test = [
        ("bottom", "MarginV=50"),
        ("low", "MarginV=100"), 
        ("medium", "MarginV=200"),
        ("high", "MarginV=300"),
        ("very_high", "MarginV=400"),
        ("top", "MarginV=500"),
    ]
    
    for name, position in positions_to_test:
        output_file = f"test_onomatopoeia_{name}.mp4"
        
        # Very simple style
        style = f"FontName=Arial,FontSize=48,PrimaryColour=&H0000FFFF,Bold=1,{position}"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f"subtitles='{srt_file}':force_style='{style}'",
            '-t', '10',  # Only first 10 seconds
            output_file
        ]
        
        print(f"Testing position: {name} ({position})")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
        
        print("-" * 50)
    
    # Clean up
    os.unlink(srt_file)
    print("Test complete! Check the test_onomatopoeia_*.mp4 files to see which position works.")

def debug_srt_file(srt_path):
    """Debug an SRT file"""
    print(f"=== DEBUGGING SRT FILE: {srt_path} ===")
    
    if not os.path.exists(srt_path):
        print(f"ERROR: File does not exist: {srt_path}")
        return
    
    file_size = os.path.getsize(srt_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print("ERROR: File is empty!")
        return
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Content:\n{content}")
        
        # Check for common issues
        lines = content.strip().split('\n')
        
        has_numbers = any(line.strip().isdigit() for line in lines)
        has_arrows = any('-->' in line for line in lines)
        has_text = any(line.strip() and not line.strip().isdigit() and '-->' not in line for line in lines)
        
        print(f"Has sequence numbers: {has_numbers}")
        print(f"Has timing arrows: {has_arrows}")
        print(f"Has text content: {has_text}")
        
        if not (has_numbers and has_arrows and has_text):
            print("WARNING: SRT file seems malformed!")
        else:
            print("SRT file appears well-formed")
            
    except Exception as e:
        print(f"ERROR reading file: {e}")
    
    print("=== END DEBUG ===")

def simple_onomatopoeia_test(input_video, srt_file):
    """Test onomatopoeia with the simplest possible approach"""
    output_file = "simple_onomatopoeia_test.mp4"
    
    # Extremely simple style - visible yellow text in middle of screen
    style = "FontName=Arial,FontSize=64,PrimaryColour=&H0000FFFF,Bold=1,MarginV=200"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-vf', f"subtitles='{srt_file}':force_style='{style}'",
        '-t', '15',  # First 15 seconds
        output_file
    ]
    
    print("Running simple onomatopoeia test...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Success! Created: {output_file}")
        print("Check this file - you should see yellow BOOM, CRASH, WHUMP text")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        print(f"stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    # Test with your input video
    input_video = input("Enter path to your input video: ").strip()
    
    if not os.path.exists(input_video):
        print(f"ERROR: Video file not found: {input_video}")
        exit(1)
    
    print("1. Testing simple onomatopoeia...")
    srt_file = create_test_srt()
    debug_srt_file(srt_file)
    
    success = simple_onomatopoeia_test(input_video, srt_file)
    
    if not success:
        print("\n2. Testing multiple positions...")
        test_subtitle_positions(input_video)
    
    os.unlink(srt_file)
    print("\nTest complete!")