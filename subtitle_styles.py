"""
Subtitle style definitions for different tracks.
Contains configuration settings for various subtitle styles
that can be applied when embedding subtitles into videos.
"""

# Track 2 (Microphone) - Teal text with darker gray stroke, using BubbleGum font
TRACK2_STYLE = (
    "FontName=BubbleGum,FontSize=16,PrimaryColour=&H00d2ff00,OutlineColour=&H00171717,"
    "BackColour=&H00000000,Bold=1,Italic=0,BorderStyle=1,Outline=3,Shadow=1,"
    "Alignment=2,MarginV=60,MarginL=40,MarginR=40"
)

# Track 3 (Desktop) - Top positioning, white text on semi-transparent background
TRACK3_STYLE = (
    "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "BackColour=&H80000000,Bold=1,Italic=0,BorderStyle=3,Outline=1,Shadow=0,"
    "Alignment=2,MarginV=20,MarginL=40,MarginR=40"
)