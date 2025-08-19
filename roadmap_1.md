# Implementation Roadmap: Enhanced Multimodal Onomatopoeia System

## Current State Analysis

### What We Have Now (Working System)

- **CLAP + Ollama Pipeline**: Audio analysis using CLAP for sound captioning, Ollama mistral-nemo for effect generation
- **8 Animation Types**: Drift/fade, wiggle, pop/shrink, shake, pulse, wave, explode-out, hyper bounce
- **ASS Subtitle Generation**: Complex animated effects with per-letter control
- **Energy-Based Filtering**: RMS energy thresholds with deduplication
- **Fixed-Chunk Processing**: 2-second chunks with 0.5-second overlap

### Current Problems

- **Effect Spamming**: Overlapping chunks create duplicate detections for same sound event
- **Context Blindness**: Same audio gets same effect regardless of visual context (underwater punch → gunshot example)
- **Limited Temporal Understanding**: No concept of sustained vs instantaneous sound events
- **No Video Awareness**: Missing dramatic visual moments that lack clear audio

## Target Vision

### What We Want (Enhanced System)

- **Multimodal Analysis**: Video AI + Audio AI working together for context-aware decisions
- **Onset-Based Detection**: Analyze actual sound events, not arbitrary chunks
- **Contextual Effect Selection**: Same "bang" becomes "THWACK!" vs "BANG!" based on visual context
- **Gaming-Optimized**: Handles rapid sequences, overlapping audio, dramatic visual moments
- **Intelligent Fusion**: Video provides context, audio provides timing and energy

## Implementation Phases

### Phase 1: Fix Audio Spamming (Immediate Priority)

**Goal**: Replace chunk-based analysis with onset detection to eliminate duplicate effects

**Implementation**:

```python
# Replace current analyze_audio_file method
def analyze_audio_file_with_onsets(self, audio_path):
    # Use librosa.onset.onset_detect() for precise timing
    # Multi-tier detection (major/medium/quick events)
    # Intelligent deduplication (3-5 second windows)
    # Rapid-fire sequence handling (machine gun → first/last only)
```

**Expected Outcome**: 80-90% reduction in effect spam while maintaining important events

**Files to Modify**:
- `modern_onomatopoeia_detector.py`: Replace chunking with onset detection
- `animations/position_calculator.py`: Update timing calculations

### Phase 2: Add Video Analysis Foundation

**Goal**: Integrate VideoMAE + X-CLIP for general video understanding

**Implementation**:

```python
class VideoAnalyzer:
    def __init__(self):
        self.videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
        self.device = "mps"  # Mac M4 optimization
    
    def analyze_video_chunk(self, video_frames):
        # Extract general visual features (VideoMAE)
        # Classify action type (X-CLIP)
        # Calculate visual drama score
        # Detect temporal patterns
```

**Expected Outcome**: Rich video features available for fusion decisions

**New Files to Create**:
- `video_analyzer.py`: Core video analysis functionality
- `multimodal_fusion.py`: Audio+Video decision making
- `video_utils.py`: Frame extraction, preprocessing utilities

### Phase 3: Multimodal Fusion Engine

**Goal**: Intelligent combination of audio and video analysis for context-aware effects

**Implementation**:

```python
class MultimodalOnomatopoeiaDetector:
    def analyze_gaming_moment(self, audio_features, video_features, timestamp):
        # Calculate audio drama score
        # Calculate visual drama score  
        # Check temporal alignment
        # Make fusion decision with confidence
        # Select contextual effect type
        # Determine timing (audio vs visual peak)
```

**Expected Outcome**: Solves underwater punch problem - effects match visual context

**Files to Modify**:
- `modern_onomatopoeia_detector.py`: Add video integration
- `video_processor.py`: Update to use multimodal detector

### Phase 4: Gaming Content Optimization

**Goal**: Fine-tune for gaming-specific content patterns

**Implementation**:

```python
# Gaming-specific enhancements:
# - Rapid sequence detection (gunfire, explosions)
# - Effect density management (max per time window)
# - Context-aware effect selection (weapon types, environments)
# - Energy-based scaling (louder = bigger effects)
```

**Expected Outcome**: Perfect balance of effects for gaming content - dramatic but not overwhelming

## Technical Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐
│   Audio Track   │    │   Video Track   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Onset Detection │    │ VideoMAE + XCLIP│
│ (Librosa)       │    │ Feature Extract │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ CLAP Captioning │    │ Action Class.   │
│ + Ollama LLM    │    │ Visual Drama    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ Multimodal      │
          │ Fusion Engine   │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Contextual      │
          │ Effect Selection│
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Animation       │
          │ Generation      │
          └─────────────────┘
```

### Data Flow

1. **Input**: Gaming video with audio tracks
2. **Audio Path**: Onset detection → CLAP captioning → Ollama generation
3. **Video Path**: Frame extraction → VideoMAE features → X-CLIP classification
4. **Fusion**: Combine audio+video evidence → Make effect decision
5. **Output**: Contextually appropriate animated onomatopoeia

## Hardware Requirements (Met by M4 + 128GB)

### Model Memory Usage

- **VideoMAE**: ~2GB VRAM
- **X-CLIP**: ~3GB VRAM
- **CLAP**: ~1GB VRAM
- **Ollama mistral-nemo**: ~8GB RAM
- **Total**: ~14GB (well within 128GB capacity)

### Performance Expectations

- **60-second gaming video**: ~20-30 seconds processing
- **Real-time analysis**: Possible for live streaming
- **Batch processing**: Multiple videos simultaneously

## Implementation Checklist

### Phase 1 Tasks

- [ ] Install librosa for onset detection
- [ ] Implement multi-tier onset detection algorithm
- [ ] Add rapid-fire sequence handling
- [ ] Update deduplication logic (3-5 second windows)
- [ ] Test on gaming content, verify spam reduction

### Phase 2 Tasks

- [ ] Install transformers, VideoMAE, X-CLIP models
- [ ] Create video frame extraction pipeline
- [ ] Implement VideoMAE feature extraction
- [ ] Implement X-CLIP action classification
- [ ] Test video analysis on gaming clips

### Phase 3 Tasks

- [ ] Create multimodal fusion engine
- [ ] Implement audio-video temporal alignment
- [ ] Add contextual effect selection logic
- [ ] Update timing decisions (audio vs visual peaks)
- [ ] Test underwater punch scenario

### Phase 4 Tasks

- [ ] Add gaming-specific optimizations
- [ ] Implement effect density management
- [ ] Fine-tune thresholds for gaming content
- [ ] Add user controls for sensitivity/density
- [ ] Performance optimization for M4

## Key Files to Create/Modify

### New Files

- `video_analyzer.py` - VideoMAE + X-CLIP integration
- `multimodal_fusion.py` - Audio+video decision engine  
- `video_utils.py` - Video preprocessing utilities
- `gaming_optimizations.py` - Gaming-specific enhancements

### Modified Files

- `modern_onomatopoeia_detector.py` - Add video integration
- `video_processor.py` - Update to use multimodal
- `ui_components.py` - Add video analysis controls
- `main.py` - Update progress tracking

## Success Metrics

- **Effect spam reduction**: 80-90% fewer duplicate effects
- **Processing speed**: 2-3x real-time on M4
- **Context accuracy**: 90%+ appropriate effect selection

---

**TL;DR**: Transform current audio-only chunking system into intelligent multimodal onset-based system that understands visual context for dramatically better onomatopoeia placement in gaming content.