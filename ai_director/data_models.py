# ai_director/data_models.py
"""
Data models for the AI Director system.

Defines the structure for tasks, responses, and the final decision timeline,
facilitating communication between the Master Director and specialist AIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Literal, Optional
import uuid

# --- Task & Response Models ---

@dataclass
class DirectorTask:
    """A task dispatched by the Master Director to a specialist AI."""
    # --- Fields without defaults come first ---
    task_type: Literal[
        "analyze_text_content",
        "analyze_audio_event",
        "analyze_visual_context"
    ]
    time_range: Tuple[float, float]
    priority: Literal["low", "medium", "high"]

    # --- Fields with defaults follow ---
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, any] = field(default_factory=dict)
    instructions: str = ""

@dataclass
class SpecialistResponse:
    """A response from a specialist AI back to the Master Director."""
    task_id: str
    result: Literal[
        "wild_content_detected",
        "awkward_content_detected",
        "fear_surprise_detected",
        "dramatic_moment_detected",
        "attention_direction_detected",
        "no_significant_event"
    ]
    confidence: float
    recommended_action: Optional[Literal[
        "zoom_to_cam",
        "zoom_to_game",
        "zoom_out"
    ]] = None
    details: Dict[str, any] = field(default_factory=dict)

# --- Decision Timeline Model ---

@dataclass
class TimelineEvent:
    """Represents a single editing decision in the final timeline."""
    timestamp: float
    action: Literal[
        "zoom_to_cam",
        "zoom_to_game",
        "zoom_out",
        "no_action"
    ]
    duration: float
    reason: str
    confidence: float
    source_task_id: Optional[str] = None