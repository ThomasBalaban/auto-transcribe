"""
Title generator for SimpleAutoSubs.

Generates a single YouTube metadata title for a finished short, primed by
the channel's analyzer corpus (synthesis + per-video analysis). The output
is the title text plus a detailed analysis record explaining what reference
shorts and patterns shaped the choice.

Strict no-fallback policy (Phase 1.3 of gameplan.md):
    - Analyzer unreachable, synthesis missing, Gemini failure, or empty
      transcript → return None. Caller writes no title field at all.
"""
import datetime
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from analyzer_client import AnalyzerClient, CHANNEL_HANDLE
from utils.models import (
    MODEL_PRO,
    get_gemini_client,
    get_safety_settings,
)


# Shorts titles are competing for swipe-time on a tiny phone screen. Keep them
# tight — 70 chars is the YouTube short-title display ceiling on mobile.
MAX_TITLE_CHARS = 70

# JSON schema we ask Gemini to fill in. Captured here (not just inline in the
# prompt) so structured-output validation happens server-side.
_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "chosen_title",
        "chosen_title_reasoning",
        "candidates_considered",
        "patterns_applied",
        "patterns_avoided",
        "reference_comparisons",
    ],
    "properties": {
        "chosen_title": {
            "type": "string",
            "description": (
                "The single YouTube metadata title, short and snappy, "
                f"no longer than {MAX_TITLE_CHARS} characters."
            ),
        },
        "chosen_title_reasoning": {
            "type": "string",
            "description": (
                "2–4 sentences explaining why this title was chosen over "
                "the others, citing specific synthesis patterns and "
                "reference shorts."
            ),
        },
        "candidates_considered": {
            "type": "array",
            "description": (
                "Every candidate title that was generated, including the "
                "chosen one. The chosen one MUST appear here with "
                "verdict='chosen'. Aim for 3–5 candidates total."
            ),
            "items": {
                "type": "object",
                "required": ["text", "verdict", "reasoning"],
                "properties": {
                    "text": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["chosen", "rejected"],
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Why this candidate was chosen or rejected, "
                            "in concrete terms (which pattern, which "
                            "reference)."
                        ),
                    },
                },
            },
        },
        "patterns_applied": {
            "type": "array",
            "description": (
                "Synthesis-derived patterns this title leans into. "
                "Each entry: 'pattern_name — concrete reason it fits this clip'."
            ),
            "items": {"type": "string"},
        },
        "patterns_avoided": {
            "type": "array",
            "description": (
                "Synthesis-derived anti-patterns explicitly avoided. "
                "Each entry: 'pattern_name — why we avoided it'."
            ),
            "items": {"type": "string"},
        },
        "reference_comparisons": {
            "type": "array",
            "description": (
                "How specific reference shorts (by exact title) "
                "informed the chosen title."
            ),
            "items": {
                "type": "object",
                "required": ["reference_title", "takeaway"],
                "properties": {
                    "reference_title": {"type": "string"},
                    "takeaway": {"type": "string"},
                },
            },
        },
    },
}


class TitleGenerator:
    def __init__(
        self,
        analyzer_client: Optional[AnalyzerClient] = None,
        log_func=print,
    ) -> None:
        self._log = log_func
        self._analyzer = analyzer_client or AnalyzerClient(log_func=log_func)
        self._client = None  # lazy — avoid Gemini-client init if we bail early

    # ── Public entry point ────────────────────────────────────────────────────

    def generate(
        self,
        mic_transcriptions_raw: List[str],
        desktop_transcriptions_raw: List[str],
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a dict on success:
            {
              "text": "<chosen title string>",
              "analysis": { ...detailed reasoning... },
              "provenance": { model, generated_at, ... }
            }
        Returns None on any failure (analyzer down, missing synthesis,
        empty transcript, Gemini failure, malformed response).
        """
        # ── Strict-fail preconditions ──────────────────────────────────────────
        if not self._has_usable_dialogue(
                mic_transcriptions_raw, desktop_transcriptions_raw):
            self._log(
                "[title] no usable dialogue in transcripts — "
                "skipping title generation")
            return None

        synthesis = self._analyzer.read_result(
            f"{CHANNEL_HANDLE}.synthesis.json")
        if not synthesis:
            self._log(
                "[title] analyzer synthesis unavailable — "
                "skipping title generation")
            return None

        per_video = self._analyzer.read_result(f"{CHANNEL_HANDLE}.json")
        if not per_video or not per_video.get("shorts"):
            self._log(
                "[title] analyzer per-video data unavailable — "
                "skipping title generation")
            return None

        # ── Build prompt + call Gemini ─────────────────────────────────────────
        prompt = self._build_prompt(
            synthesis=synthesis,
            shorts=per_video["shorts"],
            mic_transcriptions=mic_transcriptions_raw,
            desktop_transcriptions=desktop_transcriptions_raw,
            original_duration=original_duration,
            final_duration=final_duration,
            trim_segments=trim_segments,
        )

        try:
            if self._client is None:
                self._client = get_gemini_client()
            response = self._client.models.generate_content(
                model=MODEL_PRO,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=get_safety_settings(),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.HIGH,
                    ),
                    response_mime_type="application/json",
                    response_json_schema=_RESPONSE_SCHEMA,
                    temperature=0.7,
                ),
            )
        except Exception as e:
            self._log(f"[title] Gemini call failed: {e}")
            return None

        text = getattr(response, "text", None)
        if not text:
            self._log("[title] Gemini returned empty response")
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self._log(f"[title] could not parse Gemini JSON: {e}")
            return None

        chosen = (data.get("chosen_title") or "").strip()
        if not chosen:
            self._log("[title] Gemini response missing chosen_title")
            return None

        # Post-validate: enforce length cap on the server side too. If the
        # model overshoots, we still take the title but flag it in the analysis
        # so it shows up when reviewing why the model went long.
        too_long = len(chosen) > MAX_TITLE_CHARS

        analysis = {
            "chosen_title_reasoning": data.get("chosen_title_reasoning", ""),
            "candidates_considered": data.get("candidates_considered", []),
            "patterns_applied": data.get("patterns_applied", []),
            "patterns_avoided": data.get("patterns_avoided", []),
            "reference_comparisons": data.get("reference_comparisons", []),
        }
        if too_long:
            analysis["length_warning"] = (
                f"Chosen title is {len(chosen)} chars "
                f"(cap is {MAX_TITLE_CHARS})."
            )

        provenance = {
            "model": MODEL_PRO,
            "generated_at": datetime.datetime.now().isoformat(),
            "channel_handle": CHANNEL_HANDLE,
            "synthesis_total_shorts": (
                synthesis.get("metadata", {}).get("total_shorts")),
            "synthesis_small_corpus_warning": (
                synthesis.get("metadata", {}).get("small_corpus_warning")),
            "synthesis_generated_at": (
                synthesis.get("metadata", {}).get("generated_at")),
        }

        self._log(
            f"[title] chose: \"{chosen}\" "
            f"(considered {len(analysis['candidates_considered'])} candidates)"
        )
        return {
            "text": chosen,
            "analysis": analysis,
            "provenance": provenance,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _has_usable_dialogue(
        mic: List[str], desktop: List[str],
    ) -> bool:
        # Transcriber output is "start-end: word" strings. Strip timestamps
        # and check there's any actual content.
        for line in (mic or []) + (desktop or []):
            text = re.sub(r"^\s*\d+\.\d+-\d+\.\d+:\s*", "", line).strip()
            if text:
                return True
        return False

    @staticmethod
    def _strip_timestamps(lines: List[str]) -> str:
        out = []
        for line in lines:
            text = re.sub(r"^\s*\d+\.\d+-\d+\.\d+:\s*", "", line).strip()
            if text:
                out.append(text)
        return " ".join(out)

    @staticmethod
    def _trim_summary(
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> str:
        parts = []
        if original_duration:
            parts.append(f"raw: {original_duration:.1f}s")
        if final_duration:
            parts.append(f"final: {final_duration:.1f}s")
        if trim_segments:
            seg_str = ", ".join(
                f"{s:.1f}-{e:.1f}" for s, e in trim_segments[:8])
            if len(trim_segments) > 8:
                seg_str += f", … (+{len(trim_segments) - 8} more)"
            parts.append(f"kept: [{seg_str}]")
        return "; ".join(parts) if parts else "duration/trim unknown"

    def _build_prompt(
        self,
        synthesis: Dict[str, Any],
        shorts: List[Dict[str, Any]],
        mic_transcriptions: List[str],
        desktop_transcriptions: List[str],
        original_duration: Optional[float],
        final_duration: Optional[float],
        trim_segments: Optional[List[Tuple[float, float]]],
    ) -> str:
        narrative = synthesis.get("narrative", {}) or {}
        meta = synthesis.get("metadata", {}) or {}

        # Reference block: each of the 5 shorts with title, breakout score,
        # hook, and what made it work. The model uses these as both worked
        # examples and a reverse-engineering target.
        reference_blocks = []
        for s in shorts:
            ga = s.get("gemini_analysis", {}) or {}
            title_obj = ga.get("title", {}) or {}
            hook_obj = ga.get("hook", {}) or {}
            tags_obj = ga.get("tags", {}) or {}
            ref_lines = [
                f"### {s.get('title', '(untitled)')}",
                (f"breakout_score: {s.get('breakout_score')} | "
                 f"views: {s.get('views')} | "
                 f"duration: {s.get('duration_seconds')}s"),
                (f"title_why_it_worked: "
                 f"{title_obj.get('why_it_worked', '(none)')}"),
                f"hook: {hook_obj.get('description', '(none)')}",
                (f"why_the_video_worked: "
                 f"{(ga.get('why_the_video_worked') or '')[:600]}"),
            ]
            tag_summary = []
            for axis in ("title_mechanics", "title_video_relationship",
                         "hook_type"):
                vals = tags_obj.get(axis) or []
                if vals:
                    tag_summary.append(f"{axis}={','.join(vals)}")
            if tag_summary:
                ref_lines.append("relevant_tags: " + " | ".join(tag_summary))
            reference_blocks.append("\n".join(ref_lines))

        synthesis_block = "\n\n".join([
            (f"corpus: {meta.get('total_shorts')} shorts "
             f"(small_corpus_warning="
             f"{bool(meta.get('small_corpus_warning'))})"),
            f"top_quintile_signature:\n{narrative.get('top_quintile_signature', '')}",
            (f"bottom_quintile_signature:\n"
             f"{narrative.get('bottom_quintile_signature', '')}"),
            (f"load_bearing_patterns:\n"
             f"{narrative.get('load_bearing_patterns', '')}"),
            (f"conditional_insights:\n"
             f"{narrative.get('conditional_insights', '')}"),
            f"cautions:\n{narrative.get('cautions', '')}",
        ])

        mic_text = self._strip_timestamps(mic_transcriptions)
        desktop_text = self._strip_timestamps(desktop_transcriptions)
        trim_str = self._trim_summary(
            original_duration, final_duration, trim_segments)

        # Cap transcripts to keep the prompt sane. Mic is the streamer's voice
        # and is the most signal-rich for titling; desktop is game/ambient.
        if len(mic_text) > 3000:
            mic_text = mic_text[:3000] + " …(truncated)"
        if len(desktop_text) > 1500:
            desktop_text = desktop_text[:1500] + " …(truncated)"

        return f"""You are crafting a single YouTube short metadata title for the channel
@{CHANNEL_HANDLE}. Your job is to study what has worked on this channel and
propose a title that maximises the chance of a swipe-stop on the YouTube
shorts feed.

This title is metadata only — it is shown under the video on YouTube. It is
NOT burned into the video. Subtitles, on-screen text, and overlays are
handled elsewhere and must be ignored when crafting the title.

Hard constraints on the title:
- Single line, no line breaks.
- {MAX_TITLE_CHARS} characters or fewer (count carefully — shorts feed
  truncates aggressively).
- Snappy, scroll-stopping, mobile-readable.
- No clickbait deception — the title must be honest about the clip content.
- Do NOT prepend or append "#shorts" — that's added downstream if needed.

# Channel insights — synthesis from the analyzer
{synthesis_block}

# Reference shorts ({len(shorts)}) — what has actually shipped on this channel
Use these as concrete worked examples. Compare the new title against these,
borrow what made them work, and avoid what tanked the bottom-quintile ones.
Cite specific reference titles in your reasoning.

{chr(10).join(reference_blocks)}

# THIS clip — what we are titling
trim_summary: {trim_str}

mic_dialogue (streamer voice — most signal):
{mic_text or '(silent)'}

desktop_dialogue (game/desktop audio — context):
{desktop_text or '(silent)'}

# Your task
1. Identify what is happening in THIS clip from the dialogue and trim summary.
2. Generate 3–5 candidate titles. Each must satisfy the hard constraints.
3. For each candidate, explain in concrete terms which synthesis pattern or
   reference short it leans on.
4. Pick exactly one as the chosen title. Justify the choice by comparing
   against the rejected candidates AND the bottom-quintile anti-patterns.
5. List patterns applied and patterns explicitly avoided.
6. List which reference shorts (by exact title) most influenced the choice
   and what specifically you borrowed.

Return only the JSON object matching the response schema. No prose outside it.
"""
