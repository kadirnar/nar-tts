"""Evaluation, data-audit, and hard-case tools for Nar TTS."""

from nar_tts.evaluation.metrics import (
    EventScore,
    WaveformMetrics,
    analyze_waveform,
    event_alignment_score,
    transcript_error_rate,
)

__all__ = [
    "EventScore",
    "WaveformMetrics",
    "analyze_waveform",
    "event_alignment_score",
    "transcript_error_rate",
]
