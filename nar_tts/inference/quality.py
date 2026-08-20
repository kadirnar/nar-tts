"""Adaptive Best-of-N scoring and hard quality gates for synthesis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from nar_tts.evaluation.metrics import WaveformMetrics, analyze_waveform
from nar_tts.evaluation.verifiers import VerificationResult


@dataclass(frozen=True)
class QualityGateConfig:
    maximum_cer: float = 0.12
    minimum_technical_quality: float = 0.55
    maximum_repetition_similarity: float = 0.995
    minimum_duration_ratio: float = 0.55
    maximum_duration_ratio: float = 1.80
    minimum_speaker_similarity: float = 0.45
    minimum_emotion_confidence: float = 0.35
    minimum_event_f1: float = 0.50
    require_asr: bool = True
    require_speaker: bool = False
    require_emotion: bool = False
    require_events: bool = False
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "intelligibility": 0.50,
            "technical": 0.20,
            "duration": 0.10,
            "speaker": 0.10,
            "emotion": 0.05,
            "event": 0.05,
        }
    )

    def __post_init__(self):
        supported = {
            "intelligibility",
            "technical",
            "duration",
            "speaker",
            "emotion",
            "event",
        }
        unknown = sorted(set(self.weights) - supported)
        if unknown:
            raise ValueError("unknown quality-gate weights: " + ", ".join(unknown))
        if self.maximum_cer < 0:
            raise ValueError("maximum_cer cannot be negative")
        for name in (
            "minimum_technical_quality",
            "minimum_emotion_confidence",
            "minimum_event_f1",
            "maximum_repetition_similarity",
        ):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1 <= self.minimum_speaker_similarity <= 1:
            raise ValueError("minimum_speaker_similarity must be in [-1, 1]")
        if (
            self.minimum_duration_ratio <= 0
            or self.maximum_duration_ratio < self.minimum_duration_ratio
        ):
            raise ValueError("duration ratios must satisfy 0 < minimum <= maximum")
        if any(float(weight) < 0 for weight in self.weights.values()):
            raise ValueError("quality-gate weights cannot be negative")
        if not any(float(weight) > 0 for weight in self.weights.values()):
            raise ValueError("at least one quality-gate weight must be positive")

    @classmethod
    def from_dict(cls, value: Mapping | None) -> QualityGateConfig:
        value = dict(value or {})
        weights = value.pop("weights", None)
        return cls(**value, **({"weights": weights} if weights is not None else {}))


@dataclass
class Candidate:
    request_index: int
    candidate_index: int
    token_ids: list[int]
    waveform: object | None
    valid: bool
    score: float = 0.0
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    waveform_metrics: WaveformMetrics | None = None
    verification: VerificationResult | None = None
    duration_ratio: float | None = None
    audio_path: str | None = None

    def asdict(self) -> dict:
        return {
            "request_index": self.request_index,
            "candidate_index": self.candidate_index,
            "token_ids": self.token_ids,
            "valid": self.valid,
            "score": self.score,
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
            "waveform_metrics": (
                self.waveform_metrics.asdict() if self.waveform_metrics else None
            ),
            "verification": self.verification.asdict() if self.verification else None,
            "duration_ratio": self.duration_ratio,
            "audio_path": self.audio_path,
        }


def estimate_target_duration(
    text: str,
    reference_text: str,
    reference_seconds: float | None,
    *,
    explicit_seconds: float | None = None,
) -> float | None:
    if explicit_seconds is not None and explicit_seconds > 0:
        return float(explicit_seconds)
    if not reference_seconds or not reference_text.strip():
        return None
    target_units = len("".join(str(text).split()))
    reference_units = len("".join(str(reference_text).split()))
    if target_units < 1 or reference_units < 1:
        return None
    return float(reference_seconds * target_units / reference_units)


def _duration_score(ratio: float | None) -> float | None:
    if ratio is None or ratio <= 0:
        return None
    return float(math.exp(-abs(math.log(ratio)) / 0.45))


def score_candidates(
    candidates: Sequence[Candidate],
    *,
    sample_rate: int,
    target_texts: Sequence[str],
    expected_durations: Sequence[float | None],
    verifier=None,
    reference_waves: Sequence | None = None,
    controls: Sequence | None = None,
    gate: QualityGateConfig | Mapping | None = None,
) -> list[Candidate]:
    """Score candidates in one verifier batch and apply transparent thresholds."""
    gate = (
        gate
        if isinstance(gate, QualityGateConfig)
        else QualityGateConfig.from_dict(gate)
    )
    valid = [item for item in candidates if item.valid and item.waveform is not None]
    verification_by_identity = {}
    if valid and verifier is not None:
        results = verifier.verify(
            [item.waveform for item in valid],
            sample_rate,
            [target_texts[item.request_index] for item in valid],
            reference_waves=(
                [reference_waves[item.request_index] for item in valid]
                if reference_waves is not None
                else None
            ),
            controls=(
                [controls[item.request_index] for item in valid]
                if controls is not None
                else None
            ),
        )
        if len(results) != len(valid):
            raise RuntimeError("verifier returned a different number of results")
        verification_by_identity = {
            id(item): result for item, result in zip(valid, results, strict=True)
        }

    for candidate in candidates:
        reasons = []
        if not candidate.valid or candidate.waveform is None:
            candidate.score = 0.0
            candidate.accepted = False
            candidate.rejection_reasons = ["invalid_audio_tokens"]
            continue
        metrics = analyze_waveform(candidate.waveform, sample_rate)
        verification = verification_by_identity.get(id(candidate))
        control = controls[candidate.request_index] if controls is not None else None
        expected = expected_durations[candidate.request_index]
        ratio = (
            metrics.duration_seconds / expected if expected and expected > 0 else None
        )

        if gate.require_asr and (
            verification is None or verification.cer is None
        ):
            reasons.append("asr_missing")
        if gate.require_speaker and (
            verification is None or verification.speaker_similarity is None
        ):
            reasons.append("speaker_missing")
        if (
            gate.require_emotion
            and control is not None
            and getattr(control, "emotion", "neutral") != "neutral"
            and (verification is None or verification.emotion_confidence is None)
        ):
            reasons.append("emotion_missing")
        if (
            gate.require_events
            and control is not None
            and getattr(control, "events", ())
            and (verification is None or verification.event_f1 is None)
        ):
            reasons.append("event_missing")
        if (
            verification
            and verification.cer is not None
            and verification.cer > gate.maximum_cer
        ):
            reasons.append("cer")
        if metrics.technical_quality < gate.minimum_technical_quality:
            reasons.append("technical_quality")
        if metrics.repetition_similarity > gate.maximum_repetition_similarity:
            reasons.append("repetition")
        if (
            ratio is not None
            and not gate.minimum_duration_ratio <= ratio <= gate.maximum_duration_ratio
        ):
            reasons.append("duration")
        if (
            verification
            and verification.speaker_similarity is not None
            and verification.speaker_similarity < gate.minimum_speaker_similarity
        ):
            reasons.append("speaker")
        if (
            verification
            and verification.emotion_confidence is not None
            and verification.emotion_confidence < gate.minimum_emotion_confidence
        ):
            reasons.append("emotion")
        if (
            verification
            and verification.event_f1 is not None
            and verification.event_f1 < gate.minimum_event_f1
        ):
            reasons.append("event")

        components = {
            "intelligibility": (
                max(0.0, 1.0 - verification.cer)
                if verification and verification.cer is not None
                else None
            ),
            "technical": metrics.technical_quality,
            "duration": _duration_score(ratio),
            "speaker": verification.speaker_similarity if verification else None,
            "emotion": verification.emotion_confidence if verification else None,
            "event": verification.event_f1 if verification else None,
        }
        active = [
            (float(value), float(gate.weights.get(name, 0.0)))
            for name, value in components.items()
            if value is not None and float(gate.weights.get(name, 0.0)) > 0
        ]
        candidate.score = (
            sum(value * weight for value, weight in active)
            / sum(weight for _, weight in active)
            if active
            else 0.0
        )
        candidate.accepted = not reasons
        candidate.rejection_reasons = reasons
        candidate.waveform_metrics = metrics
        candidate.verification = verification
        candidate.duration_ratio = ratio
    return list(candidates)


def select_winners(
    candidates: Sequence[Candidate], request_count: int
) -> list[Candidate | None]:
    """Prefer passing candidates, then return the highest score per request."""
    winners = []
    for request_index in range(request_count):
        options = [item for item in candidates if item.request_index == request_index]
        if not options:
            winners.append(None)
            continue
        accepted = [item for item in options if item.accepted]
        winners.append(max(accepted or options, key=lambda item: item.score))
    return winners
