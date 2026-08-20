"""Transparent lexical, acoustic, event, and speaker-drift metrics."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from nar_tts.core.controls import strip_control_markup


def normalize_transcript(text: str, character_level: bool = False) -> str:
    """Apply deterministic multilingual normalization before WER or CER."""
    text = strip_control_markup(unicodedata.normalize("NFKC", str(text))).casefold()
    normalized = []
    for character in text:
        category = unicodedata.category(character)
        normalized.append(" " if category[0] in {"P", "S", "Z"} else character)
    text = " ".join("".join(normalized).split())
    return "".join(text.split()) if character_level else text


def levenshtein_distance(reference, hypothesis) -> int:
    """Memory-efficient Levenshtein distance for words, characters, or events."""
    reference, hypothesis = list(reference), list(hypothesis)
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def transcript_error_rate(
    reference: str, hypothesis: str, metric: str = "cer"
) -> float:
    """Compute CER or WER with the same normalization on both sides."""
    if metric not in {"cer", "wer"}:
        raise ValueError("metric must be 'cer' or 'wer'")
    character_level = metric == "cer"
    reference = normalize_transcript(reference, character_level=character_level)
    hypothesis = normalize_transcript(hypothesis, character_level=character_level)
    reference_units = list(reference) if character_level else reference.split()
    hypothesis_units = list(hypothesis) if character_level else hypothesis.split()
    if not reference_units:
        return 0.0 if not hypothesis_units else 1.0
    return levenshtein_distance(reference_units, hypothesis_units) / len(
        reference_units
    )


def _frames(wave: np.ndarray, size: int, hop: int) -> np.ndarray:
    if wave.size < size:
        wave = np.pad(wave, (0, size - wave.size))
    count = 1 + max(0, (wave.size - size) // hop)
    shape = (count, size)
    strides = (wave.strides[0] * hop, wave.strides[0])
    return np.lib.stride_tricks.as_strided(wave, shape=shape, strides=strides).copy()


def _safe_db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-10))


def _pitch_track(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_size = max(32, round(sample_rate * 0.040))
    hop = max(16, round(sample_rate * 0.010))
    minimum_lag = max(1, sample_rate // 500)
    maximum_lag = min(frame_size - 2, sample_rate // 60)
    window = np.hanning(frame_size).astype(np.float32)
    pitches = []
    framed = _frames(wave, frame_size, hop)
    # FFT autocorrelation keeps long candidate batches practical on CPU.
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    for start in range(0, len(framed), 512):
        chunk = framed[start : start + 512]
        chunk = (chunk - chunk.mean(axis=1, keepdims=True)) * window
        rms = np.sqrt(np.mean(chunk * chunk, axis=1))
        active = chunk[rms >= 10 ** (-45 / 20)]
        if not len(active):
            continue
        spectrum = np.fft.rfft(active, n=fft_size, axis=1)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=fft_size, axis=1
        )[:, :frame_size]
        candidates = correlation[:, minimum_lag : maximum_lag + 1]
        lags = minimum_lag + np.argmax(candidates, axis=1)
        rows = np.arange(len(active))
        confidence = correlation[rows, lags] / np.maximum(correlation[:, 0], 1e-9)
        pitches.extend((sample_rate / lags[confidence >= 0.30]).tolist())
    return np.asarray(pitches, dtype=np.float32)


def _longest_near_duplicate(wave: np.ndarray, sample_rate: int) -> float:
    """Return the strongest non-adjacent one-second window similarity."""
    size = max(16, int(sample_rate))
    hop = max(8, size // 2)
    if wave.size < size * 2:
        return 0.0
    frames = _frames(wave, size, hop)
    frames -= frames.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    frames = frames / np.maximum(norms, 1e-8)
    similarity = frames @ frames.T
    # Adjacent overlap is expected; only compare windows at least 1.5 s apart.
    exclusion = max(2, math.ceil(1.5 * sample_rate / hop))
    mask = np.triu(np.ones_like(similarity, dtype=bool), k=exclusion)
    return float(np.max(similarity[mask])) if np.any(mask) else 0.0


@dataclass(frozen=True)
class WaveformMetrics:
    duration_seconds: float
    peak: float
    rms_dbfs: float
    dc_offset: float
    clipping_ratio: float
    silence_ratio: float
    voiced_ratio: float
    pitch_median_hz: float | None
    pitch_std_semitones: float | None
    energy_std_db: float
    repetition_similarity: float
    technical_quality: float

    def asdict(self) -> dict:
        return asdict(self)


def analyze_waveform(
    wave, sample_rate: int, *, silence_db: float = -45.0
) -> WaveformMetrics:
    """Compute deterministic diagnostics; ``technical_quality`` is not a MOS."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or not wave.size:
        raise ValueError("waveform must be a non-empty mono or time-major array")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(wave).all():
        raise ValueError("waveform contains NaN or infinity")

    peak = float(np.max(np.abs(wave)))
    rms = float(np.sqrt(np.mean(wave * wave)))
    frame_size = max(16, round(sample_rate * 0.025))
    hop = max(8, round(sample_rate * 0.010))
    framed = _frames(wave, frame_size, hop)
    frame_rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-10))
    silence = frame_db < float(silence_db)
    pitch = _pitch_track(wave, sample_rate)
    pitch_std = None
    if pitch.size > 1:
        pitch_std = float(np.std(12.0 * np.log2(pitch / np.median(pitch))))

    clipping_ratio = float(np.mean(np.abs(wave) >= 0.999))
    silence_ratio = float(np.mean(silence))
    voiced_ratio = float(pitch.size / max(1, framed.shape[0]))
    active_db = frame_db[~silence]
    energy_std = float(np.std(active_db)) if active_db.size else 0.0
    repetition = _longest_near_duplicate(wave, sample_rate)

    clipping_score = math.exp(-1500.0 * clipping_ratio)
    dc_score = math.exp(-40.0 * abs(float(np.mean(wave))))
    level_score = math.exp(-abs(_safe_db(rms) + 20.0) / 18.0)
    silence_score = math.exp(-max(0.0, silence_ratio - 0.65) * 5.0)
    repetition_score = math.exp(-max(0.0, repetition - 0.985) * 80.0)
    technical_quality = float(
        np.clip(
            0.30 * clipping_score
            + 0.15 * dc_score
            + 0.20 * level_score
            + 0.15 * silence_score
            + 0.20 * repetition_score,
            0.0,
            1.0,
        )
    )
    return WaveformMetrics(
        duration_seconds=wave.size / float(sample_rate),
        peak=peak,
        rms_dbfs=_safe_db(rms),
        dc_offset=float(np.mean(wave)),
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        voiced_ratio=voiced_ratio,
        pitch_median_hz=float(np.median(pitch)) if pitch.size else None,
        pitch_std_semitones=pitch_std,
        energy_std_db=energy_std,
        repetition_similarity=repetition,
        technical_quality=technical_quality,
    )


def prosody_similarity(generated: WaveformMetrics, reference: WaveformMetrics) -> float:
    """Compare coarse pitch, energy, voicing, and pause statistics in [0, 1]."""
    scores = [
        math.exp(-abs(generated.energy_std_db - reference.energy_std_db) / 8.0),
        math.exp(-abs(generated.voiced_ratio - reference.voiced_ratio) / 0.35),
        math.exp(-abs(generated.silence_ratio - reference.silence_ratio) / 0.30),
    ]
    if generated.pitch_median_hz and reference.pitch_median_hz:
        semitone_distance = abs(
            12.0 * math.log2(generated.pitch_median_hz / reference.pitch_median_hz)
        )
        scores.append(math.exp(-semitone_distance / 8.0))
    if (
        generated.pitch_std_semitones is not None
        and reference.pitch_std_semitones is not None
    ):
        scores.append(
            math.exp(
                -abs(generated.pitch_std_semitones - reference.pitch_std_semitones)
                / 4.0
            )
        )
    return float(sum(scores) / len(scores))


@dataclass(frozen=True)
class TimedEvent:
    type: str
    start_seconds: float
    end_seconds: float | None = None

    @classmethod
    def from_value(cls, value: Mapping | object) -> TimedEvent:
        if isinstance(value, cls):
            return value
        event_type = str(value.get("type", "")).casefold()
        start = value.get("start_seconds", value.get("at_seconds"))
        if start is None:
            raise ValueError("timed event needs start_seconds or at_seconds")
        end = value.get("end_seconds")
        return cls(event_type, float(start), None if end is None else float(end))


@dataclass(frozen=True)
class EventScore:
    precision: float
    recall: float
    f1: float
    mean_position_error_seconds: float | None
    count_error_rate: float

    def asdict(self) -> dict:
        return asdict(self)


def event_alignment_score(
    expected: Iterable[TimedEvent | Mapping],
    detected: Iterable[TimedEvent | Mapping],
    *,
    position_tolerance_seconds: float = 0.75,
) -> EventScore:
    """Greedily match event type and position, then report F1 and timing error."""
    expected = [TimedEvent.from_value(item) for item in expected]
    detected = [TimedEvent.from_value(item) for item in detected]
    remaining = set(range(len(detected)))
    errors = []
    for target in expected:
        candidates = [
            (abs(target.start_seconds - detected[index].start_seconds), index)
            for index in remaining
            if detected[index].type == target.type
        ]
        if not candidates:
            continue
        error, index = min(candidates)
        if error <= position_tolerance_seconds:
            remaining.remove(index)
            errors.append(error)
    matches = len(errors)
    precision = matches / len(detected) if detected else float(not expected)
    recall = matches / len(expected) if expected else float(not detected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_error = abs(len(detected) - len(expected)) / max(1, len(expected))
    return EventScore(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_position_error_seconds=float(np.mean(errors)) if errors else None,
        count_error_rate=float(count_error),
    )


def cosine_similarity(left, right) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape or not left.size:
        raise ValueError("embeddings must be non-empty and have equal shape")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def speaker_drift_score(reference_embedding, window_embeddings: Sequence) -> dict:
    """Measure identity preservation and instability across overlapping windows."""
    if not window_embeddings:
        return {"score": None, "minimum_similarity": None, "window_std": None}
    similarities = [
        cosine_similarity(reference_embedding, embedding)
        for embedding in window_embeddings
    ]
    adjacent = [
        cosine_similarity(left, right) for left, right in pairwise(window_embeddings)
    ]
    minimum = min(similarities)
    instability = float(np.std(similarities))
    adjacent_mean = float(np.mean(adjacent)) if adjacent else 1.0
    score = float(
        np.clip(
            0.7 * max(0.0, minimum) + 0.3 * max(0.0, adjacent_mean) - instability,
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "minimum_similarity": float(minimum),
        "mean_similarity": float(np.mean(similarities)),
        "window_std": instability,
        "adjacent_similarity": adjacent_mean,
    }


def scale_invariant_sdr(reference, estimate) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    length = min(reference.size, estimate.size)
    if length < 2:
        raise ValueError("SI-SDR needs at least two aligned samples")
    reference = reference[:length] - np.mean(reference[:length])
    estimate = estimate[:length] - np.mean(estimate[:length])
    denominator = float(np.dot(reference, reference))
    if denominator <= 1e-12:
        return float("-inf")
    target = np.dot(estimate, reference) / denominator * reference
    noise = estimate - target
    return float(
        10.0
        * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12))
    )


def log_spectral_distance(reference, estimate, sample_rate: int) -> float:
    """Mean RMS log-magnitude distance over aligned STFT frames."""
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float32).reshape(-1)
    length = min(reference.size, estimate.size)
    frame_size = max(64, round(sample_rate * 0.032))
    # Use a power of two for a stable and efficient FFT.
    frame_size = 1 << (frame_size - 1).bit_length()
    hop = frame_size // 4
    reference_frames = _frames(reference[:length], frame_size, hop)
    estimate_frames = _frames(estimate[:length], frame_size, hop)
    frame_count = min(len(reference_frames), len(estimate_frames))
    window = np.hanning(frame_size).astype(np.float32)
    reference_spectrum = np.abs(
        np.fft.rfft(reference_frames[:frame_count] * window, axis=1)
    )
    estimate_spectrum = np.abs(
        np.fft.rfft(estimate_frames[:frame_count] * window, axis=1)
    )
    difference_db = 20.0 * (
        np.log10(np.maximum(reference_spectrum, 1e-7))
        - np.log10(np.maximum(estimate_spectrum, 1e-7))
    )
    return float(np.mean(np.sqrt(np.mean(difference_db * difference_db, axis=1))))


def aggregate_numeric(rows: Iterable[Mapping]) -> dict[str, dict[str, float]]:
    """Aggregate flat numeric fields without hiding per-item output."""
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in {"id", "index"}:
                continue
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": float(np.mean(items)),
            "median": float(np.median(items)),
            "p95": float(np.percentile(items, 95)),
            "count": len(items),
        }
        for key, items in sorted(values.items())
        if items
    }
