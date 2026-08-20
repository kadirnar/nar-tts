"""Offline JSONL evaluation with machine-readable and listening-test outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nar_tts.evaluation.data_quality import read_jsonl, write_jsonl_atomic
from nar_tts.evaluation.metrics import (
    aggregate_numeric,
    analyze_waveform,
    transcript_error_rate,
)
from nar_tts.evaluation.verifiers import IndependentASRVerifier


def evaluate_manifest(
    manifest,
    output_path,
    *,
    asr_config: dict | None = None,
    listening_manifest=None,
) -> dict:
    """Evaluate generated WAV files listed in JSONL and write a compact report."""
    import soundfile as sf

    manifest = Path(manifest).resolve()
    rows = list(read_jsonl(manifest))
    verifier = IndependentASRVerifier(**asr_config) if asr_config else None
    loaded = []
    for index, row in enumerate(rows):
        audio = row.get("audio") or row.get("audio_path")
        if not audio:
            raise ValueError(f"row {index} has no audio path")
        audio = Path(audio)
        if not audio.is_absolute():
            audio = manifest.parent / audio
        wave, sample_rate = sf.read(audio, dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        loaded.append((row, audio.resolve(), wave, sample_rate))

    transcripts = [str(item[0].get("hypothesis", "")) for item in loaded]
    if verifier:
        # Grouping by sample rate avoids hidden resampling differences in the ASR pipeline.
        transcripts = [""] * len(loaded)
        by_rate = {}
        for index, item in enumerate(loaded):
            by_rate.setdefault(item[3], []).append(index)
        for sample_rate, indices in by_rate.items():
            values = verifier.transcribe(
                [loaded[index][2] for index in indices], sample_rate
            )
            for index, value in zip(indices, values, strict=True):
                transcripts[index] = value

    items = []
    listening_rows = []
    for index, ((source, audio, wave, sample_rate), transcript) in enumerate(
        zip(loaded, transcripts, strict=True)
    ):
        target = str(source.get("text", source.get("target_text", "")))
        metrics = analyze_waveform(wave, sample_rate).asdict()
        item = {
            "id": source.get("id", index),
            "audio": os.fspath(audio),
            "text": target,
            "transcript": transcript,
            **metrics,
        }
        if transcript and target:
            item["cer"] = transcript_error_rate(target, transcript, "cer")
            item["wer"] = transcript_error_rate(target, transcript, "wer")
        for name in (
            "speaker_similarity",
            "emotion_accuracy",
            "event_f1",
            "speaker_drift",
        ):
            if source.get(name) is not None:
                item[name] = float(source[name])
        items.append(item)
        listening_rows.append(
            {
                "id": item["id"],
                "audio": item["audio"],
                "text": target,
                "naturalness_mos": None,
                "emotion_mos": None,
                "speaker_similarity_mos": None,
                "notes": "",
            }
        )

    report = {
        "manifest": os.fspath(manifest),
        "asr_model": verifier.model_id if verifier else None,
        "count": len(items),
        "summary": aggregate_numeric(items),
        "items": items,
        "notes": {
            "technical_quality": "signal diagnostic; not a learned or human MOS",
            "emotion": "validate classifier results with held-out human listening",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if listening_manifest:
        write_jsonl_atomic(listening_manifest, listening_rows)
    return report
