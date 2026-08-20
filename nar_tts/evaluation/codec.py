"""Mimi reconstruction audit for expressive and non-verbal audio."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from nar_tts.evaluation.metrics import (
    aggregate_numeric,
    analyze_waveform,
    log_spectral_distance,
    scale_invariant_sdr,
)


def codec_reconstruction_report(
    audio_paths,
    output,
    *,
    device="cuda:0",
    model="kyutai/mimi",
    dtype="bfloat16",
    num_codebooks: int = 32,
    batch_size: int = 8,
) -> dict:
    import soundfile as sf
    import torch
    import torchaudio

    from nar_tts.core.audio import MimiCodec

    torch_dtype = None if dtype in (None, "auto") else getattr(torch, str(dtype))
    codec = MimiCodec(
        device,
        model_id=model,
        num_codebooks=int(num_codebooks),
        dtype=torch_dtype,
    )
    loaded = []
    for value in audio_paths:
        path = Path(value).expanduser().resolve()
        wave, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if sample_rate != codec.sampling_rate:
            wave = (
                torchaudio.functional.resample(
                    torch.from_numpy(wave).unsqueeze(0),
                    sample_rate,
                    codec.sampling_rate,
                )
                .squeeze(0)
                .numpy()
            )
        loaded.append((path, wave))

    items = []
    for start in range(0, len(loaded), max(1, int(batch_size))):
        batch = loaded[start : start + max(1, int(batch_size))]
        codes, frame_counts = codec.encode([wave for _, wave in batch])
        trimmed = [
            codes[index, :, :frames] for index, frames in enumerate(frame_counts)
        ]
        decoded = codec.decode_batch(trimmed)
        for (path, reference), reconstructed in zip(batch, decoded, strict=True):
            original_metrics = analyze_waveform(reference, codec.sampling_rate)
            reconstructed_metrics = analyze_waveform(reconstructed, codec.sampling_rate)
            items.append(
                {
                    "audio": os.fspath(path),
                    "seconds": original_metrics.duration_seconds,
                    "si_sdr_db": scale_invariant_sdr(reference, reconstructed),
                    "log_spectral_distance_db": log_spectral_distance(
                        reference, reconstructed, codec.sampling_rate
                    ),
                    "pitch_median_error_hz": (
                        abs(
                            original_metrics.pitch_median_hz
                            - reconstructed_metrics.pitch_median_hz
                        )
                        if original_metrics.pitch_median_hz
                        and reconstructed_metrics.pitch_median_hz
                        else None
                    ),
                    "technical_quality_before": original_metrics.technical_quality,
                    "technical_quality_after": reconstructed_metrics.technical_quality,
                }
            )
    report = {
        "codec": model,
        "num_codebooks": int(num_codebooks),
        "sample_rate": codec.sampling_rate,
        "count": len(items),
        "summary": aggregate_numeric(items),
        "items": items,
        "note": "Use event/emotion listening tests too; signal metrics cannot prove laugh or cry retention.",
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
