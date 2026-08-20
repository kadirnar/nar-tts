"""Encode expressive JSONL data while preserving every control and provenance field."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from nar_tts.core.controls import SpeechControl, TextFrontend, render_controlled_text
from nar_tts.evaluation.data_quality import read_jsonl

METADATA_FIELDS = (
    "id",
    "speaker",
    "source",
    "license",
    "split",
    "style_reference",
    "pronunciation_references",
    "hard_case",
    "teacher_score",
    "teacher_cer",
    "source_report",
)


def _audio_path(row: dict, root: Path) -> Path:
    value = row.get("audio") or row.get("audio_path")
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        raise ValueError("expressive row has no audio path")
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _load_audio(path: Path, target_rate: int):
    import soundfile as sf
    import torch
    import torchaudio

    wave, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    wave = np.ascontiguousarray(wave, dtype=np.float32)
    if not wave.size:
        raise ValueError(f"empty audio: {path}")
    if sample_rate != target_rate:
        wave = (
            torchaudio.functional.resample(
                torch.from_numpy(wave).unsqueeze(0), sample_rate, target_rate
            )
            .squeeze(0)
            .numpy()
        )
    return np.ascontiguousarray(wave, dtype=np.float32)


def encode_expressive_manifest(
    manifest,
    output,
    *,
    tokenizer_name: str,
    checkpoint: str | None = None,
    device: str = "cuda:0",
    codec_model: str = "kyutai/mimi",
    codec_dtype: str | None = "bfloat16",
    batch_size: int = 16,
    tag_neutral: bool = False,
) -> dict:
    """Create an SFT-ready Parquet with ``input_ids`` plus retained metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    from nar_tts.core.audio import MimiCodec
    from nar_tts.core.tokens import TokenLayout

    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    rows = list(read_jsonl(manifest))
    if not rows:
        raise ValueError("expressive manifest is empty")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint or tokenizer_name)
    layout = TokenLayout.from_tokenizer(tokenizer)
    dtype = None if codec_dtype in (None, "auto") else getattr(torch, codec_dtype)
    codec = MimiCodec(
        device,
        model_id=codec_model,
        num_codebooks=layout.num_codebooks,
        dtype=dtype,
    )
    output_rows = []
    for start in range(0, len(rows), max(1, int(batch_size))):
        source_batch = rows[start : start + max(1, int(batch_size))]
        decoded = []
        prepared = []
        for source in source_batch:
            text = str(source.get("text", "")).strip()
            if not text:
                raise ValueError(f"row {start + len(prepared)} has empty text")
            language = str(source.get("language", "tr"))
            frontend = TextFrontend(language)
            lexical_text = frontend.normalize(text)
            control = SpeechControl.from_dict(
                source.get("control")
                or {
                    name: source.get(name)
                    for name in (
                        "emotion",
                        "intensity",
                        "delivery",
                        "valence",
                        "arousal",
                        "events",
                    )
                    if source.get(name) is not None
                }
            )
            model_text = render_controlled_text(
                lexical_text, control, include_neutral=tag_neutral
            )
            path = _audio_path(source, manifest.parent)
            wave = _load_audio(path, codec.sampling_rate)
            decoded.append(wave)
            prepared.append((source, path, lexical_text, model_text, language, control))

        codes, frame_counts = codec.encode(decoded)
        text_ids = tokenizer(
            [item[3] for item in prepared],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        sequences = layout.tts_sequences(text_ids, codes, n_frames=frame_counts)
        for sequence, wave, item in zip(sequences, decoded, prepared, strict=True):
            source, path, lexical_text, model_text, language, control = item
            digest = hashlib.blake2b(path.read_bytes(), digest_size=20).hexdigest()
            metadata = {name: source.get(name) for name in METADATA_FIELDS}
            output_rows.append(
                {
                    "input_ids": sequence,
                    "text": str(source["text"]),
                    "lexical_text": lexical_text,
                    "model_text": model_text,
                    "audio_path": os.fspath(path),
                    "audio_seconds": wave.size / codec.sampling_rate,
                    "audio_hash": digest,
                    "language": language,
                    "emotion": control.emotion,
                    "intensity": control.intensity,
                    "delivery": control.delivery,
                    "valence": control.valence,
                    "arousal": control.arousal,
                    "events": json.dumps(
                        [event.asdict() for event in control.events],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    **{
                        name: (
                            json.dumps(value, ensure_ascii=False, sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for name, value in metadata.items()
                    },
                }
            )

    table = pa.Table.from_pylist(output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=256)
    os.replace(temporary, output)
    counts = {}
    for row in output_rows:
        key = f"{row['emotion']}:{row['delivery']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "rows": len(output_rows),
        "output": os.fspath(output),
        "controls": dict(sorted(counts.items())),
    }
