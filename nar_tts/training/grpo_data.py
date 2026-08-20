"""Dataset adapters for post-pretraining speech GRPO."""

import json
import os
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from nar_tts.core.controls import (
    SpeechControl,
    controls_from_columns,
    render_controlled_text,
    strip_control_markup,
)
from nar_tts.core.generation import parse_audio_completion
from nar_tts.core.tokens import TokenLayout


class GRPODataError(ValueError):
    """Raised when a dataset row cannot be turned into a Nar rollout prompt."""


@dataclass(frozen=True)
class PreparedGRPOExample:
    prompt: list[int]
    target_text: str
    reference_audio_ids: list[int]
    reference_text: str
    target_duration_seconds: float
    language: str
    emotion: str = "neutral"
    intensity: float = 0.0
    delivery: str = "neutral"
    valence: float | None = None
    arousal: float | None = None
    events: tuple[dict, ...] = ()
    hard_case: bool = False

    def asdict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target_text": self.target_text,
            "reference_audio_ids": self.reference_audio_ids,
            "reference_text": self.reference_text,
            "target_duration_seconds": self.target_duration_seconds,
            "language": self.language,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delivery": self.delivery,
            "valence": self.valence,
            "arousal": self.arousal,
            "events": list(self.events),
            "hard_case": self.hard_case,
        }


def _column(row: dict, columns: dict, name: str, default=None):
    column_name = columns.get(name)
    if not column_name:
        return default
    return row.get(column_name, default)


def _text_prompt(text: str, tokenizer, layout: TokenLayout) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise GRPODataError("target text is empty")
    text_ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
    return [layout.soh, *text_ids, layout.eot, layout.eoh, layout.soa, layout.sos]


def _events(value) -> tuple[dict, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise GRPODataError("events column is not valid JSON") from error
    if not isinstance(value, (list, tuple)):
        raise GRPODataError("events must be a JSON/list array")
    return tuple(dict(item) for item in value)


def _control_from_row(row: dict, columns: dict) -> SpeechControl:
    return controls_from_columns(
        emotion=_column(row, columns, "emotion"),
        intensity=_column(row, columns, "intensity"),
        delivery=_column(row, columns, "delivery"),
        valence=_column(row, columns, "valence"),
        arousal=_column(row, columns, "arousal"),
        events=_events(_column(row, columns, "events")),
    )


def _controlled_fields(control: SpeechControl, hard_case=False) -> dict:
    return {
        "emotion": control.emotion,
        "intensity": control.intensity,
        "delivery": control.delivery,
        "valence": control.valence,
        "arousal": control.arousal,
        "events": tuple(event.asdict() for event in control.events),
        "hard_case": bool(hard_case),
    }


def _valid_audio_ids(ids, layout: TokenLayout, field_name: str) -> list[int]:
    audio_ids = [int(token_id) for token_id in (ids or [])]
    if audio_ids and audio_ids[-1] == layout.eos_speech:
        audio_ids.pop()
    parsed = parse_audio_completion([*audio_ids, layout.eos_speech], layout)
    if not parsed.valid:
        raise GRPODataError(
            f"{field_name} is not a complete {layout.num_codebooks}-codebook audio stream"
        )
    return audio_ids


def split_tts_training_sequence(
    input_ids,
    tokenizer,
    layout: TokenLayout,
    frame_rate: float = 12.5,
) -> PreparedGRPOExample:
    """Split a pretraining TTS row into prompt, transcript, and reference audio."""
    ids = [int(token_id) for token_id in input_ids]
    if not ids:
        raise GRPODataError("input_ids is empty")
    try:
        sos_index = ids.index(layout.sos)
        eot_index = ids.index(layout.eot, 0, sos_index)
    except ValueError as exc:
        raise GRPODataError("input_ids is missing the TTS prompt markers") from exc
    if ids[0] != layout.soh:
        raise GRPODataError("input_ids does not begin with SOH")
    if ids[eot_index + 1 : sos_index + 1] != [layout.eoh, layout.soa, layout.sos]:
        raise GRPODataError("input_ids has an unexpected EOT/EOH/SOA/SOS order")

    completion = parse_audio_completion(ids[sos_index + 1 :], layout)
    if not completion.valid:
        raise GRPODataError("input_ids has invalid or unterminated target audio")
    target_text = strip_control_markup(
        tokenizer.decode(ids[1:eot_index], skip_special_tokens=True)
    ).strip()
    if not target_text:
        raise GRPODataError("input_ids decodes to an empty target transcript")
    return PreparedGRPOExample(
        prompt=ids[: sos_index + 1],
        target_text=target_text,
        reference_audio_ids=completion.token_ids,
        reference_text=target_text,
        target_duration_seconds=completion.num_frames / float(frame_rate),
        language="",
    )


def prepare_grpo_example(
    row: dict,
    tokenizer,
    layout: TokenLayout,
    dataset_config: dict,
) -> PreparedGRPOExample:
    """Adapt one configured dataset row to the schema expected by GRPOTrainer."""
    mode = dataset_config.get("mode", "tts_tokens")
    columns = dataset_config.get("columns", {})
    frame_rate = float(dataset_config.get("frame_rate", 12.5))
    language = str(_column(row, columns, "language", "") or "")
    control = _control_from_row(row, columns)
    hard_case = bool(_column(row, columns, "hard_case", False))

    if mode == "tts_tokens":
        input_ids = _column(row, columns, "input_ids")
        if input_ids is None:
            raise GRPODataError("the configured input_ids column is missing")
        example = split_tts_training_sequence(
            input_ids, tokenizer, layout, frame_rate=frame_rate
        )
        target_duration = _column(row, columns, "target_duration_seconds")
        return PreparedGRPOExample(
            prompt=example.prompt,
            target_text=example.target_text,
            reference_audio_ids=example.reference_audio_ids,
            reference_text=example.reference_text,
            target_duration_seconds=(
                float(target_duration)
                if target_duration is not None
                else example.target_duration_seconds
            ),
            language=language,
            **_controlled_fields(control, hard_case),
        )

    target_text = _column(row, columns, "text")
    if not isinstance(target_text, str) or not target_text.strip():
        raise GRPODataError("the configured text column is missing or empty")
    target_text = target_text.strip()
    reference_text = str(_column(row, columns, "reference_text", "") or "").strip()
    target_duration = _column(row, columns, "target_duration_seconds")
    duration = float(target_duration) if target_duration is not None else -1.0

    if mode == "text":
        return PreparedGRPOExample(
            prompt=_text_prompt(
                render_controlled_text(target_text, control), tokenizer, layout
            ),
            target_text=target_text,
            reference_audio_ids=[],
            reference_text=reference_text,
            target_duration_seconds=duration,
            language=language,
            **_controlled_fields(control, hard_case),
        )

    if mode == "prompt_ids":
        prompt = _column(row, columns, "prompt_ids")
        if not prompt:
            raise GRPODataError("the configured prompt_ids column is missing or empty")
        reference_audio = _column(row, columns, "reference_audio_ids", [])
        reference_audio = (
            _valid_audio_ids(reference_audio, layout, "reference_audio_ids")
            if reference_audio
            else []
        )
        return PreparedGRPOExample(
            prompt=[int(token_id) for token_id in prompt],
            target_text=target_text,
            reference_audio_ids=reference_audio,
            reference_text=reference_text,
            target_duration_seconds=duration,
            language=language,
            **_controlled_fields(control, hard_case),
        )

    if mode == "voice_clone_tokens":
        reference_audio = _valid_audio_ids(
            _column(row, columns, "reference_audio_ids"), layout, "reference_audio_ids"
        )
        if not reference_text:
            raise GRPODataError("voice_clone_tokens requires reference_text")
        combined_text = f"{reference_text} {render_controlled_text(target_text, control)}".strip()
        prompt = _text_prompt(combined_text, tokenizer, layout)
        return PreparedGRPOExample(
            prompt=[*prompt, *reference_audio],
            target_text=target_text,
            reference_audio_ids=reference_audio,
            reference_text=reference_text,
            target_duration_seconds=duration,
            language=language,
            **_controlled_fields(control, hard_case),
        )

    raise GRPODataError(
        f"unknown dataset mode {mode!r}; expected text, tts_tokens, "
        "prompt_ids, or voice_clone_tokens"
    )


def _load_source(config: dict):
    path = config.get("path")
    if not path:
        raise GRPODataError("dataset.path must be set")
    kwargs = {
        "split": config.get("split", "train"),
        "streaming": bool(config.get("streaming", False)),
    }
    for name in ("data_files", "revision", "token", "trust_remote_code"):
        if config.get(name) is not None:
            kwargs[name] = config[name]
    dataset_name = config.get("name")
    if dataset_name:
        return load_dataset(path, dataset_name, **kwargs)
    return load_dataset(path, **kwargs)


def load_grpo_dataset(config: dict, tokenizer, layout: TokenLayout):
    """Load, optionally stream, and adapt a dataset without materializing audio."""
    dataset = _load_source(config)
    hard_case_column = config.get("columns", {}).get("hard_case")
    hard_case_multiplier = int(config.get("hard_case_multiplier", 1))
    if hard_case_multiplier < 1:
        raise GRPODataError("dataset.hard_case_multiplier must be at least 1")
    # Project away raw audio and every unrelated column before iteration. Audio
    # features can otherwise trigger expensive decode work even in text-only
    # streaming GRPO.
    source_columns = sorted(
        {column for column in config.get("columns", {}).values() if column}
    )
    available_columns = getattr(dataset, "column_names", None)
    if source_columns:
        if available_columns:
            required_names = {
                config.get("columns", {}).get(name)
                for name in {
                    "tts_tokens": ("input_ids",),
                    "text": ("text",),
                    "prompt_ids": ("prompt_ids", "text"),
                    "voice_clone_tokens": (
                        "text",
                        "reference_text",
                        "reference_audio_ids",
                    ),
                }[config.get("mode", "tts_tokens")]
            }
            missing = sorted(required_names - set(available_columns))
            if missing:
                raise GRPODataError(
                    "required dataset columns are missing: " + ", ".join(missing)
                )
            source_columns = sorted(set(source_columns) & set(available_columns))
        dataset = dataset.select_columns(source_columns)
    if config.get("shuffle", True):
        seed = int(config.get("seed", 42))
        if isinstance(dataset, IterableDataset):
            dataset = dataset.shuffle(
                seed=seed, buffer_size=int(config.get("shuffle_buffer", 10_000))
            )
        else:
            dataset = dataset.shuffle(seed=seed)

    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise GRPODataError("dataset.max_samples must be positive or null")
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(max_samples)
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    if hard_case_column and hard_case_multiplier > 1:
        if isinstance(dataset, IterableDataset):
            if hard_case_column in (dataset.column_names or []):
                raise GRPODataError(
                    "hard-case oversampling needs a non-streaming dataset"
                )
        elif hard_case_column in dataset.column_names:
            base_indices = list(range(len(dataset)))
            hard_indices = [
                index
                for index, value in enumerate(dataset[hard_case_column])
                if bool(value)
            ]
            dataset = dataset.select(
                base_indices + hard_indices * (hard_case_multiplier - 1)
            )

    drop_invalid = config.get("on_invalid", "error") == "drop"

    def adapt(row):
        try:
            example = prepare_grpo_example(row, tokenizer, layout, config)
            max_prompt_tokens = config.get("max_prompt_tokens")
            if max_prompt_tokens is not None and len(example.prompt) > int(
                max_prompt_tokens
            ):
                raise GRPODataError(
                    f"prompt has {len(example.prompt)} tokens, exceeding "
                    f"dataset.max_prompt_tokens={max_prompt_tokens}"
                )
            output = example.asdict()
            output["_valid"] = True
            return output
        except (GRPODataError, TypeError, ValueError):
            if not drop_invalid:
                raise
            return {
                "prompt": [],
                "target_text": "",
                "reference_audio_ids": [],
                "reference_text": "",
                "target_duration_seconds": -1.0,
                "language": "",
                "emotion": "neutral",
                "intensity": 0.0,
                "delivery": "neutral",
                "valence": None,
                "arousal": None,
                "events": [],
                "hard_case": False,
                "_valid": False,
            }

    map_kwargs = {}
    column_names = getattr(dataset, "column_names", None)
    if column_names:
        map_kwargs["remove_columns"] = column_names
    if isinstance(dataset, Dataset):
        workers = config.get("preprocess_workers", 1)
        if workers == "auto":
            try:
                workers = len(os.sched_getaffinity(0))
            except AttributeError:
                workers = os.cpu_count() or 1
        workers = max(1, int(workers))
        if workers > 1:
            map_kwargs["num_proc"] = workers
    dataset = dataset.map(adapt, **map_kwargs)
    if drop_invalid:
        dataset = dataset.filter(lambda valid: valid, input_columns=["_valid"])
    return dataset.remove_columns(["_valid"])
