"""Quality-first batched voice cloning for Nar TTS."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nar_tts.core.controls import (
    SpeechControl,
    TextFrontend,
    VocalEvent,
    render_controlled_text,
    split_long_text,
)
from nar_tts.inference.quality import (
    Candidate,
    QualityGateConfig,
    estimate_target_duration,
    score_candidates,
    select_winners,
)

DEVICE = "cuda:0"

DEFAULT_INFERENCE_CONFIG = {
    "model": {
        "checkpoint": "checkpoints/latest",
        "tokenizer": None,
        "revision": None,
        "tokenizer_revision": None,
        "device": DEVICE,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    },
    "tokens": {"text_eos_token_id": None, "pad_token_id": None},
    "codec": {
        "model": "kyutai/mimi",
        "revision": None,
        "dtype": "bfloat16",
    },
    "frontend": {
        "language": None,
        "expand_numbers": True,
        "expand_abbreviations": True,
        "lexicon": None,
    },
    "generation": {
        "frame_rate": 12.5,
        "min_audio_seconds": 0.4,
        "max_audio_seconds": 30.0,
        "batch_size": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.1,
        "seed": 42,
    },
    "best_of_n": {"initial": 2, "maximum": 4},
    "quality_gate": {
        "require_asr": True,
        "require_speaker": True,
        "require_emotion": False,
        "require_events": False,
        "maximum_cer": 0.12,
        "minimum_technical_quality": 0.55,
        "maximum_repetition_similarity": 0.995,
        "minimum_duration_ratio": 0.55,
        "maximum_duration_ratio": 1.80,
        "minimum_speaker_similarity": 0.45,
        "minimum_emotion_confidence": 0.35,
        "minimum_event_f1": 0.50,
        "weights": {
            "intelligibility": 0.50,
            "technical": 0.20,
            "duration": 0.10,
            "speaker": 0.10,
            "emotion": 0.05,
            "event": 0.05,
        },
    },
    "verification": {
        "asr": {
            "enabled": True,
            "model": "openai/whisper-large-v3-turbo",
            "revision": None,
            "device": "auto",
            "dtype": "bfloat16",
            "batch_size": 4,
            "language": None,
            "attn_implementation": "sdpa",
            "compile": False,
            "compile_mode": "reduce-overhead",
        },
        "speaker": {
            "enabled": True,
            "backend": "espnet",
            "model": "espnet/voxcelebs12_ecapa_wavlm_joint",
            "revision": None,
            "device": "auto",
            "dtype": "float32",
            "batch_size": 1,
            "reference_cache_size": 256,
        },
        "emotion": {
            "enabled": False,
            "model": None,
            "revision": None,
            "device": "auto",
            "label_map": {},
        },
        "events": {
            "enabled": False,
            "model": None,
            "revision": None,
            "device": "auto",
            "label_map": {},
            "window_seconds": 1.0,
            "hop_seconds": 0.5,
            "threshold": 0.35,
        },
    },
    "acceleration": {
        "kv_cache": True,
        "tf32": True,
        "compile_model": False,
        "compile_codec": False,
        "compile_mode": "reduce-overhead",
    },
    "cache": {"reference_entries": 64},
    "long_form": {
        "max_characters": 260,
        "carry_previous_chunk": True,
        "context_audio_seconds": 4.0,
        "crossfade_milliseconds": 40.0,
    },
    "artifacts": {
        "write_reports": True,
        "save_candidates": True,
        "hard_cases": "infer_out/hard_cases.jsonl",
    },
}


def _merge(base: dict, override: Mapping) -> dict:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _merge(dict(output[key]), value)
        else:
            output[key] = value
    return output


def load_inference_config(path=None) -> dict:
    import yaml

    config = copy.deepcopy(DEFAULT_INFERENCE_CONFIG)
    if path is None:
        config["_config_path"] = None
        return config
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    if not isinstance(override, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    config = _merge(config, override)
    config["_config_path"] = os.fspath(path)
    return config


@dataclass
class SynthesisRequest:
    text: str
    reference_audio: str | os.PathLike | None
    reference_text: str
    output_path: str | os.PathLike | None = None
    id: str | None = None
    language: str = "tr"
    control: SpeechControl | Mapping | None = None
    target_duration_seconds: float | None = None
    seed: int | None = None
    reference_waveform: object | None = field(default=None, repr=False)
    reference_sample_rate: int | None = field(default=None, repr=False)

    def __post_init__(self):
        self.text = str(self.text).strip()
        self.reference_text = str(self.reference_text).strip()
        if not self.text:
            raise ValueError("synthesis text cannot be empty")
        if not self.reference_text:
            raise ValueError("reference_text cannot be empty")
        if self.reference_audio is None and self.reference_waveform is None:
            raise ValueError("reference_audio or reference_waveform is required")
        if self.reference_waveform is not None and not self.reference_sample_rate:
            raise ValueError(
                "reference_sample_rate is required with reference_waveform"
            )
        if not isinstance(self.control, SpeechControl):
            self.control = SpeechControl.from_dict(self.control)
        if self.output_path is not None:
            self.output_path = os.fspath(self.output_path)


@dataclass
class SynthesisResult:
    request: SynthesisRequest
    waveform: object | None
    sample_rate: int
    winner: Candidate | None
    candidates: list[Candidate]
    report_path: str | None = None
    chunks: list[SynthesisResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return bool(self.winner and self.winner.accepted)


@dataclass
class _Reference:
    ids: list[int]
    waveform: np.ndarray
    seconds: float


@dataclass
class _Prepared:
    request: SynthesisRequest
    target_text: str
    reference_text: str
    controlled_text: str
    reference: _Reference
    prompt: list[int]


class NarTTS:
    """Batched voice cloning with controls, caching, verification, and reranking."""

    def __init__(
        self,
        checkpoint=None,
        tokenizer_name=None,
        device=None,
        *,
        config: str | os.PathLike | Mapping | None = None,
        verifier=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from nar_tts.core.audio import MimiCodec
        from nar_tts.core.tokens import TokenLayout

        if config is None:
            settings = load_inference_config()
        elif isinstance(config, Mapping):
            settings = _merge(load_inference_config(), config)
        else:
            settings = load_inference_config(config)
        model_config = settings.setdefault("model", {})
        if checkpoint is not None:
            model_config["checkpoint"] = checkpoint
        if tokenizer_name is not None:
            model_config["tokenizer"] = tokenizer_name
        if device is not None:
            model_config["device"] = device

        self.config = settings
        self.torch = torch
        self.device = model_config.get("device", DEVICE)
        checkpoint = model_config.get("checkpoint")
        if not checkpoint:
            raise ValueError("set model.checkpoint or pass --checkpoint")
        tokenizer_name = model_config.get("tokenizer") or checkpoint
        self.tok = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=model_config.get("tokenizer_revision"),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        self.layout = TokenLayout.from_tokenizer(self.tok)
        token_config = settings.get("tokens", {})
        expected_text_eos = token_config.get("text_eos_token_id")
        if expected_text_eos is not None and int(expected_text_eos) != self.layout.eot:
            raise ValueError(
                "tokens.text_eos_token_id does not match the selected tokenizer"
            )
        expected_pad = token_config.get("pad_token_id")
        if expected_pad is not None:
            expected_pad = int(expected_pad)
            if not 0 <= expected_pad < len(self.tok):
                raise ValueError("tokens.pad_token_id is outside the tokenizer")
            if self.tok.pad_token_id not in (None, expected_pad):
                raise ValueError(
                    "tokens.pad_token_id does not match the selected tokenizer"
                )
            self.tok.pad_token_id = expected_pad
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.layout.eot

        dtype_name = model_config.get("dtype", "bfloat16")
        dtype = (
            None if dtype_name in (None, "auto") else getattr(torch, str(dtype_name))
        )
        model_kwargs = {
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
            "low_cpu_mem_usage": bool(model_config.get("low_cpu_mem_usage", True)),
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        for name in ("revision", "attn_implementation"):
            if model_config.get(name) is not None:
                model_kwargs[name] = model_config[name]
        self.model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
            .to(self.device)
            .eval()
        )
        required_vocabulary = self.layout.base + self.layout.num_added_tokens + 1
        input_vocabulary = self.model.get_input_embeddings().num_embeddings
        output_vocabulary = self.model.get_output_embeddings().weight.shape[0]
        if min(input_vocabulary, output_vocabulary) < required_vocabulary:
            raise ValueError(
                "checkpoint does not contain Nar's complete Mimi token vocabulary: "
                f"need {required_vocabulary}, got input={input_vocabulary}, "
                f"output={output_vocabulary}"
            )
        self.model.config.eos_token_id = self.layout.eos_speech
        self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.config.use_cache = bool(
            settings.get("acceleration", {}).get("kv_cache", True)
        )
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.eos_token_id = self.layout.eos_speech
            self.model.generation_config.pad_token_id = self.tok.pad_token_id
            self.model.generation_config.use_cache = self.model.config.use_cache
        acceleration = settings.get("acceleration", {})
        if torch.cuda.is_available() and acceleration.get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if acceleration.get("compile_model", False):
            self.model.compile(mode=acceleration.get("compile_mode", "reduce-overhead"))

        codec_config = settings.get("codec", {})
        codec_dtype_name = codec_config.get("dtype", dtype_name)
        codec_dtype = (
            None
            if codec_dtype_name in (None, "auto")
            else getattr(torch, str(codec_dtype_name))
        )
        self.codec = MimiCodec(
            self.device,
            model_id=codec_config.get("model", "kyutai/mimi"),
            num_codebooks=self.layout.num_codebooks,
            dtype=codec_dtype,
            revision=codec_config.get("revision"),
            compile_model=bool(acceleration.get("compile_codec", False)),
            compile_mode=acceleration.get("compile_mode", "reduce-overhead"),
            allow_tf32=bool(acceleration.get("tf32", True)),
        )
        self.reference_cache = OrderedDict()
        self._path_digest_cache = OrderedDict()
        self.reference_cache_size = max(
            1, int(settings.get("cache", {}).get("reference_entries", 64))
        )
        self.verifier = verifier if verifier is not None else self._build_verifier()

    def _build_verifier(self):
        from nar_tts.evaluation.verifiers import (
            AudioClassificationVerifier,
            CompositeVerifier,
            ESPnetSpeakerVerifier,
            IndependentASRVerifier,
            TransformersSpeakerVerifier,
        )

        config = self.config.get("verification", {})
        asr_config = dict(config.get("asr", {}))
        asr = None
        if asr_config.pop("enabled", False):
            asr = IndependentASRVerifier(**asr_config)
        speaker_config = dict(config.get("speaker", {}))
        speaker = None
        if speaker_config.pop("enabled", False):
            speaker_backend = speaker_config.pop("backend", "espnet")
            if speaker_backend == "espnet":
                speaker = ESPnetSpeakerVerifier(**speaker_config)
            elif speaker_backend == "transformers_xvector":
                speaker = TransformersSpeakerVerifier(**speaker_config)
            else:
                raise ValueError(
                    "verification.speaker.backend must be espnet or transformers_xvector"
                )
        emotion_config = dict(config.get("emotion", {}))
        emotion = None
        if emotion_config.pop("enabled", False):
            emotion = AudioClassificationVerifier(**emotion_config)
        event_config = dict(config.get("events", {}))
        events = None
        event_options = {}
        if event_config.pop("enabled", False):
            for name in ("window_seconds", "hop_seconds", "threshold"):
                if name in event_config:
                    event_options[f"event_{name}"] = event_config.pop(name)
            events = AudioClassificationVerifier(**event_config)
        gate = QualityGateConfig.from_dict(self.config.get("quality_gate"))
        required = {
            "asr": (gate.require_asr, asr),
            "speaker": (gate.require_speaker, speaker),
            "emotion": (gate.require_emotion, emotion),
            "events": (gate.require_events, events),
        }
        for name, (enabled, verifier) in required.items():
            if enabled and verifier is None:
                raise ValueError(
                    f"quality_gate.require_{name} needs "
                    f"verification.{name}.enabled: true"
                )
        if not any((asr, speaker, emotion, events)):
            return None
        return CompositeVerifier(
            asr=asr,
            speaker=speaker,
            emotion=emotion,
            events=events,
            **event_options,
        )

    def _path_key(self, path) -> tuple:
        path = Path(path).expanduser().resolve()
        stat = path.stat()
        identity = (os.fspath(path), stat.st_size, stat.st_mtime_ns)
        digest = self._path_digest_cache.get(identity)
        if digest is None:
            hasher = hashlib.blake2b(digest_size=16)
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            self._path_digest_cache[identity] = digest
            while len(self._path_digest_cache) > self.reference_cache_size * 2:
                self._path_digest_cache.popitem(last=False)
        else:
            self._path_digest_cache.move_to_end(identity)
        return ("path", *identity, digest)

    @staticmethod
    def _wave_key(wave, sample_rate: int) -> tuple:
        array = np.ascontiguousarray(wave, dtype=np.float32)
        digest = hashlib.blake2b(array.view(np.uint8), digest_size=16).hexdigest()
        return ("wave", int(sample_rate), array.shape, digest)

    def _load_reference_wave(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        if request.reference_waveform is not None:
            wave = np.asarray(request.reference_waveform, dtype=np.float32)
            sample_rate = int(request.reference_sample_rate)
        else:
            import soundfile as sf

            wave, sample_rate = sf.read(
                request.reference_audio, dtype="float32", always_2d=False
            )
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if wave.ndim != 1 or not wave.size:
            raise ValueError("reference audio must contain a non-empty waveform")
        if sample_rate != self.codec.sampling_rate:
            import torchaudio

            tensor = self.torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)
            wave = (
                torchaudio.functional.resample(
                    tensor, sample_rate, self.codec.sampling_rate
                )
                .squeeze(0)
                .numpy()
            )
            sample_rate = self.codec.sampling_rate
        return np.ascontiguousarray(wave, dtype=np.float32), int(sample_rate)

    def _reference_key(self, request: SynthesisRequest) -> tuple:
        if request.reference_waveform is not None:
            return self._wave_key(
                request.reference_waveform, request.reference_sample_rate
            )
        return self._path_key(request.reference_audio)

    def _references(self, requests: Sequence[SynthesisRequest]) -> list[_Reference]:
        keys = [self._reference_key(request) for request in requests]
        missing = []
        seen = set()
        for key, request in zip(keys, requests, strict=True):
            if key not in self.reference_cache and key not in seen:
                seen.add(key)
                wave, _ = self._load_reference_wave(request)
                missing.append((key, wave))
        if missing:
            codes, frame_counts = self.codec.encode([wave for _, wave in missing])
            ids = self.layout.codes_batch_to_ids(codes, n_frames=frame_counts)
            for (key, wave), audio_ids in zip(missing, ids, strict=True):
                self.reference_cache[key] = _Reference(
                    ids=audio_ids,
                    waveform=wave,
                    seconds=wave.size / self.codec.sampling_rate,
                )
                self.reference_cache.move_to_end(key)
                while len(self.reference_cache) > self.reference_cache_size:
                    self.reference_cache.popitem(last=False)
        output = []
        for key in keys:
            reference = self.reference_cache[key]
            self.reference_cache.move_to_end(key)
            output.append(reference)
        return output

    def _frontend(self, language: str) -> TextFrontend:
        config = dict(self.config.get("frontend", {}))
        configured_language = config.pop("language", None)
        lexicon_path = config.pop("lexicon", None)
        lexicon = None
        if lexicon_path:
            lexicon = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        return TextFrontend(
            configured_language or language,
            lexicon=lexicon,
            **config,
        )

    def _prepare(self, requests: Sequence[SynthesisRequest]) -> list[_Prepared]:
        references = self._references(requests)
        prepared = []
        for request, reference in zip(requests, references, strict=True):
            frontend = self._frontend(request.language)
            target_text = frontend.normalize(request.text)
            reference_text = frontend.normalize(request.reference_text)
            controlled = render_controlled_text(target_text, request.control)
            text_ids = self.tok(
                f"{reference_text} {controlled}".strip(),
                add_special_tokens=False,
            ).input_ids
            prompt = [
                self.layout.soh,
                *text_ids,
                self.layout.eot,
                self.layout.eoh,
                self.layout.soa,
                self.layout.sos,
                *reference.ids,
            ]
            prepared.append(
                _Prepared(
                    request=request,
                    target_text=target_text,
                    reference_text=reference_text,
                    controlled_text=controlled,
                    reference=reference,
                    prompt=prompt,
                )
            )
        return prepared

    def _generate(
        self,
        indexed: Sequence[tuple[int, _Prepared]],
        number: int,
        *,
        candidate_offset: int = 0,
    ) -> list[Candidate]:
        from transformers import LogitsProcessorList

        from nar_tts.core.generation import (
            AudioTokenLogitsProcessor,
            parse_audio_completion,
        )

        if number < 1 or not indexed:
            return []
        generation = self.config.get("generation", {})
        frame_rate = float(generation.get("frame_rate", 12.5))
        min_frames = max(
            1,
            int(np.ceil(float(generation.get("min_audio_seconds", 0.4)) * frame_rate)),
        )
        max_frames = max(
            min_frames,
            int(np.ceil(float(generation.get("max_audio_seconds", 30.0)) * frame_rate)),
        )
        batch_size = max(1, int(generation.get("batch_size", len(indexed))))
        candidates = []
        for batch_start in range(0, len(indexed), batch_size):
            batch = indexed[batch_start : batch_start + batch_size]
            width = max(len(item.prompt) for _, item in batch)
            context_limit = getattr(self.model.config, "max_position_embeddings", None)
            completion_tokens = max_frames * self.layout.num_codebooks + 1
            if context_limit and width + completion_tokens > int(context_limit):
                raise ValueError(
                    "prompt plus maximum audio exceeds the model context: "
                    f"{width} + {completion_tokens} > {context_limit}; shorten the "
                    "reference/text or lower generation.max_audio_seconds"
                )
            input_ids = self.torch.full(
                (len(batch), width),
                self.tok.pad_token_id,
                dtype=self.torch.long,
                device=self.device,
            )
            attention_mask = self.torch.zeros_like(input_ids)
            for row, (_, item) in enumerate(batch):
                prompt = self.torch.tensor(
                    item.prompt, dtype=self.torch.long, device=self.device
                )
                input_ids[row, -len(item.prompt) :] = prompt
                attention_mask[row, -len(item.prompt) :] = 1
            grammar = AudioTokenLogitsProcessor(
                self.layout,
                prompt_length=width,
                min_frames=min_frames,
                max_frames=max_frames,
            )
            seed = next(
                (
                    item.request.seed
                    for _, item in batch
                    if item.request.seed is not None
                ),
                generation.get("seed"),
            )
            if seed is not None:
                self.torch.manual_seed(int(seed) + candidate_offset)
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": completion_tokens,
                "do_sample": True,
                "temperature": float(generation.get("temperature", 0.7)),
                "top_p": float(generation.get("top_p", 0.9)),
                "top_k": int(generation.get("top_k", 0)),
                "repetition_penalty": float(generation.get("repetition_penalty", 1.1)),
                "eos_token_id": self.layout.eos_speech,
                "pad_token_id": self.tok.pad_token_id,
                "num_return_sequences": int(number),
                "use_cache": bool(
                    self.config.get("acceleration", {}).get("kv_cache", True)
                ),
                "logits_processor": LogitsProcessorList([grammar]),
            }
            with self.torch.inference_mode():
                sequences = self.model.generate(**kwargs)
            completions = sequences[:, width:].cpu().tolist()
            parsed = [parse_audio_completion(ids, self.layout) for ids in completions]
            batch_candidates = []
            for output_index, item in enumerate(parsed):
                local_request = output_index // number
                request_index = batch[local_request][0]
                batch_candidates.append(
                    Candidate(
                        request_index=request_index,
                        candidate_index=candidate_offset + output_index % number,
                        token_ids=[*item.token_ids, self.layout.eos_speech]
                        if item.terminated
                        else item.token_ids,
                        waveform=None,
                        valid=item.valid,
                    )
                )
            valid = [
                (candidate, item.codes)
                for candidate, item in zip(batch_candidates, parsed, strict=True)
                if item.valid
            ]
            if valid:
                waves = self.codec.decode_batch([codes for _, codes in valid])
                for (candidate, _), wave in zip(valid, waves, strict=True):
                    candidate.waveform = wave
            candidates.extend(batch_candidates)
        return candidates

    def _score(
        self,
        candidates: Sequence[Candidate],
        prepared: Sequence[_Prepared],
    ) -> list[Candidate]:
        target_texts = [item.target_text for item in prepared]
        expected = [
            estimate_target_duration(
                item.target_text,
                item.reference_text,
                item.reference.seconds,
                explicit_seconds=item.request.target_duration_seconds,
            )
            for item in prepared
        ]
        return score_candidates(
            candidates,
            sample_rate=self.codec.sampling_rate,
            target_texts=target_texts,
            expected_durations=expected,
            verifier=self.verifier,
            reference_waves=[item.reference.waveform for item in prepared],
            controls=[item.request.control for item in prepared],
            gate=self.config.get("quality_gate"),
        )

    def _write_results(
        self,
        prepared: Sequence[_Prepared],
        candidates: Sequence[Candidate],
        winners: Sequence[Candidate | None],
        timings: Mapping[str, float] | None = None,
    ) -> list[SynthesisResult]:
        import soundfile as sf

        artifacts = self.config.get("artifacts", {})
        save_candidates = bool(artifacts.get("save_candidates", True))
        write_reports = bool(artifacts.get("write_reports", True))
        hard_case_path = artifacts.get("hard_cases")
        hard_cases = None
        if hard_case_path:
            from nar_tts.evaluation.data_quality import HardCaseStore

            hard_cases = HardCaseStore(hard_case_path)
        results = []
        for index, (item, winner) in enumerate(zip(prepared, winners, strict=True)):
            options = [
                candidate
                for candidate in candidates
                if candidate.request_index == index
            ]
            output_path = (
                Path(item.request.output_path).expanduser().resolve()
                if item.request.output_path
                else None
            )
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path and save_candidates:
                for candidate in options:
                    if candidate.waveform is None:
                        continue
                    candidate_path = output_path.with_name(
                        f"{output_path.stem}.candidate-{candidate.candidate_index:02d}{output_path.suffix or '.wav'}"
                    )
                    sf.write(
                        candidate_path, candidate.waveform, self.codec.sampling_rate
                    )
                    candidate.audio_path = os.fspath(candidate_path)
            if output_path and winner and winner.waveform is not None:
                sf.write(output_path, winner.waveform, self.codec.sampling_rate)
                winner.audio_path = os.fspath(output_path)

            report_path = None
            report = {
                "id": item.request.id
                or (output_path.stem if output_path else str(index)),
                "text": item.request.text,
                "normalized_text": item.target_text,
                "controlled_text": item.controlled_text,
                "reference_audio": (
                    os.fspath(Path(item.request.reference_audio).expanduser().resolve())
                    if item.request.reference_audio is not None
                    else "in-memory"
                ),
                "reference_text": item.request.reference_text,
                "control": item.request.control.asdict(),
                "hard_case": any(
                    candidate.candidate_index
                    >= int(self.config.get("best_of_n", {}).get("initial", 2))
                    for candidate in options
                ),
                "winner": winner.asdict() if winner else None,
                "candidates": [candidate.asdict() for candidate in options],
                "timings": dict(timings or {}),
            }
            if output_path and write_reports:
                report_path = output_path.with_suffix(".json")
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if hard_cases and (winner is None or not winner.accepted):
                hard_cases.append({**report, "failure": "quality_gate"})
            results.append(
                SynthesisResult(
                    request=item.request,
                    waveform=winner.waveform if winner else None,
                    sample_rate=self.codec.sampling_rate,
                    winner=winner,
                    candidates=options,
                    report_path=os.fspath(report_path) if report_path else None,
                    timings=dict(timings or {}),
                )
            )
        return results

    def synthesize_batch(
        self, requests: Sequence[SynthesisRequest | Mapping]
    ) -> list[SynthesisResult]:
        requests = [
            request
            if isinstance(request, SynthesisRequest)
            else SynthesisRequest(**request)
            for request in requests
        ]
        if not requests:
            return []
        started = time.perf_counter()
        prepared = self._prepare(requests)
        prepared_at = time.perf_counter()
        best_of_n = self.config.get("best_of_n", {})
        initial = max(1, int(best_of_n.get("initial", 2)))
        maximum = max(initial, int(best_of_n.get("maximum", 4)))
        candidates = self._generate(list(enumerate(prepared)), initial)
        generated_at = time.perf_counter()
        self._score(candidates, prepared)
        verified_at = time.perf_counter()
        winners = select_winners(candidates, len(prepared))
        difficult = [
            index
            for index, winner in enumerate(winners)
            if winner is None or not winner.accepted
        ]
        if difficult and maximum > initial:
            fallback = self._generate(
                [(index, prepared[index]) for index in difficult],
                maximum - initial,
                candidate_offset=initial,
            )
            self._score(fallback, prepared)
            candidates.extend(fallback)
            winners = select_winners(candidates, len(prepared))
        completed_at = time.perf_counter()
        audio_seconds = sum(
            len(winner.waveform) / self.codec.sampling_rate
            for winner in winners
            if winner is not None and winner.waveform is not None
        )
        timings = {
            "prepare_seconds": prepared_at - started,
            "initial_generation_seconds": generated_at - prepared_at,
            "initial_verification_seconds": verified_at - generated_at,
            "fallback_seconds": completed_at - verified_at,
            "total_seconds": completed_at - started,
            "audio_seconds": audio_seconds,
            "real_time_factor": (completed_at - started) / max(audio_seconds, 1e-9),
        }
        return self._write_results(prepared, candidates, winners, timings=timings)

    def synthesize(self, request: SynthesisRequest | Mapping) -> SynthesisResult:
        return self.synthesize_batch([request])[0]

    def clone(
        self,
        text,
        ref_wav,
        ref_text,
        output_path=None,
        *,
        language="tr",
        emotion="neutral",
        intensity=0.0,
        delivery="neutral",
        events=(),
    ):
        """Backward-compatible API; return the selected mono float32 waveform."""
        result = self.synthesize(
            SynthesisRequest(
                text=text,
                reference_audio=ref_wav,
                reference_text=ref_text,
                output_path=output_path,
                language=language,
                control=SpeechControl(
                    emotion=emotion,
                    intensity=intensity,
                    delivery=delivery,
                    events=tuple(events),
                ),
            )
        )
        return result.waveform

    def synthesize_long(
        self,
        request: SynthesisRequest | Mapping,
        *,
        on_chunk: Callable[[int, SynthesisResult], None] | None = None,
    ) -> SynthesisResult:
        """Stateful sentence-level synthesis with overlap-add boundaries."""
        import soundfile as sf

        request = (
            request
            if isinstance(request, SynthesisRequest)
            else SynthesisRequest(**request)
        )
        config = self.config.get("long_form", {})
        chunks = split_long_text(
            request.text, max_characters=int(config.get("max_characters", 260))
        )
        if len(chunks) <= 1:
            return self.synthesize(request)
        controls = _split_control(request.control, chunks)
        context_seconds = max(0.0, float(config.get("context_audio_seconds", 4.0)))
        stateful = bool(config.get("carry_previous_chunk", True))
        original_wave, original_rate = self._load_reference_wave(request)
        reference_wave = original_wave
        reference_text = request.reference_text
        chunk_results = []
        waves = []
        for index, (text, control) in enumerate(zip(chunks, controls, strict=True)):
            chunk_request = SynthesisRequest(
                text=text,
                reference_audio=None,
                reference_text=reference_text,
                output_path=None,
                id=f"{request.id or 'long'}-{index:03d}",
                language=request.language,
                control=control,
                seed=None if request.seed is None else request.seed + index,
                reference_waveform=reference_wave,
                reference_sample_rate=original_rate,
            )
            result = self.synthesize(chunk_request)
            if result.waveform is None:
                raise RuntimeError(f"long-form chunk {index} produced no valid audio")
            chunk_results.append(result)
            waves.append(np.asarray(result.waveform, dtype=np.float32))
            if on_chunk:
                on_chunk(index, result)
            if stateful and context_seconds > 0:
                tail_samples = int(context_seconds * original_rate)
                tail = waves[-1][-tail_samples:]
                pause = np.zeros(round(0.08 * original_rate), dtype=np.float32)
                reference_wave = np.concatenate((original_wave, pause, tail))
                reference_text = f"{request.reference_text} {text}".strip()
        combined = crossfade_waveforms(
            waves,
            original_rate,
            milliseconds=float(config.get("crossfade_milliseconds", 40.0)),
        )
        if request.output_path:
            output_path = Path(request.output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, combined, original_rate)
        scores = [result.winner.score for result in chunk_results if result.winner]
        winner = Candidate(
            request_index=0,
            candidate_index=0,
            token_ids=[],
            waveform=combined,
            valid=True,
            score=float(sum(scores) / len(scores)) if scores else 0.0,
            accepted=all(result.accepted for result in chunk_results),
            rejection_reasons=sorted(
                {
                    reason
                    for result in chunk_results
                    for reason in (
                        result.winner.rejection_reasons
                        if result.winner
                        else ["missing_chunk"]
                    )
                }
            ),
            audio_path=os.fspath(Path(request.output_path).resolve())
            if request.output_path
            else None,
        )
        compute_seconds = sum(
            result.timings.get("total_seconds", 0.0) for result in chunk_results
        )
        timings = {
            "total_seconds": compute_seconds,
            "audio_seconds": combined.size / original_rate,
            "real_time_factor": compute_seconds
            / max(combined.size / original_rate, 1e-9),
        }
        report_path = None
        if request.output_path and self.config.get("artifacts", {}).get(
            "write_reports", True
        ):
            report_path = Path(request.output_path).resolve().with_suffix(".json")
            report_path.write_text(
                json.dumps(
                    {
                        "id": request.id or report_path.stem,
                        "text": request.text,
                        "reference_audio": (
                            os.fspath(request.reference_audio)
                            if request.reference_audio is not None
                            else "in-memory"
                        ),
                        "reference_text": request.reference_text,
                        "control": request.control.asdict(),
                        "long_form": True,
                        "accepted": winner.accepted,
                        "score": winner.score,
                        "audio_path": winner.audio_path,
                        "timings": timings,
                        "chunks": [
                            {
                                "id": result.request.id,
                                "text": result.request.text,
                                "accepted": result.accepted,
                                "score": result.winner.score if result.winner else None,
                                "rejection_reasons": (
                                    result.winner.rejection_reasons
                                    if result.winner
                                    else ["missing_chunk"]
                                ),
                                "timings": result.timings,
                            }
                            for result in chunk_results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return SynthesisResult(
            request=request,
            waveform=combined,
            sample_rate=original_rate,
            winner=winner,
            candidates=[winner],
            chunks=chunk_results,
            report_path=os.fspath(report_path) if report_path else None,
            timings=timings,
        )


def _split_control(
    control: SpeechControl, chunks: Sequence[str]
) -> list[SpeechControl]:
    counts = [len(re.findall(r"\w+", chunk, flags=re.UNICODE)) for chunk in chunks]
    offsets = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += count
    assigned = [[] for _ in chunks]
    for event in control.events:
        if event.after_word is None:
            assigned[0].append(event)
            continue
        chunk_index = len(chunks) - 1
        for index, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
            if event.after_word <= offset + count:
                chunk_index = index
                break
        assigned[chunk_index].append(
            VocalEvent(
                type=event.type,
                after_word=max(0, event.after_word - offsets[chunk_index]),
                duration=event.duration,
                count=event.count,
            )
        )
    return [
        SpeechControl(
            emotion=control.emotion,
            intensity=control.intensity,
            delivery=control.delivery,
            valence=control.valence,
            arousal=control.arousal,
            events=tuple(events),
        )
        for events in assigned
    ]


def crossfade_waveforms(waves: Sequence, sample_rate: int, milliseconds: float = 40.0):
    """Join chunks with a short equal-amplitude overlap to suppress clicks."""
    arrays = [
        np.asarray(wave, dtype=np.float32).reshape(-1) for wave in waves if len(wave)
    ]
    if not arrays:
        return np.empty(0, dtype=np.float32)
    output = arrays[0].copy()
    requested = max(0, round(sample_rate * milliseconds / 1000.0))
    for current in arrays[1:]:
        overlap = min(requested, output.size, current.size)
        if overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
            fade_in = 1.0 - fade_out
            mixed = output[-overlap:] * fade_out + current[:overlap] * fade_in
            output = np.concatenate((output[:-overlap], mixed, current[overlap:]))
        else:
            output = np.concatenate((output, current))
    return np.ascontiguousarray(output, dtype=np.float32)


def main(argv=None):
    """Keep direct script execution as a convenience; the main command is nar-tts."""
    from nar_tts.cli import main as cli_main

    return cli_main(["infer", *(argv or [])])


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
