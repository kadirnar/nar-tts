"""Lazy, replaceable verifiers used by inference and offline evaluation."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

from nar_tts.evaluation.metrics import (
    event_alignment_score,
    transcript_error_rate,
)


@dataclass(frozen=True)
class VerificationResult:
    transcript: str = ""
    cer: float | None = None
    wer: float | None = None
    speaker_similarity: float | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None
    event_f1: float | None = None
    extra: Mapping = field(default_factory=dict)

    def asdict(self) -> dict:
        return asdict(self)


class IndependentASRVerifier:
    """ASR verifier that stays independent from the Qwen3-ASR GRPO reward."""

    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        *,
        device="auto",
        dtype="auto",
        batch_size: int = 4,
        language: str | None = None,
        revision: str | None = None,
        attn_implementation: str | None = "sdpa",
        compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        if "qwen3-asr" in model.casefold():
            raise ValueError("independent evaluation must not reuse Qwen3-ASR")
        self.model_id = model
        self.device = device
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))
        self.language = language
        self.revision = revision
        self.attn_implementation = attn_implementation
        self.compile = bool(compile)
        self.compile_mode = compile_mode
        self.pipeline = None

    def _load(self):
        if self.pipeline is not None:
            return
        import torch
        from transformers import pipeline

        if self.device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = self.device
        kwargs = {"model": self.model_id, "device": device}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.dtype not in (None, "auto"):
            dtype = getattr(torch, str(self.dtype), None)
            if dtype is None:
                raise ValueError(f"unknown ASR dtype: {self.dtype!r}")
            kwargs["dtype"] = dtype
        if self.attn_implementation:
            kwargs["model_kwargs"] = {"attn_implementation": self.attn_implementation}
        self.pipeline = pipeline("automatic-speech-recognition", **kwargs)
        if self.compile:
            self.pipeline.model.forward = torch.compile(
                self.pipeline.model.forward,
                mode=self.compile_mode,
                fullgraph=True,
            )

    def transcribe(self, waves: Sequence, sample_rate: int) -> list[str]:
        self._load()
        inputs = [{"array": wave, "sampling_rate": int(sample_rate)} for wave in waves]
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language
        kwargs = {"batch_size": self.batch_size}
        if generate_kwargs:
            kwargs["generate_kwargs"] = generate_kwargs
        outputs = self.pipeline(inputs, **kwargs)
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(output.get("text", "")).strip() for output in outputs]

    def verify(
        self,
        waves: Sequence,
        sample_rate: int,
        target_texts: Sequence[str],
        **kwargs,
    ) -> list[VerificationResult]:
        del kwargs
        transcripts = self.transcribe(waves, sample_rate)
        if len(transcripts) != len(target_texts):
            raise RuntimeError("ASR returned a different number of transcripts")
        return [
            VerificationResult(
                transcript=transcript,
                cer=transcript_error_rate(target, transcript, "cer"),
                wer=transcript_error_rate(target, transcript, "wer"),
            )
            for target, transcript in zip(target_texts, transcripts, strict=True)
        ]


class AudioClassificationVerifier:
    """Optional emotion/event classifier with an explicit label mapping."""

    def __init__(
        self,
        model: str,
        *,
        label_map: Mapping[str, str] | None = None,
        device="auto",
        revision: str | None = None,
    ):
        self.model_id = model
        self.label_map = {
            str(key).casefold(): str(value).casefold()
            for key, value in (label_map or {}).items()
        }
        self.device = device
        self.revision = revision
        self.pipeline = None

    def _load(self):
        if self.pipeline is not None:
            return
        import torch
        from transformers import pipeline

        device = (
            0
            if self.device == "auto" and torch.cuda.is_available()
            else -1
            if self.device == "auto"
            else self.device
        )
        kwargs = {"model": self.model_id, "device": device}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        self.pipeline = pipeline("audio-classification", **kwargs)

    def classify(self, waves: Sequence, sample_rate: int) -> list[list[dict]]:
        self._load()
        inputs = [{"array": wave, "sampling_rate": int(sample_rate)} for wave in waves]
        outputs = self.pipeline(inputs, top_k=None)
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        normalized = []
        for predictions in outputs:
            normalized.append(
                [
                    {
                        "label": self.label_map.get(
                            str(item["label"]).casefold(),
                            str(item["label"]).casefold(),
                        ),
                        "score": float(item["score"]),
                    }
                    for item in predictions
                ]
            )
        return normalized


class TransformersSpeakerVerifier:
    """Reference similarity with a lazy Transformers x-vector model."""

    def __init__(
        self,
        model: str = "microsoft/wavlm-base-plus-sv",
        *,
        device="auto",
        dtype="float32",
        revision: str | None = None,
        batch_size: int = 8,
        reference_cache_size: int = 256,
    ):
        self.model_id = model
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.batch_size = max(1, int(batch_size))
        self.reference_cache_size = max(1, int(reference_cache_size))
        self.reference_cache = OrderedDict()
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.processor = AutoFeatureExtractor.from_pretrained(
            self.model_id, revision=self.revision
        )
        kwargs = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.dtype not in (None, "auto"):
            kwargs["dtype"] = getattr(torch, str(self.dtype))
        device = (
            torch.device("cuda")
            if self.device == "auto" and torch.cuda.is_available()
            else torch.device("cpu")
            if self.device == "auto"
            else torch.device(self.device)
        )
        self.model = (
            AutoModelForAudioXVector.from_pretrained(self.model_id, **kwargs)
            .to(device)
            .eval()
        )

    def embeddings(self, waves: Sequence, sample_rate: int):
        import numpy as np
        import torch
        import torch.nn.functional as F

        self._load()
        target_rate = int(self.processor.sampling_rate)
        if int(sample_rate) != target_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(int(sample_rate), target_rate)
            waves = [
                resampler(torch.as_tensor(wave, dtype=torch.float32)).numpy()
                for wave in waves
            ]
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        output = []
        for start in range(0, len(waves), self.batch_size):
            batch = self.processor(
                [
                    np.asarray(wave, dtype=np.float32)
                    for wave in waves[start : start + self.batch_size]
                ],
                sampling_rate=target_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            batch = {
                name: value.to(device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device)
                for name, value in batch.items()
            }
            with torch.inference_mode():
                embeddings = self.model(**batch).embeddings
            output.extend(F.normalize(embeddings.float(), dim=-1).cpu().numpy())
        return output

    def similarities(self, waves, references, sample_rate: int) -> list[float | None]:
        import numpy as np

        present = [index for index, wave in enumerate(references) if wave is not None]
        generated_embeddings = self.embeddings(waves, sample_rate)
        if not present:
            return [None] * len(waves)
        keys = {}
        missing = {}
        for index in present:
            array = np.ascontiguousarray(references[index], dtype=np.float32)
            digest = hashlib.blake2b(array.view(np.uint8), digest_size=16).digest()
            key = (int(sample_rate), array.shape, digest)
            keys[index] = key
            if key not in self.reference_cache:
                missing.setdefault(key, array)
        if missing:
            embeddings = self.embeddings(list(missing.values()), sample_rate)
            for key, embedding in zip(missing, embeddings, strict=True):
                self.reference_cache[key] = embedding
                self.reference_cache.move_to_end(key)
                while len(self.reference_cache) > self.reference_cache_size:
                    self.reference_cache.popitem(last=False)
        values = []
        for index, generated in enumerate(generated_embeddings):
            key = keys.get(index)
            reference = self.reference_cache.get(key)
            if reference is None:
                values.append(None)
            else:
                self.reference_cache.move_to_end(key)
                values.append(float(np.clip(np.dot(generated, reference), -1.0, 1.0)))
        return values


class ESPnetSpeakerVerifier(TransformersSpeakerVerifier):
    """Higher-quality WavLM-Large + ECAPA verifier used by the main recipe."""

    def __init__(
        self,
        model: str = "espnet/voxcelebs12_ecapa_wavlm_joint",
        *,
        device="auto",
        dtype="float32",
        revision: str | None = None,
        batch_size: int = 1,
        reference_cache_size: int = 256,
    ):
        if revision is not None:
            raise ValueError("ESPnet model tags do not accept a Hub revision")
        super().__init__(
            model,
            device=device,
            dtype=dtype,
            revision=None,
            batch_size=batch_size,
            reference_cache_size=reference_cache_size,
        )

    def _load(self):
        if self.model is not None:
            return
        try:
            from espnet2.bin.spk_inference import Speech2Embedding
        except ImportError as error:
            raise ImportError(
                'ESPnet speaker verification needs `pip install -e ".[evaluation]"`'
            ) from error
        import torch

        device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.device == "auto"
            else str(self.device)
        )
        self.model = Speech2Embedding.from_pretrained(
            model_tag=self.model_id,
            device=device,
            dtype=str(self.dtype),
        )

    def embeddings(self, waves: Sequence, sample_rate: int):
        import numpy as np
        import torch
        import torch.nn.functional as F

        self._load()
        if int(sample_rate) != 16_000:
            import torchaudio

            resampler = torchaudio.transforms.Resample(int(sample_rate), 16_000)
            waves = [
                resampler(torch.as_tensor(wave, dtype=torch.float32)).numpy()
                for wave in waves
            ]
        output = []
        with torch.inference_mode():
            for wave in waves:
                embedding = self.model(np.ascontiguousarray(wave, dtype=np.float32))
                embedding = torch.as_tensor(embedding).float().reshape(-1)
                output.append(F.normalize(embedding, dim=0).cpu().numpy())
        return output


class CompositeVerifier:
    """Combine independent ASR, speaker, emotion, and sliding event checks."""

    def __init__(
        self,
        *,
        asr: IndependentASRVerifier | None = None,
        speaker: TransformersSpeakerVerifier | None = None,
        emotion: AudioClassificationVerifier | None = None,
        events: AudioClassificationVerifier | None = None,
        event_window_seconds: float = 1.0,
        event_hop_seconds: float = 0.5,
        event_threshold: float = 0.35,
    ):
        self.asr = asr
        self.speaker = speaker
        self.emotion = emotion
        self.events = events
        self.event_window_seconds = float(event_window_seconds)
        self.event_hop_seconds = float(event_hop_seconds)
        self.event_threshold = float(event_threshold)

    def verify(
        self,
        waves: Sequence,
        sample_rate: int,
        target_texts: Sequence[str],
        *,
        reference_waves: Sequence | None = None,
        controls: Sequence | None = None,
    ) -> list[VerificationResult]:
        results = (
            self.asr.verify(waves, sample_rate, target_texts)
            if self.asr
            else [VerificationResult() for _ in waves]
        )
        if self.speaker:
            references = reference_waves or [None] * len(waves)
            values = self.speaker.similarities(waves, references, sample_rate)
            results = [
                replace(item, speaker_similarity=value)
                for item, value in zip(results, values, strict=True)
            ]
        if self.emotion and controls:
            predictions = self.emotion.classify(waves, sample_rate)
            updated = []
            for result, prediction, control in zip(
                results, predictions, controls, strict=True
            ):
                target = getattr(control, "emotion", "neutral")
                scores = {item["label"]: item["score"] for item in prediction}
                best = (
                    max(prediction, key=lambda item: item["score"])
                    if prediction
                    else None
                )
                updated.append(
                    replace(
                        result,
                        emotion=best["label"] if best else None,
                        emotion_confidence=float(scores.get(target, 0.0)),
                    )
                )
            results = updated
        if self.events and controls:
            results = self._event_results(
                results, waves, sample_rate, target_texts, controls
            )
        return results

    def _event_results(self, results, waves, sample_rate, target_texts, controls):
        window = max(1, round(self.event_window_seconds * sample_rate))
        hop = max(1, round(self.event_hop_seconds * sample_rate))
        output = []
        for result, wave, text, control in zip(
            results, waves, target_texts, controls, strict=True
        ):
            segments, starts = [], []
            for start in range(0, max(1, len(wave) - window + 1), hop):
                segment = wave[start : start + window]
                if len(segment) < window:
                    import numpy as np

                    segment = np.pad(segment, (0, window - len(segment)))
                segments.append(segment)
                starts.append(start / sample_rate)
            predictions = self.events.classify(segments, sample_rate)
            detected = []
            for start, items in zip(starts, predictions, strict=True):
                for item in items:
                    if item["score"] >= self.event_threshold:
                        detected.append({"type": item["label"], "start_seconds": start})
                        break
            expected = []
            word_count = max(1, len(str(text).split()))
            duration = len(wave) / sample_rate
            for event in getattr(control, "events", ()):
                if event.at_seconds is not None:
                    position = event.at_seconds
                elif event.after_word is not None:
                    position = duration * event.after_word / word_count
                else:
                    position = duration / 2
                expected.append({"type": event.type, "start_seconds": position})
            score = event_alignment_score(expected, detected)
            output.append(
                replace(
                    result,
                    event_f1=score.f1,
                    extra={
                        **dict(result.extra),
                        "events": score.asdict(),
                        "detected_events": detected,
                    },
                )
            )
        return output
