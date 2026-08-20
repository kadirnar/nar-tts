"""Batched waveform rewards for Nar TTS GRPO post-training."""

import hashlib
import math
import unicodedata
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from packaging.version import Version

from nar_tts.core.audio import MIMI_MODEL_ID, MimiCodec
from nar_tts.core.generation import parse_audio_completion
from nar_tts.core.model_ids import QWEN3_ASR_MODEL_ID, WAVLM_SPEAKER_MODEL_ID
from nar_tts.core.tokens import TokenLayout

QWEN3_ASR_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def normalize_transcript(text: str, character_level: bool = False) -> str:
    """Apply a deterministic multilingual normalization for ASR comparison."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = []
    for character in text:
        category = unicodedata.category(character)
        normalized.append(" " if category[0] in {"P", "S", "Z"} else character)
    text = " ".join("".join(normalized).split())
    return "".join(text.split()) if character_level else text


def levenshtein_distance(reference, hypothesis) -> int:
    """Memory-efficient Levenshtein distance for words or characters."""
    reference, hypothesis = list(reference), list(hypothesis)
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def transcript_error_rate(
    reference: str, hypothesis: str, metric: str = "cer"
) -> float:
    """Compute CER or WER after the same normalization on both strings."""
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


def error_rate_reward(error_rate: float, alpha: float = 3.0) -> float:
    """CER/WER reward from the LLM-TTS-GRPO paper."""
    return float(1.0 - math.tanh(float(alpha) * max(0.0, float(error_rate))))


def nll_reward(nll: float, alpha: float = 3.0) -> float:
    """ASR ground-truth NLL reward from the LLM-TTS-GRPO paper."""
    if alpha <= 0:
        raise ValueError("NLL alpha must be positive")
    return float(math.exp(-max(0.0, float(nll)) / float(alpha)))


def weighted_harmonic_mean(values, weights, epsilon: float = 1e-8) -> float:
    """Weighted harmonic mean that strongly penalizes a weak reward component."""
    pairs = [
        (max(float(value), epsilon), float(weight))
        for value, weight in zip(values, weights)
        if value is not None and float(weight) > 0
    ]
    if not pairs:
        return 0.0
    total_weight = sum(weight for _, weight in pairs)
    return float(total_weight / sum(weight / value for value, weight in pairs))


def duration_consistency_reward(
    generated_seconds: float,
    target_seconds: float,
    mode: str = "binary",
    tolerance=(0.75, 1.25),
    scale: float = 0.35,
) -> float | None:
    """Reward duration consistency using a paper-faithful or smooth objective."""
    if generated_seconds <= 0 or target_seconds <= 0:
        return None
    ratio = generated_seconds / target_seconds
    lower, upper = (float(value) for value in tolerance)
    if not 0 < lower <= upper:
        raise ValueError("duration tolerance must satisfy 0 < lower <= upper")
    if mode == "binary":
        return float(lower <= ratio <= upper)
    if mode == "smooth_log":
        if scale <= 0:
            raise ValueError("duration scale must be positive")
        return float(math.exp(-abs(math.log(ratio)) / scale))
    raise ValueError("duration mode must be 'binary' or 'smooth_log'")


def _dtype(name):
    if name in (None, "auto"):
        return None
    if isinstance(name, torch.dtype):
        return name
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unknown torch dtype: {name!r}")
    return dtype


def _device(name):
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch, device, dtype=None):
    output = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype=dtype)
        output[name] = value
    return output


def _canonical_asr_language(language):
    if language in (None, ""):
        return None
    language = str(language).strip()
    mapped = QWEN3_ASR_LANGUAGE_NAMES.get(language.casefold())
    if mapped is not None:
        return mapped
    for name in QWEN3_ASR_LANGUAGE_NAMES.values():
        if language.casefold() == name.casefold():
            return name
    supported = ", ".join(sorted(QWEN3_ASR_LANGUAGE_NAMES))
    raise ValueError(
        f"Qwen3-ASR does not support language {language!r}; use one of: {supported}"
    )


class SpeechRewardSuite:
    """Decode each rollout once and compute configured speech rewards in batches."""

    def __init__(self, layout: TokenLayout, config: dict):
        self.layout = layout
        self.config = config
        self.frame_rate = float(config.get("frame_rate", 12.5))
        self.invalid_reward = float(config.get("invalid_reward", 0.0))
        self.codec = None
        self.asr_processor = None
        self.asr_model = None
        self.asr_compile_config = None
        self.speaker_processor = None
        self.speaker_model = None
        self.reference_cache = OrderedDict()
        self.resamplers = {}

    @property
    def weights(self):
        configured = self.config.get("weights", {})
        return {
            "intelligibility": float(configured.get("intelligibility", 1.0)),
            "speaker": float(configured.get("speaker", 0.0)),
            "duration": float(configured.get("duration", 0.0)),
            "speed": float(configured.get("speed", 0.0)),
            "format": float(configured.get("format", 0.0)),
        }

    def _get_codec(self):
        if self.codec is None:
            config = self.config.get("codec", {})
            self.codec = MimiCodec(
                device=_device(config.get("device")),
                model_id=config.get("model", MIMI_MODEL_ID),
                num_codebooks=self.layout.num_codebooks,
                dtype=_dtype(config.get("dtype")),
                revision=config.get("revision"),
            )
        return self.codec

    def _resample(self, waves, source_rate: int, target_rate: int):
        if source_rate == target_rate:
            return [np.ascontiguousarray(wave, dtype=np.float32) for wave in waves]
        key = (int(source_rate), int(target_rate))
        resampler = self.resamplers.get(key)
        if resampler is None:
            resampler = self.resamplers[key] = torchaudio.transforms.Resample(*key)
        output = []
        for wave in waves:
            tensor = torch.from_numpy(
                np.ascontiguousarray(wave, dtype=np.float32)
            ).unsqueeze(0)
            resampled = resampler(tensor).squeeze(0)
            output.append(resampled.numpy())
        return output

    def _load_asr(self):
        if self.asr_model is not None:
            return
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        config = self.config.get("asr", {})
        model_id = config.get("model", QWEN3_ASR_MODEL_ID)
        if model_id != QWEN3_ASR_MODEL_ID:
            raise ValueError(
                f"Nar ASR rewards require {QWEN3_ASR_MODEL_ID}, got {model_id!r}"
            )
        if Version(transformers.__version__) < Version("5.13.0"):
            raise ImportError(
                "native Qwen3-ASR requires transformers>=5.13.0; upgrade the "
                "core Nar environment before enabling intelligibility rewards"
            )
        self.asr_processor = AutoProcessor.from_pretrained(
            model_id, revision=config.get("revision")
        )
        model_kwargs = {}
        dtype = _dtype(config.get("dtype", "bfloat16"))
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if config.get("revision") is not None:
            model_kwargs["revision"] = config["revision"]
        self.asr_model = (
            AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
            .to(_device(config.get("device")))
            .eval()
        )
        if config.get("compile", False):
            from transformers import CompileConfig

            self.asr_compile_config = CompileConfig()

    @staticmethod
    def _asr_conversation(wave, target_text, language):
        return [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": wave}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"language {language}<asr_text>{target_text}",
                    }
                ],
            },
        ]

    def _qwen3_asr_nll(self, waves, target_texts, languages, device, dtype):
        values = [None] * len(waves)
        valid_indices = [
            index for index, language in enumerate(languages) if language is not None
        ]
        if not valid_indices:
            return values
        conversations = [
            self._asr_conversation(
                waves[index], target_texts[index], languages[index]
            )
            for index in valid_indices
        ]
        prefix_inputs = self.asr_processor.apply_transcription_request(
            audio=[waves[index] for index in valid_indices],
            language=[languages[index] for index in valid_indices],
        )
        inputs = self.asr_processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"output_labels": True},
        )
        labels = inputs.pop("labels")
        for row in range(labels.shape[0]):
            full_positions = inputs["attention_mask"][row].nonzero(
                as_tuple=False
            ).flatten()
            prefix_positions = prefix_inputs["attention_mask"][row].nonzero(
                as_tuple=False
            ).flatten()
            full_ids = inputs["input_ids"][row, full_positions].tolist()
            prefix_ids = prefix_inputs["input_ids"][row, prefix_positions].tolist()
            common_prefix = 0
            for full_id, prefix_id in zip(full_ids, prefix_ids):
                if full_id != prefix_id:
                    break
                common_prefix += 1
            if common_prefix < 1:
                raise RuntimeError(
                    "Qwen3-ASR full and transcription-prefix templates diverged"
                )
            labels[row, full_positions[:common_prefix]] = -100
        inputs = _move_batch(inputs, device, dtype=dtype)
        labels = labels.to(device)
        with torch.inference_mode():
            outputs = self.asr_model(**inputs, use_cache=False)
        shift_logits = outputs.logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(shift_labels.shape)
        token_nll = token_nll * mask
        if self.config.get("asr", {}).get("nll_reduction", "sum") == "mean":
            reduced = token_nll.sum(-1) / mask.sum(-1).clamp_min(1)
        else:
            reduced = token_nll.sum(-1)
        for index, value in zip(valid_indices, reduced.cpu().tolist(), strict=True):
            values[index] = float(value)
        return values

    def _asr_scores(self, waves, target_texts, languages=None):
        config = self.config.get("asr", {})
        if not config.get("enabled", True):
            return [""] * len(waves), [None] * len(waves)
        self._load_asr()
        feature_extractor = getattr(
            self.asr_processor, "feature_extractor", self.asr_processor
        )
        sample_rate = int(feature_extractor.sampling_rate)
        source_rate = self._get_codec().sampling_rate
        waves = self._resample(waves, source_rate, sample_rate)
        batch_size = max(1, int(config.get("batch_size", len(waves) or 1)))
        transcripts, nll_values = [], []
        need_transcript = config.get("error_weight", 0.6) > 0
        need_nll = config.get("nll_weight", 0.4) > 0
        device = next(self.asr_model.parameters()).device
        dtype = next(self.asr_model.parameters()).dtype
        languages = languages or [None] * len(waves)
        configured_language = config.get("language")
        language_hints = [
            _canonical_asr_language(configured_language or language)
            for language in languages
        ]

        for start in range(0, len(waves), batch_size):
            chunk_waves = waves[start : start + batch_size]
            chunk_texts = target_texts[start : start + batch_size]
            chunk_hints = language_hints[start : start + batch_size]
            must_generate = need_transcript or (
                need_nll and any(language is None for language in chunk_hints)
            )
            if must_generate:
                features = self.asr_processor.apply_transcription_request(
                    audio=chunk_waves,
                    language=chunk_hints,
                )
                prompt_length = features["input_ids"].shape[1]
                features = _move_batch(features, device, dtype=dtype)
                generation_kwargs = {
                    "max_new_tokens": int(config.get("max_new_tokens", 256)),
                    "do_sample": False,
                }
                if self.asr_compile_config is not None:
                    generation_kwargs.update(
                        {
                            "cache_implementation": "static",
                            "compile_config": self.asr_compile_config,
                        }
                    )
                with torch.inference_mode():
                    generated = self.asr_model.generate(**features, **generation_kwargs)
                generated = generated[:, prompt_length:]
                parsed = self.asr_processor.decode(
                    generated, return_format="parsed"
                )
                chunk_transcripts = [item["transcription"] for item in parsed]
                detected_languages = [item["language"] for item in parsed]
            else:
                chunk_transcripts = [""] * len(chunk_waves)
                detected_languages = [None] * len(chunk_waves)
            if need_transcript:
                transcripts.extend(chunk_transcripts)
            else:
                transcripts.extend([""] * len(chunk_waves))
            resolved_languages = [
                hint or _canonical_asr_language(detected)
                for hint, detected in zip(
                    chunk_hints, detected_languages, strict=True
                )
            ]

            if need_nll:
                nll_values.extend(
                    self._qwen3_asr_nll(
                        chunk_waves,
                        chunk_texts,
                        resolved_languages,
                        device,
                        dtype,
                    )
                )
            else:
                nll_values.extend([None] * len(chunk_waves))
        return transcripts, nll_values

    def _load_speaker(self):
        if self.speaker_model is not None:
            return
        config = self.config.get("speaker", {})
        backend = config.get("backend", "espnet")
        model_id = config.get("model", WAVLM_SPEAKER_MODEL_ID)
        if backend == "espnet":
            try:
                from espnet2.bin.spk_inference import Speech2Embedding
            except ImportError as error:
                raise ImportError(
                    "the high-quality speaker reward requires ESPnet, "
                    "espnet-model-zoo, and s3prl; install `nar-tts[speaker-quality]`"
                ) from error
            if model_id != WAVLM_SPEAKER_MODEL_ID:
                raise ValueError(
                    f"the ESPnet WavLM-Large backend requires {WAVLM_SPEAKER_MODEL_ID}"
                )
            dtype = config.get("dtype", "float32")
            self.speaker_model = Speech2Embedding.from_pretrained(
                model_tag=model_id,
                device=str(_device(config.get("device"))),
                dtype=str(dtype),
            )
            return

        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.speaker_processor = AutoFeatureExtractor.from_pretrained(
            model_id, revision=config.get("revision")
        )
        model_kwargs = {}
        dtype = _dtype(config.get("dtype", "float32"))
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if config.get("revision") is not None:
            model_kwargs["revision"] = config["revision"]
        self.speaker_model = (
            AutoModelForAudioXVector.from_pretrained(model_id, **model_kwargs)
            .to(_device(config.get("device")))
            .eval()
        )

    def _speaker_embeddings(self, waves):
        self._load_speaker()
        config = self.config.get("speaker", {})
        if config.get("backend", "espnet") == "espnet":
            source_rate = self._get_codec().sampling_rate
            waves = self._resample(waves, source_rate, 16_000)
            embeddings = []
            with torch.inference_mode():
                for wave in waves:
                    embedding = self.speaker_model(
                        np.ascontiguousarray(wave, dtype=np.float32)
                    )
                    embedding = torch.as_tensor(embedding).float().reshape(-1)
                    embeddings.append(F.normalize(embedding, dim=0).cpu())
            return embeddings
        feature_extractor = getattr(
            self.speaker_processor, "feature_extractor", self.speaker_processor
        )
        sample_rate = int(feature_extractor.sampling_rate)
        source_rate = self._get_codec().sampling_rate
        waves = self._resample(waves, source_rate, sample_rate)
        batch_size = max(1, int(config.get("batch_size", len(waves) or 1)))
        device = next(self.speaker_model.parameters()).device
        dtype = next(self.speaker_model.parameters()).dtype
        embeddings = []
        for start in range(0, len(waves), batch_size):
            features = self.speaker_processor(
                waves[start : start + batch_size],
                sampling_rate=sample_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            features = _move_batch(features, device, dtype=dtype)
            with torch.inference_mode():
                chunk = self.speaker_model(**features).embeddings
            embeddings.extend(F.normalize(chunk.float(), dim=-1).cpu())
        return embeddings

    @staticmethod
    def _audio_key(audio_ids):
        values = np.asarray(audio_ids, dtype=np.int32)
        return hashlib.blake2b(values.tobytes(), digest_size=16).digest()

    def _speaker_scores(self, generated_waves, reference_audio_ids):
        config = self.config.get("speaker", {})
        if not config.get("enabled", False):
            return [None] * len(generated_waves)
        codec = self._get_codec()
        cache_limit = max(0, int(config.get("reference_cache_size", 2048)))
        keys = [self._audio_key(ids) if ids else None for ids in reference_audio_ids]
        if not any(key is not None for key in keys):
            return [None] * len(generated_waves)
        missing = {}
        for key, ids in zip(keys, reference_audio_ids):
            if key is not None and key not in self.reference_cache:
                missing.setdefault(key, ids)
        if missing:
            parsed = [
                parse_audio_completion([*ids, self.layout.eos_speech], self.layout)
                for ids in missing.values()
            ]
            valid_items = [
                (key, item.codes) for key, item in zip(missing, parsed) if item.valid
            ]
            if valid_items:
                waves = codec.decode_batch([codes for _, codes in valid_items])
                embeddings = self._speaker_embeddings(waves)
                for (key, _), embedding in zip(valid_items, embeddings):
                    self.reference_cache[key] = embedding
                    self.reference_cache.move_to_end(key)
                    while len(self.reference_cache) > cache_limit:
                        self.reference_cache.popitem(last=False)

        generated_embeddings = self._speaker_embeddings(generated_waves)
        scores = []
        for generated, key in zip(generated_embeddings, keys):
            reference = self.reference_cache.get(key)
            if reference is None:
                scores.append(None)
            else:
                similarity = F.cosine_similarity(
                    generated.unsqueeze(0), reference.unsqueeze(0)
                ).item()
                scores.append(
                    float(min(1.0, max(float(config.get("minimum", 0.0)), similarity)))
                )
        return scores

    def _expected_duration(
        self,
        explicit_duration,
        reference_ids,
        reference_text,
        target_text,
    ):
        if explicit_duration is not None and float(explicit_duration) > 0:
            return float(explicit_duration)
        if not reference_ids or not reference_text:
            return -1.0
        reference_seconds = (
            len(reference_ids) / self.layout.num_codebooks / self.frame_rate
        )
        reference_units = len(
            normalize_transcript(reference_text, character_level=True)
        )
        target_units = len(normalize_transcript(target_text, character_level=True))
        if reference_units < 1 or target_units < 1:
            return -1.0
        return reference_seconds * target_units / reference_units

    def _intelligibility_scores(self, waves, target_texts, languages):
        config = self.config.get("asr", {})
        transcripts, nll_values = self._asr_scores(
            waves, target_texts, languages
        )
        scores, errors = [], []
        for target, transcript, nll, language in zip(
            target_texts, transcripts, nll_values, languages
        ):
            metric = config.get("metric", "cer")
            if metric == "auto":
                metric = "wer" if str(language).lower().startswith("en") else "cer"
            error = (
                transcript_error_rate(target, transcript, metric=metric)
                if config.get("error_weight", 0.6) > 0
                else None
            )
            error_score = (
                error_rate_reward(error, alpha=float(config.get("error_alpha", 3.0)))
                if error is not None
                else None
            )
            nll_score = (
                nll_reward(nll, alpha=float(config.get("nll_alpha", 3.0)))
                if nll is not None
                else None
            )
            scores.append(
                weighted_harmonic_mean(
                    [error_score, nll_score],
                    [config.get("error_weight", 0.6), config.get("nll_weight", 0.4)],
                )
            )
            errors.append(error)
        return scores, transcripts, errors, nll_values

    def __call__(
        self,
        completion_ids,
        target_text,
        reference_audio_ids=None,
        reference_text=None,
        target_duration_seconds=None,
        language=None,
        log_extra=None,
        log_metric=None,
        **kwargs,
    ):
        """TRL-compatible reward callable; all expensive models are lazy and batched."""
        count = len(completion_ids)
        reference_audio_ids = reference_audio_ids or [[] for _ in range(count)]
        reference_text = reference_text or [""] * count
        target_duration_seconds = target_duration_seconds or [-1.0] * count
        language = language or [""] * count
        parsed = [parse_audio_completion(ids, self.layout) for ids in completion_ids]
        valid_indices = [index for index, item in enumerate(parsed) if item.valid]
        generated_seconds = [item.num_frames / self.frame_rate for item in parsed]
        waves_by_index = {}
        if valid_indices:
            waves = self._get_codec().decode_batch(
                [parsed[index].codes for index in valid_indices]
            )
            waves_by_index.update(zip(valid_indices, waves))

        intelligibility = [None] * count
        transcripts = [""] * count
        errors = [None] * count
        nll_values = [None] * count
        if valid_indices and self.weights["intelligibility"] > 0:
            values = self._intelligibility_scores(
                [waves_by_index[index] for index in valid_indices],
                [target_text[index] for index in valid_indices],
                [language[index] for index in valid_indices],
            )
            for destination, source in enumerate(valid_indices):
                intelligibility[source] = values[0][destination]
                transcripts[source] = values[1][destination]
                errors[source] = values[2][destination]
                nll_values[source] = values[3][destination]

        speaker = [None] * count
        if valid_indices and self.weights["speaker"] > 0:
            speaker_values = self._speaker_scores(
                [waves_by_index[index] for index in valid_indices],
                [reference_audio_ids[index] for index in valid_indices],
            )
            for destination, source in enumerate(valid_indices):
                speaker[source] = speaker_values[destination]

        duration = [None] * count
        duration_config = self.config.get("duration", {})
        if self.weights["duration"] > 0:
            for index in valid_indices:
                expected = self._expected_duration(
                    target_duration_seconds[index],
                    reference_audio_ids[index],
                    reference_text[index],
                    target_text[index],
                )
                duration[index] = duration_consistency_reward(
                    generated_seconds[index],
                    expected,
                    mode=duration_config.get("mode", "binary"),
                    tolerance=duration_config.get("tolerance", [0.75, 1.25]),
                    scale=float(duration_config.get("scale", 0.35)),
                )

        speed = [None] * count
        speed_config = self.config.get("speed", {})
        if self.weights["speed"] > 0:
            maximum = max(float(speed_config.get("max_seconds", 12.0)), 1e-6)
            direction = speed_config.get("direction", "fast")
            if direction not in {"fast", "slow"}:
                raise ValueError("speed.direction must be 'fast' or 'slow'")
            for index in valid_indices:
                normalized = min(generated_seconds[index] / maximum, 1.0)
                speed[index] = 1.0 - normalized if direction == "fast" else normalized

        component_values = {
            "intelligibility": intelligibility,
            "speaker": speaker,
            "duration": duration,
            "speed": speed,
            "format": [float(item.valid) for item in parsed],
        }
        weights = self.weights
        rewards = []
        for index, item in enumerate(parsed):
            if not item.valid:
                rewards.append(self.invalid_reward)
                continue
            active = [
                (component_values[name][index], weight)
                for name, weight in weights.items()
                if weight > 0 and component_values[name][index] is not None
            ]
            if not active:
                rewards.append(self.invalid_reward)
                continue
            numerator = sum(value * weight for value, weight in active)
            rewards.append(float(numerator / sum(weight for _, weight in active)))

        if log_extra:
            log_extra("asr_transcript", transcripts)
            log_extra("audio_seconds", [round(value, 4) for value in generated_seconds])
            log_extra("audio_valid", [item.valid for item in parsed])
        if log_metric:
            for name, values in component_values.items():
                present = [float(value) for value in values if value is not None]
                if present:
                    log_metric(f"speech/{name}", sum(present) / len(present))
            present_errors = [float(value) for value in errors if value is not None]
            if present_errors:
                log_metric(
                    "speech/error_rate", sum(present_errors) / len(present_errors)
                )
            present_nll = [float(value) for value in nll_values if value is not None]
            if present_nll:
                log_metric("speech/asr_nll", sum(present_nll) / len(present_nll))
        return rewards
