"""Configuration loading and fail-fast validation for speech GRPO."""

import math
import os
from pathlib import Path

import yaml

from nar_tts.core.model_ids import QWEN3_ASR_MODEL_ID, WAVLM_SPEAKER_MODEL_ID
from nar_tts.core.tokens import NUM_CODEBOOKS


class GRPOConfigError(ValueError):
    """Raised when a GRPO scenario is internally inconsistent."""


def _merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_grpo_config(path, _seen=None) -> dict:
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise GRPOConfigError(f"cyclic GRPO config inheritance at {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise GRPOConfigError(f"{path} must contain a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        base = load_grpo_config(parent_path, _seen=seen)
        base.pop("_config_path", None)
        config = _merge_config(base, config)
    config["_config_path"] = str(path)
    seen.remove(path)
    return config


def audio_frame_limits(config: dict) -> tuple[int, int]:
    generation = config.get("generation", {})
    frame_rate = float(generation.get("frame_rate", 12.5))
    minimum = float(generation.get("min_audio_seconds", 0.4))
    maximum = float(generation.get("max_audio_seconds", 12.0))
    if frame_rate <= 0 or minimum < 0 or maximum <= 0 or minimum > maximum:
        raise GRPOConfigError(
            "generation must satisfy frame_rate > 0 and "
            "0 <= min_audio_seconds <= max_audio_seconds"
        )
    return max(1, math.ceil(minimum * frame_rate)), math.ceil(maximum * frame_rate)


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def dataloader_worker_count(value, world_size: int = 1) -> int:
    if value in (None, "auto"):
        return max(1, available_cpu_count() // max(1, int(world_size)))
    workers = int(value)
    if workers < 0:
        raise GRPOConfigError("runtime.dataloader_workers cannot be negative")
    return workers


def _required(config: dict, path: str):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or value.get(key) in (None, ""):
            raise GRPOConfigError(f"{path} must be set")
        value = value[key]
    return value


def validate_grpo_config(config: dict, world_size: int | None = None) -> dict:
    """Validate one post-training scenario and return useful derived values."""
    _required(config, "model.checkpoint")
    _required(config, "model.tokenizer")
    _required(config, "dataset.path")
    _required(config, "training.output_dir")
    training = config.get("training", {})
    grpo = config.get("grpo", {})
    generation = config.get("generation", {})
    dataset = config.get("dataset", {})
    rewards = config.get("rewards", {})
    weights = rewards.get("weights", {})
    rollout = config.get("rollout", {})
    model = config.get("model", {})

    min_frames, max_frames = audio_frame_limits(config)
    if float(generation.get("temperature", 1.0)) <= 0:
        raise GRPOConfigError("generation.temperature must be positive")
    num_generations = int(grpo.get("num_generations", 4))
    if num_generations < 2:
        raise GRPOConfigError("grpo.num_generations must be at least 2")
    per_device_batch = int(training.get("per_device_train_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if per_device_batch < 1 or accumulation < 1:
        raise GRPOConfigError("training batch size and accumulation must be positive")

    if world_size is None:
        world_size = int(config.get("runtime", {}).get("expected_world_size", 1))
    if int(world_size) < 1:
        raise GRPOConfigError("world_size must be positive")
    effective_batch = per_device_batch * accumulation * int(world_size)
    if effective_batch % num_generations:
        raise GRPOConfigError(
            "world_size * per_device_train_batch_size * "
            "gradient_accumulation_steps must be divisible by num_generations; "
            f"got {effective_batch} and {num_generations}"
        )
    generation_batch = int(grpo.get("generation_batch_size", effective_batch))
    global_microbatch = per_device_batch * int(world_size)
    if generation_batch % num_generations:
        raise GRPOConfigError(
            "grpo.generation_batch_size must be divisible by num_generations"
        )
    if generation_batch % global_microbatch:
        raise GRPOConfigError(
            "grpo.generation_batch_size must be divisible by the global microbatch"
        )

    if dataset.get("streaming", False) and int(training.get("max_steps", -1)) <= 0:
        raise GRPOConfigError(
            "streaming datasets require a positive training.max_steps"
        )
    mode = dataset.get("mode", "tts_tokens")
    if mode not in {"tts_tokens", "text", "prompt_ids", "voice_clone_tokens"}:
        raise GRPOConfigError(f"unsupported dataset.mode: {mode!r}")
    columns = dataset.get("columns", {})
    required_columns = {
        "tts_tokens": ("input_ids",),
        "text": ("text",),
        "prompt_ids": ("prompt_ids", "text"),
        "voice_clone_tokens": ("text", "reference_text", "reference_audio_ids"),
    }[mode]
    missing = [name for name in required_columns if not columns.get(name)]
    if missing:
        raise GRPOConfigError(
            f"dataset mode {mode!r} requires column mappings: {', '.join(missing)}"
        )
    if dataset.get("on_invalid", "error") not in {"error", "drop"}:
        raise GRPOConfigError("dataset.on_invalid must be 'error' or 'drop'")

    active_weights = {
        name: float(weights.get(name, 1.0 if name == "intelligibility" else 0.0))
        for name in ("intelligibility", "speaker", "duration", "speed", "format")
    }
    if any(weight < 0 for weight in active_weights.values()):
        raise GRPOConfigError("reward weights cannot be negative")
    if not any(weight > 0 for weight in active_weights.values()):
        raise GRPOConfigError("at least one reward weight must be positive")
    if active_weights["intelligibility"] > 0 and not rewards.get("asr", {}).get(
        "enabled", True
    ):
        raise GRPOConfigError(
            "the intelligibility reward requires rewards.asr.enabled: true"
        )
    asr = rewards.get("asr", {})
    if asr.get("enabled", active_weights["intelligibility"] > 0):
        asr_model = asr.get("model", QWEN3_ASR_MODEL_ID)
        if asr_model != QWEN3_ASR_MODEL_ID:
            raise GRPOConfigError(
                "Nar standardizes ASR rewards on the full Qwen3-ASR-1.7B model; "
                f"set rewards.asr.model to {QWEN3_ASR_MODEL_ID!r}"
            )
    if active_weights["speaker"] > 0 and not rewards.get("speaker", {}).get(
        "enabled", False
    ):
        raise GRPOConfigError(
            "the speaker reward requires rewards.speaker.enabled: true"
        )
    if active_weights["speed"] > 0 and rewards.get("speed", {}).get(
        "direction", "fast"
    ) not in {"fast", "slow"}:
        raise GRPOConfigError("rewards.speed.direction must be 'fast' or 'slow'")
    if active_weights["intelligibility"] > 0 and not any(
        float(asr.get(name, default)) > 0
        for name, default in (("error_weight", 0.6), ("nll_weight", 0.4))
    ):
        raise GRPOConfigError(
            "the intelligibility reward needs a positive ASR error or NLL weight"
        )
    if asr.get("metric", "cer") not in {"cer", "wer", "auto"}:
        raise GRPOConfigError("rewards.asr.metric must be cer, wer, or auto")
    if asr.get("nll_reduction", "sum") not in {"sum", "mean"}:
        raise GRPOConfigError("rewards.asr.nll_reduction must be sum or mean")
    if rewards.get("duration", {}).get("mode", "binary") not in {
        "binary",
        "smooth_log",
    }:
        raise GRPOConfigError("rewards.duration.mode must be binary or smooth_log")

    speaker = rewards.get("speaker", {})
    speaker_backend = speaker.get("backend", "espnet")
    if speaker_backend not in {"espnet", "transformers_xvector"}:
        raise GRPOConfigError(
            "rewards.speaker.backend must be espnet or transformers_xvector"
        )
    if active_weights["speaker"] > 0 and speaker_backend == "espnet":
        speaker_model = speaker.get("model", WAVLM_SPEAKER_MODEL_ID)
        if speaker_model != WAVLM_SPEAKER_MODEL_ID:
            raise GRPOConfigError(
                "the ESPnet speaker backend must use the checked high-quality "
                f"WavLM-Large checkpoint {WAVLM_SPEAKER_MODEL_ID!r}"
            )

    model_loader = model.get("loader", "transformers")
    if model_loader != "transformers":
        raise GRPOConfigError(
            "GRPO model.loader must be transformers; use finetune_unsloth.yaml "
            "for Unsloth LoRA post-training in its separate environment"
        )

    if "use_vllm" in grpo:
        raise GRPOConfigError(
            "grpo.use_vllm was replaced by rollout.backend; use transformers, "
            "vllm, or sglang there"
        )
    rollout_backend = rollout.get("backend", "transformers")
    if rollout_backend not in {"transformers", "vllm", "sglang"}:
        raise GRPOConfigError(
            "rollout.backend must be transformers, vllm, or sglang"
        )
    if rollout_backend == "vllm":
        vllm = rollout.get("vllm", {})
        if vllm.get("mode", "colocate") not in {"colocate", "server"}:
            raise GRPOConfigError("rollout.vllm.mode must be colocate or server")
        utilization = float(vllm.get("gpu_memory_utilization", 0.3))
        if not 0 < utilization <= 1:
            raise GRPOConfigError(
                "rollout.vllm.gpu_memory_utilization must be in (0, 1]"
            )
        if int(vllm.get("tensor_parallel_size", 1)) < 1:
            raise GRPOConfigError(
                "rollout.vllm.tensor_parallel_size must be positive"
            )
    if rollout_backend == "sglang":
        sglang = rollout.get("sglang", {})
        for name in ("base_url", "adapter_sync_dir"):
            if sglang.get(name) in (None, ""):
                raise GRPOConfigError(f"rollout.sglang.{name} must be set")
        if not config.get("peft", {}).get("enabled", True):
            raise GRPOConfigError(
                "SGLang rollout weight synchronization currently requires PEFT/LoRA"
            )
    if grpo.get("use_transformers_continuous_batching", False):
        raise GRPOConfigError(
            "continuous batching is not supported by Nar's constrained generator"
        )
    constrain_log_probs = config.get("action_space", {}).get(
        "constrain_log_probs", True
    )
    if constrain_log_probs and training.get("use_liger_kernel", False):
        raise GRPOConfigError(
            "training.use_liger_kernel must be false with constrained log-probabilities"
        )

    return {
        "min_frames": min_frames,
        "max_frames": max_frames,
        "max_completion_length": max_frames * NUM_CODEBOOKS + 1,
        "effective_batch_size": effective_batch,
        "generation_batch_size": generation_batch,
        "world_size": int(world_size),
        "rollout_backend": rollout_backend,
    }
