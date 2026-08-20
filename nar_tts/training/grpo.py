"""Quality-focused GRPO post-training for Nar TTS.

Launch with the checked-in eight-GPU recipe::

    torchrun --standalone --nproc-per-node=8 \
        nar_tts/training/grpo.py --config nar_tts/configs/grpo.yaml
"""

import argparse
import copy
import json
import os
from importlib.metadata import entry_points
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig

from nar_tts.core.tokens import EOS_SPEECH, TokenLayout
from nar_tts.integrations.vllm import GRAMMAR_ARGUMENT, audio_grammar_arguments
from nar_tts.training.grpo_config import (
    dataloader_worker_count,
    load_grpo_config,
    validate_grpo_config,
)
from nar_tts.training.grpo_data import load_grpo_dataset
from nar_tts.training.grpo_rewards import SpeechRewardSuite
from nar_tts.training.grpo_trainer import NarGRPOTrainer

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "grpo.yaml"
)


def _dtype(name):
    if name in (None, "auto"):
        return None
    value = getattr(torch, str(name), None)
    if not isinstance(value, torch.dtype):
        raise TypeError(f"unknown model dtype: {name!r}")
    return value


def _prepare_tokenizer(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    layout = TokenLayout.from_tokenizer(tokenizer)
    original_eot = layout.eot
    if original_eot is None:
        raise ValueError("the base text tokenizer must define eos_token_id")
    vocab = tokenizer.get_vocab()
    custom_zero = vocab.get("<custom_token_0>")
    if custom_zero is None:
        tokenizer.add_tokens(layout.added_token_strings())

    # Verify the complete contiguous mapping instead of silently accepting an
    # expanded tokenizer with a different audio layout.
    final_token = layout.added_token_strings()[-1]
    for token, expected_id in (
        ("<custom_token_0>", layout.base),
        (f"<custom_token_{EOS_SPEECH}>", layout.eos_speech),
        (final_token, layout.base + layout.num_added_tokens),
    ):
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(
                f"tokenizer audio layout mismatch for {token}: "
                f"expected {expected_id}, got {actual_id}"
            )

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = original_eot
    tokenizer.eos_token = f"<custom_token_{EOS_SPEECH}>"
    if tokenizer.eos_token_id != layout.eos_speech:
        raise ValueError("failed to assign EOS_SPEECH as the rollout terminator")
    return tokenizer, layout


def _load_model(config: dict, tokenizer, layout: TokenLayout):
    dtype = _dtype(config.get("dtype", "bfloat16"))
    kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    for name in ("revision", "attn_implementation"):
        if config.get(name) is not None:
            kwargs[name] = config[name]
    model = AutoModelForCausalLM.from_pretrained(config["checkpoint"], **kwargs)
    embedding_size = model.get_input_embeddings().num_embeddings
    if embedding_size != len(tokenizer):
        raise ValueError(
            "GRPO must start after speech pretraining with the expanded Nar vocabulary: "
            f"checkpoint has {embedding_size} embeddings, tokenizer requires {len(tokenizer)}"
        )
    if model.get_output_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError(
            "checkpoint LM head does not match the expanded Nar vocabulary"
        )
    model.config.eos_token_id = layout.eos_speech
    model.config.pad_token_id = layout.eot
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = layout.eos_speech
        model.generation_config.pad_token_id = layout.eot
    return model


def _peft_config(config: dict):
    if not config.get("enabled", True):
        return None
    target_modules = config.get("target_modules", "all-linear")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("rank", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.0)),
        target_modules=target_modules,
        bias=config.get("bias", "none"),
        use_rslora=bool(config.get("use_rslora", True)),
    )


def _training_args(
    config: dict,
    derived: dict,
    tokenizer,
    world_size: int,
    layout: TokenLayout | None = None,
):
    training = config["training"]
    grpo = config.get("grpo", {})
    generation = config.get("generation", {})
    logging = config.get("logging", {})
    runtime = config.get("runtime", {})
    rollout = config.get("rollout", {})
    rollout_backend = rollout.get("backend", "transformers")
    configured_reward_weights = config.get("rewards", {}).get("weights", {})
    reward_names = (
        "intelligibility",
        "speaker",
        "duration",
        "speed",
        "format",
        "naturalness",
        "prosody",
        "emotion",
        "event",
        "speaker_drift",
    )
    reward_weights = [
        float(
            configured_reward_weights.get(
                name, 1.0 if name == "intelligibility" else 0.0
            )
        )
        for name in reward_names
        if float(
            configured_reward_weights.get(
                name, 1.0 if name == "intelligibility" else 0.0
            )
        )
        > 0
    ]
    workers = dataloader_worker_count(
        runtime.get("dataloader_workers", "auto"), world_size=world_size
    )
    report_to = logging.get(
        "report_to", "wandb" if logging.get("enabled", True) else "none"
    )
    if isinstance(report_to, str) and report_to != "none":
        report_to = [report_to]

    if rollout_backend == "vllm":
        layout = layout or TokenLayout.from_tokenizer(tokenizer)
        generation_kwargs = {
            "stop_token_ids": [layout.eos_speech],
            "extra_args": {
                GRAMMAR_ARGUMENT: audio_grammar_arguments(
                    layout, derived["min_frames"], derived["max_frames"]
                )
            },
        }
    else:
        generation_kwargs = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

    kwargs = {
        "output_dir": training["output_dir"],
        "num_train_epochs": float(training.get("epochs", 1.0)),
        "max_steps": int(training.get("max_steps", -1)),
        "per_device_train_batch_size": int(
            training.get("per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 1)
        ),
        "learning_rate": float(training.get("learning_rate", 1e-5)),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        # Transformers 5 accepts an absolute count (>= 1) or a fraction (< 1)
        # through warmup_steps; warmup_ratio is retained only as a YAML alias.
        "warmup_steps": float(
            training.get("warmup_steps", training.get("warmup_ratio", 0.03))
        ),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "optim": training.get("optim", "adamw_torch_fused"),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "tf32": training.get("tf32", True),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": training.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        "use_cache": False,
        "use_liger_kernel": bool(training.get("use_liger_kernel", False)),
        "logging_steps": int(training.get("logging_steps", 1)),
        "logging_first_step": True,
        "save_strategy": training.get("save_strategy", "steps"),
        "save_steps": int(training.get("save_steps", 100)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "report_to": report_to,
        "run_name": logging.get("run_name"),
        "project": logging.get("project", "nar-tts-grpo"),
        "remove_unused_columns": False,
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": bool(runtime.get("pin_memory", True)),
        "dataloader_persistent_workers": workers > 0,
        "dataloader_prefetch_factor": (
            int(runtime.get("dataloader_prefetch_factor", 2)) if workers > 0 else None
        ),
        "seed": int(training.get("seed", 42)),
        "data_seed": int(training.get("data_seed", training.get("seed", 42))),
        "ddp_find_unused_parameters": False,
        "average_tokens_across_devices": bool(
            training.get("average_tokens_across_devices", True)
        ),
        "num_generations": int(grpo.get("num_generations", 4)),
        "num_generations_eval": int(
            grpo.get("num_generations_eval", grpo.get("num_generations", 4))
        ),
        "max_completion_length": derived["max_completion_length"],
        "temperature": float(generation.get("temperature", 0.9)),
        "top_p": float(generation.get("top_p", 0.95)),
        "top_k": int(generation.get("top_k", 0)),
        "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        "generation_kwargs": generation_kwargs,
        "beta": float(grpo.get("beta", 0.1)),
        "num_iterations": int(grpo.get("num_iterations", 1)),
        "epsilon": float(grpo.get("epsilon", 0.2)),
        "epsilon_high": float(grpo.get("epsilon_high", grpo.get("epsilon", 0.2))),
        "importance_sampling_level": grpo.get("importance_sampling_level", "token"),
        "scale_rewards": grpo.get("scale_rewards", "group"),
        "multi_objective_aggregation": grpo.get(
            "multi_objective_aggregation", "normalize_then_sum"
        ),
        "reward_weights": reward_weights,
        "loss_type": grpo.get("loss_type", "dapo"),
        "mask_truncated_completions": bool(
            grpo.get("mask_truncated_completions", True)
        ),
        "entropy_coef": float(grpo.get("entropy_coef", 0.0)),
        "disable_dropout": True,
        "use_vllm": rollout_backend == "vllm",
        "use_transformers_continuous_batching": False,
        "log_completions": bool(logging.get("log_completions", True)),
        "num_completions_to_print": int(logging.get("num_completions_to_print", 2)),
    }
    for name in ("generation_batch_size", "steps_per_generation"):
        if grpo.get(name) is not None:
            kwargs[name] = int(grpo[name])
    if rollout_backend == "vllm":
        vllm = rollout.get("vllm", {})
        kwargs.update(
            {
                "vllm_mode": vllm.get("mode", "colocate"),
                "vllm_model_impl": vllm.get("model_impl", "vllm"),
                "vllm_enable_sleep_mode": bool(
                    vllm.get("enable_sleep_mode", False)
                ),
                "vllm_server_base_url": vllm.get("server_base_url"),
                "vllm_server_host": vllm.get("server_host", "0.0.0.0"),
                "vllm_server_port": int(vllm.get("server_port", 8000)),
                "vllm_server_timeout": float(vllm.get("server_timeout", 240.0)),
                "vllm_group_port": int(vllm.get("group_port", 51216)),
                "vllm_gpu_memory_utilization": float(
                    vllm.get("gpu_memory_utilization", 0.3)
                ),
                "vllm_max_model_length": vllm.get("max_model_length"),
                "vllm_tensor_parallel_size": int(
                    vllm.get("tensor_parallel_size", 1)
                ),
                "vllm_importance_sampling_correction": bool(
                    vllm.get("importance_sampling_correction", True)
                ),
                "vllm_importance_sampling_mode": vllm.get(
                    "importance_sampling_mode", "sequence_mask"
                ),
                "vllm_importance_sampling_clip_max": float(
                    vllm.get("importance_sampling_clip_max", 3.0)
                ),
            }
        )
    return GRPOConfig(**kwargs)


def _verify_vllm_plugin():
    group = entry_points().select(group="vllm.logits_processors")
    if not any(
        item.value == "nar_tts.integrations.vllm:NarAudioLogitsProcessor"
        for item in group
    ):
        raise RuntimeError(
            "vLLM rollout requires Nar's logits-processor entry point. Install "
            "this checkout once with `pip install -e .` in the training and "
            "vLLM-server environments."
        )


def train(config: dict):
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    derived = validate_grpo_config(config, world_size=actual_world_size)
    training = config["training"]
    logging = config.get("logging", {})
    if logging.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(logging["project"]))
    if logging.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(logging["entity"]))
    set_seed(int(training.get("seed", 42)))

    tokenizer, layout = _prepare_tokenizer(config["model"])
    dataset_config = copy.deepcopy(config["dataset"])
    dataset_config.setdefault(
        "frame_rate", config.get("generation", {}).get("frame_rate", 12.5)
    )
    train_dataset = load_grpo_dataset(dataset_config, tokenizer, layout)
    peft_settings = config.get("peft", {})
    model = _load_model(config["model"], tokenizer, layout)
    trainer_peft_config = _peft_config(peft_settings)
    reward_config = copy.deepcopy(config["rewards"])
    reward_config.setdefault(
        "frame_rate", config.get("generation", {}).get("frame_rate", 12.5)
    )
    rewards = SpeechRewardSuite(layout, reward_config)
    reward_functions, _ = rewards.reward_functions()
    args = _training_args(
        config, derived, tokenizer, actual_world_size, layout=layout
    )
    rollout_func = None
    rollout_backend = derived["rollout_backend"]
    if rollout_backend == "vllm":
        _verify_vllm_plugin()
    elif rollout_backend == "sglang":
        from nar_tts.integrations.sglang import SGLangRollout

        rollout_func = SGLangRollout(
            config["rollout"]["sglang"],
            layout,
            derived["min_frames"],
            derived["max_frames"],
        )
    trainer = NarGRPOTrainer(
        model=model,
        args=args,
        reward_funcs=reward_functions,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=trainer_peft_config,
        rollout_func=rollout_func,
        token_layout=layout,
        min_audio_frames=derived["min_frames"],
        max_audio_frames=derived["max_frames"],
        constrain_log_probs=bool(
            config.get("action_space", {}).get("constrain_log_probs", True)
        ),
    )
    resume = training.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(training["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training["output_dir"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the scenario and print derived rollout sizes without loading models",
    )
    args = parser.parse_args(argv)
    config = load_grpo_config(args.config)
    if args.validate_only:
        expected_world_size = int(
            config.get("runtime", {}).get("expected_world_size", 1)
        )
        derived = validate_grpo_config(config, world_size=expected_world_size)
        print(
            json.dumps(
                {
                    "config": config["_config_path"],
                    **derived,
                },
                indent=2,
            )
        )
        return
    train(config)


if __name__ == "__main__":
    main()
