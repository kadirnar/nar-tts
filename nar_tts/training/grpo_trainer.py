"""TRL GRPO trainer specialized for Nar's 32-codebook speech action space."""

from numbers import Integral

import torch
from transformers import LogitsProcessorList
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad

from nar_tts.core.generation import (
    AudioTokenLogitsProcessor,
    constrained_audio_log_probs,
    parse_audio_completion,
)
from nar_tts.core.tokens import TokenLayout


def _is_token_prompt(prompt) -> bool:
    return isinstance(prompt, (list, tuple)) and all(
        isinstance(token_id, Integral) for token_id in prompt
    )


class NarGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with direct token prompts and grammar-constrained speech."""

    def __init__(
        self,
        *args,
        token_layout: TokenLayout,
        min_audio_frames: int,
        max_audio_frames: int,
        constrain_log_probs: bool = True,
        **kwargs,
    ):
        self.token_layout = token_layout
        self.min_audio_frames = int(min_audio_frames)
        self.max_audio_frames = int(max_audio_frames)
        self.constrain_log_probs = bool(constrain_log_probs)
        if self.min_audio_frames < 0:
            raise ValueError("min_audio_frames cannot be negative")
        if self.max_audio_frames < 1 or self.min_audio_frames > self.max_audio_frames:
            raise ValueError(
                "max_audio_frames must be positive and >= min_audio_frames"
            )
        training_args = kwargs.get("args")
        if training_args is None and len(args) > 1:
            training_args = args[1]
        if training_args is not None:
            if getattr(training_args, "use_transformers_continuous_batching", False):
                raise ValueError(
                    "Nar's codebook grammar currently requires regular Transformers generation"
                )
            if self.constrain_log_probs and getattr(
                training_args, "use_liger_kernel", False
            ):
                raise ValueError(
                    "use_liger_kernel must be false when constrained log-probabilities "
                    "are enabled"
                )
        super().__init__(*args, **kwargs)

    def _tokenize_prompts(self, prompts):
        if prompts and all(_is_token_prompt(prompt) for prompt in prompts):
            return (
                [[int(token_id) for token_id in prompt] for prompt in prompts],
                None,
                {},
            )
        raise TypeError(
            "NarGRPOTrainer expects dataset prompt rows to contain token-id lists; "
            "use nar_tts.training.grpo_data to prepare the dataset"
        )

    def _generate_single_turn(
        self,
        prompt_ids,
        images,
        multimodal_fields,
        has_tool_images=False,
    ):
        if images is not None or multimodal_fields or has_tool_images:
            raise ValueError(
                "Nar speech GRPO does not accept multimodal processor inputs"
            )
        if self.use_vllm:
            completion_ids, logprobs = super()._generate_single_turn(
                prompt_ids,
                images,
                multimodal_fields,
                has_tool_images=has_tool_images,
            )
            for index, ids in enumerate(completion_ids):
                if not parse_audio_completion(ids, self.token_layout).valid:
                    raise RuntimeError(
                        "vLLM produced an invalid Nar audio sequence at batch "
                        f"index {index}; ensure this checkout is installed with "
                        "`pip install -e .` so its logits processor is loaded"
                    )
            return completion_ids, logprobs
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids]
        padded_ids = pad(
            prompt_tensors,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        attention_mask = pad(
            [torch.ones_like(tensor) for tensor in prompt_tensors],
            padding_value=0,
            padding_side="left",
        ).to(device)
        processor = AudioTokenLogitsProcessor(
            self.token_layout,
            prompt_length=padded_ids.shape[1],
            min_frames=self.min_audio_frames,
            max_frames=self.max_audio_frames,
        )

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            self._dist.summon_full_params(self.model_wrapped, recurse=False),
        ):
            generated = unwrapped_model.generate(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                logits_processor=LogitsProcessorList([processor]),
            )

        generated = generated[:, padded_ids.shape[1] :].cpu().tolist()
        completion_ids = []
        for ids in generated:
            try:
                eos_index = ids.index(self.token_layout.eos_speech)
            except ValueError:
                completion_ids.append(ids)
            else:
                completion_ids.append(ids[: eos_index + 1])
        return completion_ids, None

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        **forward_kwargs,
    ):
        if not self.constrain_log_probs:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                **forward_kwargs,
            )

        unsupported = {
            name: value
            for name, value in forward_kwargs.items()
            if value is not None and name not in {"token_type_ids"}
        }
        if unsupported:
            raise ValueError(
                "Nar constrained log-probabilities do not support these model inputs: "
                + ", ".join(sorted(unsupported))
            )

        batch_size = batch_size or input_ids.shape[0]
        all_log_probs = []
        all_entropies = []
        all_aux_losses = []
        for start in range(0, input_ids.shape[0], batch_size):
            model_inputs = {
                "input_ids": input_ids[start : start + batch_size],
                "attention_mask": attention_mask[start : start + batch_size],
                "use_cache": False,
            }
            token_type_ids = forward_kwargs.get("token_type_ids")
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[
                    start : start + batch_size
                ]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            if compute_aux_loss:
                model_inputs["output_router_logits"] = True

            outputs = model(**model_inputs)
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            completion_ids = model_inputs["input_ids"][:, -logits_to_keep:]
            log_probs, entropies = constrained_audio_log_probs(
                logits,
                completion_ids,
                self.token_layout,
                min_frames=self.min_audio_frames,
                max_frames=self.max_audio_frames,
                pad_token_id=self._tokenizer.pad_token_id,
                temperature=self.temperature,
                compute_entropy=compute_entropy,
                entropy_requires_grad=self._entropy_bonus_enabled,
            )
            all_log_probs.append(log_probs)
            if compute_entropy:
                all_entropies.append(entropies)
            if compute_aux_loss:
                all_aux_losses.append(outputs.aux_loss)

        log_probs = torch.cat(all_log_probs, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        aux_loss = torch.stack(all_aux_losses).mean() if compute_aux_loss else None
        return log_probs, entropies, aux_loss
