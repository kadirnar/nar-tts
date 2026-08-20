"""SGLang rollout client with constrained audio tokens and LoRA synchronization."""

import json
from numbers import Integral
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nar_tts.core.generation import constrain_audio_logits, parse_audio_completion
from nar_tts.core.tokens import TokenLayout
from nar_tts.integrations.vllm import (
    GRAMMAR_ARGUMENT,
    _parse_grammar,
    audio_grammar_arguments,
)

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
except ImportError:  # Config validation does not require the optional runtime.

    class CustomLogitProcessor:  # type: ignore[no-redef]
        @classmethod
        def to_str(cls):
            raise ImportError(
                "SGLang is required to serialize Nar's custom logits processor; "
                "install the nar-tts[sglang] extra in the rollout client environment"
            )


class NarAudioSGLangLogitsProcessor(CustomLogitProcessor):
    """Server-side SGLang mask for one position in every active request."""

    def __call__(self, logits, custom_param_list=None):
        if not custom_param_list or len(custom_param_list) != logits.shape[0]:
            raise ValueError("Nar requires one SGLang custom-parameter mapping per row")
        for row, params in enumerate(custom_param_list):
            if not isinstance(params, dict):
                raise TypeError("Nar SGLang custom parameters must be mappings")
            parsed = _parse_grammar(params.get(GRAMMAR_ARGUMENT))
            if parsed is None:
                raise ValueError(f"missing {GRAMMAR_ARGUMENT} custom parameter")
            request = params.get("__req__")
            if request is None:
                raise ValueError("SGLang did not expose request state to the processor")
            layout, min_frames, max_frames = parsed
            constrain_audio_logits(
                logits[row],
                layout,
                generated_tokens=len(request.output_ids),
                min_frames=min_frames,
                max_frames=max_frames,
                in_place=True,
            )
        return logits


class SGLangRollout:
    """TRL ``rollout_func`` backed by a trusted external SGLang server.

    The current LoRA adapter is saved and hot-reloaded once per optimizer step.
    The server and trainer therefore need access to the same synchronization
    directory.
    """

    def __init__(
        self,
        config: dict,
        layout: TokenLayout,
        min_frames: int,
        max_frames: int,
    ):
        self.config = config
        self.layout = layout
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.base_url = str(config["base_url"]).rstrip("/")
        self.adapter_sync_dir = Path(config["adapter_sync_dir"]).expanduser().resolve()
        self.adapter_name = str(config.get("adapter_name", "nar_grpo_live"))
        self.timeout = float(config.get("timeout", 300.0))
        self.api_key = config.get("api_key")
        self.serialized_processor = NarAudioSGLangLogitsProcessor.to_str()
        self._loaded_step = -1

    def _post(self, endpoint: str, payload: dict, *, allow_error: bool = False):
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if allow_error:
                return None
            raise RuntimeError(
                f"SGLang {endpoint} failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"cannot reach the configured SGLang server at {self.base_url}: {error}"
            ) from error

    def _sync_adapter(self, trainer):
        from accelerate.utils import broadcast_object_list

        step = int(trainer.state.global_step)
        if step == self._loaded_step:
            return
        error_message = [None]
        if trainer.accelerator.is_main_process:
            try:
                model = trainer.accelerator.unwrap_model(trainer.model)
                if not hasattr(model, "peft_config"):
                    raise TypeError(
                        "SGLang live synchronization requires a PEFT model"
                    )
                if self._loaded_step >= 0:
                    self._post(
                        "unload_lora_adapter",
                        {"lora_name": self.adapter_name},
                        allow_error=True,
                    )
                self.adapter_sync_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(
                    self.adapter_sync_dir,
                    safe_serialization=True,
                    selected_adapters=["default"],
                )
                self._post(
                    "load_lora_adapter",
                    {
                        "lora_name": self.adapter_name,
                        "lora_path": str(self.adapter_sync_dir),
                        "pinned": bool(self.config.get("pin_adapter", True)),
                    },
                )
            # Every failure must be broadcast or the other ranks can deadlock.
            except Exception as error:  # noqa: BLE001
                error_message[0] = f"{type(error).__name__}: {error}"
        broadcast_object_list(error_message, from_process=0)
        if error_message[0] is not None:
            raise RuntimeError(
                f"failed to synchronize the live SGLang adapter: {error_message[0]}"
            )
        self._loaded_step = step

    def __call__(self, prompts, trainer):
        prompt_ids = []
        for prompt in prompts:
            if not isinstance(prompt, (list, tuple)) or not all(
                isinstance(token_id, Integral) for token_id in prompt
            ):
                raise TypeError("SGLang rollout expects Nar token-id prompts")
            prompt_ids.append([int(token_id) for token_id in prompt])
        self._sync_adapter(trainer)

        grammar = audio_grammar_arguments(
            self.layout, self.min_frames, self.max_frames
        )
        top_k = int(getattr(trainer, "top_k", 0))
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": int(trainer.max_completion_length),
                "temperature": float(trainer.temperature),
                "top_p": float(trainer.top_p),
                "top_k": -1 if top_k == 0 else top_k,
                "repetition_penalty": float(trainer.repetition_penalty),
                "stop_token_ids": [self.layout.eos_speech],
                "no_stop_trim": True,
                "skip_special_tokens": False,
                "custom_params": {GRAMMAR_ARGUMENT: grammar},
            },
            "custom_logit_processor": self.serialized_processor,
            "lora_path": self.adapter_name,
        }
        response = self._post("generate", payload)
        rows = response if isinstance(response, list) else [response]
        if len(rows) != len(prompt_ids):
            raise RuntimeError(
                "SGLang returned a different number of completions than prompts"
            )

        completion_ids = []
        for index, (prompt, row) in enumerate(zip(prompt_ids, rows, strict=True)):
            ids = [int(token_id) for token_id in row.get("output_ids", [])]
            completion_count = row.get("meta_info", {}).get("completion_tokens")
            if completion_count is not None:
                ids = ids[-int(completion_count) :]
            elif ids[: len(prompt)] == prompt:
                ids = ids[len(prompt) :]
            parsed = parse_audio_completion(ids, self.layout)
            if not parsed.valid:
                raise RuntimeError(
                    f"SGLang produced an invalid Nar audio sequence at batch index {index}; "
                    "verify --enable-custom-logit-processor and the configured stop token"
                )
            eos_index = ids.index(self.layout.eos_speech)
            completion_ids.append(ids[: eos_index + 1])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            # Nar recomputes exact constrained policy log-probabilities locally.
            "logprobs": None,
        }
