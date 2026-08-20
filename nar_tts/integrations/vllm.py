"""vLLM V1 logits processor for Nar's frame-interleaved audio grammar."""

from dataclasses import dataclass
from typing import Any

import torch

from nar_tts.core.generation import constrain_audio_logits
from nar_tts.core.tokens import TokenLayout

try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
except ImportError:  # Keep config validation and unit tests lightweight.

    class AdapterLogitsProcessor:  # type: ignore[no-redef]
        """Fallback base used only when vLLM is not installed."""


GRAMMAR_ARGUMENT = "nar_audio_grammar"
_GRAMMAR_FIELDS = {
    "base",
    "eot",
    "num_codebooks",
    "codebook_size",
    "min_frames",
    "max_frames",
}


def audio_grammar_arguments(
    layout: TokenLayout, min_frames: int, max_frames: int
) -> dict[str, int]:
    """Return JSON-safe request arguments understood by Nar's vLLM plugin."""
    return {
        "base": int(layout.base),
        "eot": int(layout.eot),
        "num_codebooks": int(layout.num_codebooks),
        "codebook_size": int(layout.codebook_size),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
    }


def _parse_grammar(value: Any) -> tuple[TokenLayout, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{GRAMMAR_ARGUMENT} must be a mapping")
    missing = sorted(_GRAMMAR_FIELDS - value.keys())
    if missing:
        raise ValueError(
            f"{GRAMMAR_ARGUMENT} is missing fields: {', '.join(missing)}"
        )
    parsed = {name: int(value[name]) for name in _GRAMMAR_FIELDS}
    layout = TokenLayout(
        base=parsed["base"],
        eot=parsed["eot"],
        num_codebooks=parsed["num_codebooks"],
        codebook_size=parsed["codebook_size"],
    )
    min_frames, max_frames = parsed["min_frames"], parsed["max_frames"]
    if layout.base < 0 or layout.eot < 0:
        raise ValueError("Nar token ids cannot be negative")
    if layout.num_codebooks < 1 or layout.codebook_size < 1:
        raise ValueError("Nar codebook counts and sizes must be positive")
    if min_frames < 0 or max_frames < 1 or min_frames > max_frames:
        raise ValueError("Nar frame limits must satisfy 0 <= min <= max and max > 0")
    return layout, min_frames, max_frames


@dataclass
class NarAudioRequestLogitsProcessor:
    """Request-level mask wrapped by vLLM's persistent-batch adapter."""

    layout: TokenLayout
    min_frames: int
    max_frames: int

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        return constrain_audio_logits(
            logits,
            self.layout,
            generated_tokens=len(output_ids),
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            in_place=True,
        )


class NarAudioLogitsProcessor(AdapterLogitsProcessor):
    """Auto-discovered vLLM plugin activated by ``nar_audio_grammar``."""

    @classmethod
    def validate_params(cls, params):
        extra_args = getattr(params, "extra_args", None) or {}
        _parse_grammar(extra_args.get(GRAMMAR_ARGUMENT))

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        extra_args = getattr(params, "extra_args", None) or {}
        parsed = _parse_grammar(extra_args.get(GRAMMAR_ARGUMENT))
        if parsed is None:
            return None
        layout, min_frames, max_frames = parsed
        return NarAudioRequestLogitsProcessor(layout, min_frames, max_frames)
