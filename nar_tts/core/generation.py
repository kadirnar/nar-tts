"""Generation constraints for Nar's frame-interleaved Mimi token stream."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import LogitsProcessor

from nar_tts.core.tokens import AUDIO_OFFSET, TokenLayout


@dataclass(frozen=True)
class AudioCompletion:
    """Parsed speech completion and the structural checks used by rewards."""

    token_ids: list[int]
    codes: np.ndarray | None
    terminated: bool
    valid: bool
    num_frames: int


def audio_token_bounds(layout: TokenLayout, codebook: int) -> tuple[int, int]:
    """Return the half-open vocabulary range for one Mimi codebook."""
    if not 0 <= codebook < layout.num_codebooks:
        raise ValueError(
            f"codebook must be in [0, {layout.num_codebooks}), got {codebook}"
        )
    start = layout.base + AUDIO_OFFSET + codebook * layout.codebook_size
    return start, start + layout.codebook_size


def constrain_audio_logits(
    scores: torch.Tensor,
    layout: TokenLayout,
    generated_tokens: int,
    min_frames: int,
    max_frames: int,
    *,
    in_place: bool = False,
) -> torch.Tensor:
    """Apply Nar's next-token grammar to one or more rows of logits.

    This is the backend-neutral implementation used by Transformers, vLLM,
    and SGLang. ``generated_tokens`` counts completion tokens only; prompt
    tokens must not be included.
    """
    generated_tokens = int(generated_tokens)
    min_frames = int(min_frames)
    max_frames = int(max_frames)
    if scores.ndim < 1:
        raise ValueError("scores must have a vocabulary dimension")
    if generated_tokens < 0:
        raise ValueError("generated_tokens cannot be negative")
    if min_frames < 0 or max_frames < 1 or min_frames > max_frames:
        raise ValueError("expected 0 <= min_frames <= max_frames and max_frames > 0")
    highest_audio_id = audio_token_bounds(layout, layout.num_codebooks - 1)[1] - 1
    if max(highest_audio_id, layout.eos_speech) >= scores.shape[-1]:
        raise ValueError("logits vocabulary is smaller than the configured token layout")

    min_audio_tokens = min_frames * layout.num_codebooks
    max_audio_tokens = max_frames * layout.num_codebooks
    if generated_tokens >= max_audio_tokens:
        eos_value = scores[..., layout.eos_speech].clone()
        constrained = scores if in_place else torch.full_like(scores, -torch.inf)
        if in_place:
            constrained.fill_(-torch.inf)
        constrained[..., layout.eos_speech] = eos_value
        return constrained

    codebook = generated_tokens % layout.num_codebooks
    lower, upper = audio_token_bounds(layout, codebook)
    allowed_values = scores[..., lower:upper].clone() if in_place else None
    eos_allowed = codebook == 0 and generated_tokens >= min_audio_tokens
    eos_value = (
        scores[..., layout.eos_speech].clone() if in_place and eos_allowed else None
    )
    constrained = scores if in_place else torch.full_like(scores, -torch.inf)
    if in_place:
        constrained.fill_(-torch.inf)
        constrained[..., lower:upper] = allowed_values
    else:
        constrained[..., lower:upper] = scores[..., lower:upper]
    if eos_allowed:
        constrained[..., layout.eos_speech] = (
            eos_value if in_place else scores[..., layout.eos_speech]
        )
    return constrained


def parse_audio_completion(ids, layout: TokenLayout) -> AudioCompletion:
    """Validate ``audio frames + EOS_SPEECH`` and decode all complete frames.

    Tokens after the first speech EOS are ignored. A completion is structurally
    valid only when it contains at least one full 32-codebook frame, every token
    belongs to the expected codebook position, and it terminates with speech EOS.
    """
    token_ids = [int(token_id) for token_id in ids]
    try:
        eos_index = token_ids.index(layout.eos_speech)
    except ValueError:
        payload = token_ids
        terminated = False
    else:
        payload = token_ids[:eos_index]
        terminated = True

    expected = layout.num_codebooks
    valid_tokens = 0
    for position, token_id in enumerate(payload):
        lower, upper = audio_token_bounds(layout, position % expected)
        if not lower <= token_id < upper:
            break
        valid_tokens += 1

    num_frames = valid_tokens // expected
    valid = (
        terminated
        and num_frames > 0
        and valid_tokens == len(payload)
        and len(payload) % expected == 0
    )
    codes = layout.ids_to_codes(payload) if num_frames else None
    return AudioCompletion(
        token_ids=payload,
        codes=codes,
        terminated=terminated,
        valid=valid,
        num_frames=num_frames,
    )


class AudioTokenLogitsProcessor(LogitsProcessor):
    """Restrict autoregressive sampling to Nar's valid Mimi token grammar."""

    def __init__(
        self,
        layout: TokenLayout,
        prompt_length: int,
        min_frames: int,
        max_frames: int,
    ):
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if min_frames < 0:
            raise ValueError("min_frames cannot be negative")
        if max_frames < 1 or min_frames > max_frames:
            raise ValueError("max_frames must be positive and >= min_frames")
        self.layout = layout
        self.prompt_length = int(prompt_length)
        self.min_audio_tokens = int(min_frames) * layout.num_codebooks
        self.max_audio_tokens = int(max_frames) * layout.num_codebooks

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generated = input_ids.shape[-1] - self.prompt_length
        if generated < 0:
            raise ValueError("input is shorter than the rollout prompt")
        return constrain_audio_logits(
            scores,
            self.layout,
            generated_tokens=generated,
            min_frames=self.min_audio_tokens // self.layout.num_codebooks,
            max_frames=self.max_audio_tokens // self.layout.num_codebooks,
        )


def constrained_audio_log_probs(
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    layout: TokenLayout,
    min_frames: int,
    max_frames: int,
    pad_token_id: int,
    temperature: float = 1.0,
    compute_entropy: bool = False,
    entropy_requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Log probabilities under the same constrained policy used for rollouts.

    The denominator contains only the 2,048 entries of the expected codebook,
    plus speech EOS at legal frame boundaries. This both matches sampling and
    avoids a full-vocabulary softmax over Nar's large audio-token vocabulary.
    """
    if logits.ndim != 3 or selected_ids.ndim != 2:
        raise ValueError("expected logits (batch, time, vocab) and ids (batch, time)")
    if logits.shape[:2] != selected_ids.shape:
        raise ValueError("logits and selected_ids must have matching batch/time axes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_frames < 0 or max_frames < 1 or min_frames > max_frames:
        raise ValueError("expected 0 <= min_frames <= max_frames and max_frames > 0")

    _, sequence_length, vocab_size = logits.shape
    highest_audio_id = audio_token_bounds(layout, layout.num_codebooks - 1)[1] - 1
    if max(highest_audio_id, layout.eos_speech, pad_token_id) >= vocab_size:
        raise ValueError(
            "logits vocabulary is smaller than the configured token layout"
        )

    work_dtype = torch.float32
    selected_logits = (
        logits.gather(dim=-1, index=selected_ids.unsqueeze(-1))
        .squeeze(-1)
        .to(work_dtype)
    )
    selected_logits = selected_logits / temperature
    log_probs = torch.full_like(selected_logits, -torch.inf)
    entropies = torch.zeros_like(selected_logits) if compute_entropy else None
    positions = torch.arange(sequence_length, device=logits.device)
    min_audio_tokens = min_frames * layout.num_codebooks
    max_audio_tokens = max_frames * layout.num_codebooks

    for codebook in range(layout.num_codebooks):
        codebook_positions = positions[codebook :: layout.num_codebooks]
        if codebook_positions.numel() == 0:
            continue
        lower, upper = audio_token_bounds(layout, codebook)
        action_logits = (
            logits[:, codebook_positions, lower:upper].to(work_dtype) / temperature
        )
        log_normalizer = torch.logsumexp(action_logits, dim=-1)
        code_is_valid = (selected_ids[:, codebook_positions] >= lower) & (
            selected_ids[:, codebook_positions] < upper
        )

        if compute_entropy:
            entropy_logits = (
                action_logits if entropy_requires_grad else action_logits.detach()
            )
            action_log_probs = torch.log_softmax(entropy_logits, dim=-1)
            action_probs = action_log_probs.exp()
            action_entropy = -(action_probs * action_log_probs).sum(dim=-1)

        if codebook == 0:
            eos_logits = (
                logits[:, codebook_positions, layout.eos_speech].to(work_dtype)
                / temperature
            )
            eos_allowed = (codebook_positions >= min_audio_tokens) & (
                codebook_positions < max_audio_tokens
            )
            force_eos = codebook_positions >= max_audio_tokens
            log_normalizer = torch.where(
                eos_allowed.unsqueeze(0),
                torch.logaddexp(log_normalizer, eos_logits),
                log_normalizer,
            )
            log_normalizer = torch.where(
                force_eos.unsqueeze(0), eos_logits, log_normalizer
            )
            eos_is_valid = selected_ids[:, codebook_positions] == layout.eos_speech
            valid = (code_is_valid & ~force_eos.unsqueeze(0)) | (
                eos_is_valid & (eos_allowed | force_eos).unsqueeze(0)
            )

            if compute_entropy:
                entropy_eos = (
                    eos_logits if entropy_requires_grad else eos_logits.detach()
                )
                with_eos = torch.cat(
                    [entropy_logits, entropy_eos.unsqueeze(-1)], dim=-1
                )
                with_eos_log_probs = torch.log_softmax(with_eos, dim=-1)
                with_eos_entropy = -(with_eos_log_probs.exp() * with_eos_log_probs).sum(
                    dim=-1
                )
                action_entropy = torch.where(
                    eos_allowed.unsqueeze(0), with_eos_entropy, action_entropy
                )
                action_entropy = torch.where(
                    force_eos.unsqueeze(0),
                    torch.zeros_like(action_entropy),
                    action_entropy,
                )
                entropies[:, codebook_positions] = action_entropy
        else:
            valid = code_is_valid & (codebook_positions < max_audio_tokens).unsqueeze(0)
            if compute_entropy:
                entropies[:, codebook_positions] = action_entropy

        chosen = selected_logits[:, codebook_positions] - log_normalizer
        padded = selected_ids[:, codebook_positions] == pad_token_id
        log_probs[:, codebook_positions] = torch.where(
            padded,
            torch.zeros_like(chosen),
            torch.where(valid, chosen, torch.full_like(chosen, -torch.inf)),
        )
        if compute_entropy:
            entropies[:, codebook_positions] = torch.where(
                padded,
                torch.zeros_like(entropies[:, codebook_positions]),
                entropies[:, codebook_positions],
            )

    return log_probs, entropies
