import math
import sys

import numpy as np
import torch

from nar_tts.core.audio import MimiCodec
from nar_tts.core.tokens import AUDIO_OFFSET, CODEBOOK_SIZE, NUM_CODEBOOKS, TokenLayout

# A quick standalone sanity check — confirms the codec loads and the flatten/
# unflatten layout in core/tokens.py matches what Mimi produces, before launching
# a long encode/train job. Run: python tests/test_mimi.py


def make_sine(seconds, sr, freq=440.0):
    """A pure tone, for a deterministic encode/decode reference signal."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def snr_db(ref, est):
    """Signal-to-noise ratio (dB) between a reference and a reconstruction."""
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    p_ref = float(np.mean(ref ** 2)) + 1e-12
    p_noise = float(np.mean((ref - est) ** 2)) + 1e-12
    return 10.0 * math.log10(p_ref / p_noise)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] Mimi on {device}")
    codec = MimiCodec(device)
    cfg = codec.model.config
    print(f"[cfg] sampling_rate={codec.sampling_rate}  num_quantizers={cfg.num_quantizers}  "
          f"codebook_size={cfg.codebook_size}  frame_rate={cfg.frame_rate}")

    # encode / decode a 2s sine through the full 32-codebook codec
    seconds = 2.0
    audio = make_sine(seconds, codec.sampling_rate)
    codes, n_frames = codec.encode([audio])
    codes = codes[0][:, :n_frames[0]]                   # (nq, T)
    print(f"[encode] codes shape={codes.shape}  min={int(codes.min())} max={int(codes.max())}")
    assert codes.shape[0] == NUM_CODEBOOKS, codes.shape
    assert 0 <= codes.min() and codes.max() < CODEBOOK_SIZE, "code out of range"
    expected_T = round(seconds * cfg.frame_rate)
    print(f"[encode] T_frames={codes.shape[-1]} (expected ~{expected_T})")

    rec = codec.decode(codes)
    print(f"[decode] out len={rec.shape[0]}  snr_vs_input={snr_db(audio, rec):.2f} dB")
    batched_rec = codec.decode_batch([codes, codes])
    assert len(batched_rec) == 2 and all(item.shape == rec.shape for item in batched_rec)
    max_batch_delta = max(float(np.max(np.abs(item - rec))) for item in batched_rec)
    print(f"[decode-batch] 2 clips  max_delta_vs_single={max_batch_delta:.2e}")

    # codes <-> LM token ids round-trip via core/tokens.py
    layout = TokenLayout(base=100, eot=2)               # base only shifts IDs uniformly
    ids = layout.codes_to_ids(codes)
    rebuilt = layout.ids_to_codes(ids)
    assert np.array_equal(rebuilt, codes), "codes_to_ids / ids_to_codes mismatch"
    print(f"[round-trip] {len(ids)} ids ({codes.shape[-1]} frames x {NUM_CODEBOOKS} cb) OK")

    add = NUM_CODEBOOKS * CODEBOOK_SIZE + AUDIO_OFFSET
    print(f"[lm-vocab] add_tokens = {NUM_CODEBOOKS}*{CODEBOOK_SIZE}+{AUDIO_OFFSET} = {add}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
