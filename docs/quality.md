# Quality system

## Included capabilities

Nar now uses the following workflow with a single inference configuration:

1. Turkish and English normalization for numbers, dates, currencies,
   abbreviations, and user-defined lexicons.
2. A content-addressed Mimi token cache for repeated reference audio.
3. True batched generation and a KV cache.
4. Two initial candidates, expanding to four only when the quality gate fails.
5. Whisper verification independent of the Qwen3-ASR training reward.
6. Checks for speaker similarity, CER, duration, clipping, silence, and
   repetition.
7. The winning WAV, every candidate, and a machine-readable JSON report.
8. Sentence splitting for long text, acoustic context from the previous chunk,
   and crossfading.

Default settings are embedded in the inference code. Use the optional
[`override.yaml`](../nar_tts/configs/inference/override.yaml) for persistent
changes. Best-of-N generation and verification require additional computation.
Use `real_time_factor` in the JSON report to compare speed and quality
experiments.

## Data loop

```text
raw manifest
  -> audit-data
  -> codec-check
  -> encode-expressive
  -> SFT
  -> GRPO
  -> independent evaluation + listening test
  -> hard cases / distill
  -> next SFT or GRPO round
```

`audit-data` separates corrupt and duplicate recordings, as well as recordings
with excessive silence, clipping, or a suspicious text-to-duration ratio.
Alongside `input_ids`, `encode-expressive` retains speaker, emotion, event,
source, and license fields in the Parquet output. `distill` adds only winning
Best-of-N samples that pass all thresholds to the new SFT manifest. These rows
carry `hard_case=true` and can be sampled more frequently during GRPO.

## GRPO

The single [`grpo.yaml`](../nar_tts/configs/train/grpo.yaml) normalizes the
following active components independently within each prompt group:

- Qwen3-ASR CER + ground-truth NLL
- Speaker similarity
- Duration consistency
- Technical signal quality
- Coarse prosody alignment with the reference
- Speaker drift over sliding windows

The emotion and non-verbal event reward implementations are ready, but their
weights remain zero. Do not enable them until an independently validated
classifier has been selected using synthetic Turkish speech. Using the same SER
model as both the reward and the success metric encourages reward hacking.

## Success criteria

Use the same fixed evaluation set for every model release:

- Turkish, English, and Japanese CER and WER
- Speaker similarity and long-form speaker drift
- p50/p95 RTF, VRAM, and time to first chunk
- Clipping, silence, repetition, and truncation rates
- Emotion accuracy, event F1, and position error
- Blinded human A/B tests for naturalness, emotion, and speaker identity

`technical_quality` is a signal diagnostic, not a MOS or naturalness model. An
emotion-classifier score alone is not evidence of product quality.

## Changes that require retraining

Emotion markup does not add new capabilities to an existing checkpoint.
Crying speech, speech-laugh, laughter, and sobbing require labeled expressive
SFT data. If the codec, speech-token layout, or special control tokens change,
all speech data must be encoded again and the model must be retrained. For this
reason, alternative codecs, a new decoder, and true frame-level streaming do
not apply automatically to the current checkpoint; evaluate them as a separate
model generation.
