# Nar TTS: Qwen3 + Mimi Text-to-Speech

## Overview

Nar TTS converts text to speech by having a Qwen3 LM generate interleaved
[Mimi](https://huggingface.co/kyutai/mimi) audio-codec tokens, which are decoded
back to a waveform. Give it a reference clip and it clones that voice in any
language (EN/JA). Built on the Orpheus recipe.

## Installation

```bash
git clone https://github.com/kadirnar/nar-tts.git
cd nar-tts
pip install -r requirements.txt            # Python 3.10, CUDA 12.x
pip install -e .                           # installs Nar + vLLM plugin metadata
hf auth login                               # once, for private pulls/pushes
```

Multi-reward speaker scoring uses the higher-quality WavLM-Large + ECAPA model:

```bash
pip install -e ".[speaker-quality]"
```

## Inference

```python
from nar_tts.inference.infer import NarTTS

engine = NarTTS(
    checkpoint="checkpoints/checkpoint-171622",
    tokenizer_name="Qwen/Qwen3-0.6B",
    device="cuda:0",
)

audio = engine.clone(
    text="Hello, this is a cloned voice.",
    ref_wav="andrew.wav",
    ref_text="all sorts of things, mainly browsing.",
    output_path="output.wav",
)
```

Batch synthesis — edit `CKPT` and `JOBS` at the top of the script, then:

```bash
python -m nar_tts.inference.infer
```

## Dataset Preparation

Encode raw (audio, text) into `input_ids` parquet. The smaller encoders keep
their source settings near the top of each script. The streaming pretraining
encoder reads every run setting from
[`nar_tts/configs/preprocess_pretrain.yaml`](nar_tts/configs/preprocess_pretrain.yaml):

```bash
python -m nar_tts.preprocessing.encode_textqa     # text-QA stream
python -m nar_tts.preprocessing.encode_pretrain   # speech (TTS) stream
python -m nar_tts.preprocessing.encode_finetune   # single-voice set
```

The speech encoders default to one GPU with a parallel CPU decode/resample pool,
duration-bucketed GPU batches, batched text tokenization, and asynchronous
Parquet output. In the pretraining config, `runtime.cpu_workers: auto` detects
the process's CPU affinity and activates every available logical CPU. To keep a
machine-specific config elsewhere, pass it explicitly:

```bash
python -m nar_tts.preprocessing.encode_pretrain --config /path/to/preprocess.yaml
```

### Multi-terabyte streaming run

`encode_pretrain` can download, tokenize, and push concurrently without first
storing the full source or encoded dataset. Edit the checked-in YAML once—at a
minimum, set `source.repo`, `target.repo`, `target.output_root`, and optionally
`target.download_root`. Keep `target.push_to_hub: true`,
`runtime.num_gpus: 1`, and `runtime.cpu_workers: auto`. Then the complete run is:

```bash
python -m nar_tts.preprocessing.encode_pretrain
```

The run resolves the source, tokenizer, and Mimi branches to immutable commits
and records them with the output-affecting settings in `run-state.json`.
Restarts reuse those exact snapshots and reject silent setting drift. Optional
`source.revision`, `tokenizer.revision`, and `mimi.revision` values can pin them
explicitly from the first run. To intentionally move to a changed snapshot or
accept changed encoding settings, use a new output root (safest) or set
`resume.reset_run_state: true`. The target repo is checked before downloading
anything and receives the same manifest as
`nar-tts-run-state.json`; a fresh machine cannot silently resume an incompatible
target. Use a new target repo when changing tokenization. For an intentionally
append-only source update, set both `resume.reset_run_state: true` and
`resume.accept_remote_state_change: true`; this accepts the new manifests while
continuing to reuse existing remote shard paths. Shards already present in a
compatible target are skipped.

One future source shard downloads while the current shard uses the CPU/GPU.
Completed outputs upload in parallel through `hf-xet`; commits are batched,
retried, checked for remote path and byte size, and only then removed locally.
Restarting the same command resumes from both remote commits and durable local
outputs. Old flat outputs produced by earlier versions are also recognized and
uploaded without re-tokenization. To resume a legacy target that already has
Parquet shards but predates the remote manifest, first verify its tokenizer and
Mimi settings, then set `resume.adopt_legacy_target: true`.

If the 2 TB source is already present, set `source.local_root` to its repository
root (the directory containing paths such as `data/train-....parquet`). Those
files are borrowed in place and never deleted. Set it to `null` to stage remote
shards on demand.

Local disk use is bounded: source storage is roughly the current shard plus
`runtime.download_prefetch` future shards, while not-yet-verified outputs are
limited by `transfer.max_upload_staging_gb` plus the shard being written. Upload
staging uses hardlinks, not another copy of each Parquet file. Disable
`transfer.delete_downloaded_source` or `transfer.delete_after_upload` only when
you intentionally want those files retained. Failed downloaded shards are also
removed by default so repeated data errors cannot fill the disk; enable
`transfer.keep_failed_downloads` when retaining them for debugging is worth
that space.

Transfer tuning controls are `transfer.upload_batch_files`,
`transfer.upload_batch_gb`, `transfer.upload_flush_seconds`, and
`transfer.attempts`. New output paths are distributed over 256 Hub
subdirectories by default so large shard counts remain below the per-folder
limit; the dataset card uses `data/**/*.parquet`. Ensure the target account has
enough [Hub storage](https://huggingface.co/docs/hub/storage-limits) before
starting a multi-terabyte push.

`tokenization.max_batch_seconds` limits
`gpu_batch_size * longest_clip_seconds`, which is the padded workload Mimi
actually sees. Tune it before a fresh run: lower it after an out-of-memory
trial, or raise it while GPU memory and utilization leave headroom. Changing it
after `run-state.json` is created requires the explicit state-reset/new output
root flow described above. For very long jobs, `mimi.compile: true` enables
dynamic `torch.compile` after a one-time warm-up. `mimi.dtype: bfloat16` is
faster on supported GPUs but can change codec ids; leave it `null` when
reproducing a float32-encoded dataset. Disable `mimi.allow_tf32` when float32
reproducibility matters. Exact re-encoding also requires the same GPU
batch/bucket settings, because a padded clip's final Mimi frame can depend on
the batch's padded length.

The CPU-only QA encoder uses persistent process-local tokenizers and native
batch encoding. Tune it with `NAR_TTS_TEXT_WORKERS`, `NAR_TTS_TOKENIZER_THREADS`, and
`NAR_TTS_TEXT_BATCH_SIZE`; its automatic defaults divide the available CPU threads
without nesting an unrestricted tokenizer pool inside every process.

Sanity-check the codec and token round-trip: `python -m tests.test_mimi`.

## Training

```bash
# Pre-training (8-GPU FSDP; knobs in configs/pretrain.yaml)
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
                  -m nar_tts.training.pretrain

# Supervised post-training on a voice/domain (all knobs in configs/finetune.yaml)
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
                  -m nar_tts.training.finetune \
                  --config nar_tts/configs/finetune.yaml

# Efficient one-GPU LoRA SFT in a separate Unsloth environment
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
                  -m nar_tts.training.finetune \
                  --config nar_tts/configs/finetune_unsloth.yaml

# One-GPU GRPO: Qwen3-ASR-1.7B intelligibility + NLL + duration
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
                  -m nar_tts.training.grpo \
                  --config nar_tts/configs/grpo_intelligibility.yaml

# Eight-GPU G=8 GRPO: Qwen3-ASR + WavLM-Large speaker + duration
accelerate launch --config_file nar_tts/configs/launch/multi_gpu.yaml \
                  -m nar_tts.training.grpo \
                  --config nar_tts/configs/grpo_multireward.yaml
```

GRPO rollouts are constrained to the correct 32-codebook Mimi grammar and use
the same action space for policy/reference log-probabilities. Existing
pretraining Parquet can be consumed directly without raw audio; text-only,
voice-cloning, streaming, LoRA style, one-GPU, and multi-GPU scenarios are also
supported. Transformers, vLLM, and SGLang rollout backends share that grammar;
Unsloth LoRA SFT and a separate LLaMA-Factory SFT stage are included as well.
See [the GRPO research and post-training guide](docs/grpo.md).

## Acknowledgements

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) 
- [Kyutai Mimi](https://huggingface.co/kyutai/mimi)
- [Qwen3](https://huggingface.co/Qwen)
