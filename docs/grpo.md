# GRPO post-training for Nar TTS

Nar now supports end-to-end speech GRPO after causal speech-token pretraining.
The policy samples Mimi tokens, Nar decodes each completion to a waveform, frozen
speech models score the result, and TRL applies group-relative policy updates.
The implementation supports a single GPU and distributed groups spanning
multiple GPUs.

## Research basis

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced Group Relative Policy
Optimization as a critic-free PPO variant: multiple completions for the same
prompt provide the baseline used to normalize advantages. This removes the
separate value model, which is particularly useful in TTS because a codec and
one or more speech reward models already consume substantial memory.

[Group Relative Policy Optimization for Text-to-Speech with Large Language
Models](https://arxiv.org/abs/2509.18798) demonstrated the direct speech loop
implemented here: sample speech tokens, reconstruct waveforms, and use an
off-the-shelf ASR model for rewards. It combines transcription character error
rate and teacher-forced ground-truth ASR negative log-likelihood:

```text
R_error = 1 - tanh(alpha_error * error_rate)
R_nll   = exp(-NLL / alpha_nll)
R_intel = (lambda_error + lambda_nll)
          / (lambda_error / R_error + lambda_nll / R_nll)
```

The paper used Whisper-large-v3, `alpha_error = alpha_nll = 3`, component
weights `0.6/0.4`, `G = 8`, KL coefficient `0.1`, learning rate `1e-5`, and a
4,000-sentence alignment set. Its combined reward improved objective
intelligibility and subjective naturalness while largely preserving speaker
similarity.

Subsequent TTS work motivates the optional rewards included in Nar:

- [F5R-TTS](https://arxiv.org/abs/2504.02407) and
  [ASRRL-TTS](https://arxiv.org/abs/2407.05421) use intelligibility and speaker
  similarity signals.
- [Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at
  Scale](https://arxiv.org/abs/2511.21270) adds speaker similarity, duration
  consistency, entropy control, and prosody alignment.
- [GLASS](https://arxiv.org/abs/2606.05889) trains small LoRA acoustic
  directions with a WER anchor plus speed or pitch rewards. It also shows why
  style adapters are useful after a general quality-alignment stage.

These newer results are preprints. Treat their hyperparameters as starting
points and verify improvements with held-out ASR, speaker, and listening tests.

## Nar-specific design

Nar does not generate a single semantic codebook. Every audio frame is 32
consecutive Mimi actions, one from each codebook. Unconstrained text generation
can select an impossible codebook and make the rest of a rollout undecodable.
The GRPO trainer therefore uses a strict action grammar:

1. Position `t mod 32` can select only that position's 2,048 Mimi entries.
2. `EOS_SPEECH` is available only between complete frames and after the minimum
   duration.
3. At the maximum duration, `EOS_SPEECH` is forced; no partial frame is left.
4. Policy, old-policy, and reference log-probabilities use the same constrained
   denominator as sampling. Invalid text-vocabulary logits are not part of the
   speech action distribution.

The inference engine applies this same grammar, so a GRPO adapter is evaluated
under the policy it was actually trained to optimize.

This is also more efficient than applying a full-vocabulary softmax for reward
optimization. The language-model head still produces logits, but normalization
touches only one 2,048-entry codebook slice per position.

The reward path is fused around waveform decoding. A rollout batch is decoded
once with Mimi; that waveform batch is reused by ASR, speaker verification,
duration, and speed rewards. Reference speaker embeddings are cached, and Mimi
and ASR inference are batched.

Nar intentionally standardizes intelligibility rewards on the native
[Qwen3-ASR-1.7B Transformers checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf).
The smaller 0.6B checkpoint and Whisper are rejected by config validation. The
1.7B model supports batched multilingual transcription, automatic/forced
language handling, and target-text likelihood through its training labels. Its
native integration requires Transformers 5.13 or newer; optional static-cache
compilation is enabled in the quality recipes.

Speaker preservation defaults to ESPnet's
[WavLM-Large + ECAPA joint model](https://huggingface.co/espnet/voxcelebs12_ecapa_wavlm_joint),
not WavLM Base Plus. Microsoft's official downstream comparison reports that
[fine-tuned WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm#speaker-verification)
substantially outperforms the other large SSL baselines and reaches 0.33%
VoxCeleb1-O EER with large-margin fine-tuning and score calibration. The
directly loadable ESPnet WavLM-Large + ECAPA checkpoint is the reproducible
runtime choice here; its model card reports 0.394% EER and 0.03797 minDCF. Nar
feeds it 16 kHz audio and caches normalized 192-D reference embeddings. The
older Transformers Base-Plus x-vector path remains available only as an
explicitly selected compatibility backend.

Install that quality speaker stack before using a speaker-enabled recipe:

```bash
pip install -e ".[speaker-quality]"
```

[TRL's GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) supplies the
distributed optimizer, clipped objective, group gathering, reference KL,
checkpointing, and DAPO length normalization. Nar owns direct token prompts,
speech generation, constrained log-probabilities, and rewards. With LoRA, TRL
computes the frozen reference policy by disabling the adapter, so a second copy
of the Nar model is not loaded.

## Post-pretraining scenarios

| Stage | Checked-in config | Purpose | Typical hardware |
|---|---|---|---|
| Supervised adaptation | `finetune.yaml` | High-quality voice/domain or language continuation | existing FSDP setup |
| GRPO intelligibility | `grpo_intelligibility.yaml` | CER + mean ASR-NLL, gentle duration anchor | 1 GPU, G=4 |
| Text-only streaming GRPO | `grpo_text_streaming.yaml` | ASR alignment without paired audio | 1 GPU, G=4 |
| GRPO multi-reward | `grpo_multireward.yaml` | Paper CER + summed NLL, WavLM speaker and duration | 8 GPUs, G=8 |
| GRPO style LoRA | `grpo_style_fast.yaml` / `grpo_style_slow.yaml` | Opposite speed directions with ASR and speaker anchors | 1 GPU, G=4 |
| vLLM rollout | `grpo_vllm.yaml` | Colocated/server continuous batching with Nar grammar | large 1 GPU or rollout GPUs |
| SGLang rollout | `grpo_sglang.yaml` | External server plus live LoRA synchronization | trainer GPU(s) + rollout GPU(s) |
| Unsloth LoRA SFT | `finetune_unsloth.yaml` | Efficient supervised voice/domain adaptation | 1 GPU, separate environment |
| LLaMA-Factory text SFT | `llama_factory/language_retention_sft.yaml` | Preserve/adapt language capability after speech pretraining | separate environment |

The recommended order is:

```text
speech pretraining -> optional supervised adaptation -> general GRPO quality
                   -> optional independent style LoRA adapters
```

The checked-in paths are editable placeholders. When chaining two LoRA stages,
merge the first adapter into a full checkpoint (or otherwise load it as the
next stage's base) before creating the next independent adapter. Saved Nar
tokenizers are reloadable directly; the loader preserves the original text EOT
while using `EOS_SPEECH` for rollouts.

Use the checkpoint directory itself for `model.tokenizer`. New pretraining and
supervised checkpoints save the expanded tokenizer alongside the weights. This
is mandatory for vLLM, which initializes its engine from `model.name_or_path`
and must discover exactly the same custom-token IDs there.

Run supervised adaptation from its config:

```bash
accelerate launch --config_file nar_tts/configs/launch/fsdp.yaml \
  -m nar_tts.training.finetune --config nar_tts/configs/finetune.yaml
```

Validate a GRPO scenario without loading models or data:

```bash
python -m nar_tts.training.grpo \
  --config nar_tts/configs/grpo_intelligibility.yaml --validate-only
```

Run the one-GPU alignment scenario:

```bash
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
  -m nar_tts.training.grpo \
  --config nar_tts/configs/grpo_intelligibility.yaml
```

Run the G=8 multi-GPU scenario:

```bash
accelerate launch --config_file nar_tts/configs/launch/multi_gpu.yaml \
  -m nar_tts.training.grpo \
  --config nar_tts/configs/grpo_multireward.yaml
```

All model paths, datasets, generation bounds, rewards, optimizer settings,
logging, and output paths live in YAML. No environment-variable block or source
edit is required. Authentication continues to use the credential saved by
`hf auth login`.

## Rollout and training backends

`rollout.backend` selects `transformers`, `vllm`, or `sglang`. All three use the
same backend-neutral audio mask: codebook ranges, legal frame-boundary EOS, and
the forced maximum-duration terminator are identical. Nar still recomputes
policy/reference log-probabilities locally over the constrained action space.

### vLLM

Install the checkout and vLLM together so vLLM discovers Nar's processor from
the `vllm.logits_processors` package entry point:

```bash
pip install -e ".[vllm]"
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
  -m nar_tts.training.grpo --config nar_tts/configs/grpo_vllm.yaml
```

The checked-in scenario uses TRL's colocated engine, sleep mode, live weight
synchronization, and importance-sampling correction. Change
`rollout.vllm.mode` to `server` and set its server fields for a dedicated
rollout service. The same editable Nar installation must exist in that server
environment so the plugin is loaded at engine startup.

### SGLang

SGLang runs as a trusted external service with custom-logit processing and
dynamic LoRA loading. The config is the sole source for host, port, tensor
parallelism, LoRA limits, server arguments, and the shared adapter directory.
On the rollout host:

```bash
pip install -e ".[sglang]"
python -m nar_tts.integrations.sglang_server \
  --config nar_tts/configs/grpo_sglang.yaml
```

Then launch training with the same GRPO config. Rank zero saves the current LoRA
once per optimizer step, asks SGLang to reload it, broadcasts any synchronization
error, and every rank submits its local prompt batch. The server launcher enables
LoRA, the custom processor, raw token-ID mode, and the configured tensor-parallel
size. Because SGLang deserializes a custom processor, expose this endpoint only
on a trusted network.

### Unsloth and LLaMA-Factory

Unsloth is available for efficient one-GPU LoRA supervised post-training in
`finetune_unsloth.yaml`. Its loader verifies the expanded Nar vocabulary, uses
Unsloth gradient checkpointing and rsLoRA, and deliberately leaves fast
inference off. Install and run it in a separate environment:

```bash
python -m venv .venv-unsloth
source .venv-unsloth/bin/activate
pip install -e ".[unsloth]"
accelerate launch --config_file nar_tts/configs/launch/single_gpu.yaml \
  -m nar_tts.training.finetune \
  --config nar_tts/configs/finetune_unsloth.yaml
```

There is a current upstream dependency boundary: native Qwen3-ASR needs
Transformers >=5.13, while the current
[Unsloth dependency range](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)
caps Transformers below that and
[LLaMA-Factory's range](https://github.com/hiyouga/LlamaFactory/blob/main/pyproject.toml)
also precedes 5.13. Do not downgrade the core Nar environment or silently swap
in a smaller ASR. Run those integrations in separate environments. Use the
Transformers/vLLM/SGLang scenarios for Qwen3-ASR quality alignment until the
upstream ranges converge; Nar rejects `model.loader: unsloth` in a GRPO config
instead of constructing an unsupported mixed environment.

The official [LLaMA-Factory project](https://github.com/hiyouga/LlamaFactory)
does not currently expose GRPO in its documented training stages,
so Nar integrates it honestly as a post-pretraining text/language SFT launcher,
not as a codec-token GRPO backend. Put an Alpaca-format
`data/llama_factory/language_retention.jsonl` in place, then run:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
pip install -e ./LlamaFactory
pip install -e .
python -m nar_tts.integrations.llama_factory \
  --config nar_tts/configs/llama_factory/language_retention_sft.yaml \
  --validate-only
python -m nar_tts.integrations.llama_factory \
  --config nar_tts/configs/llama_factory/language_retention_sft.yaml
```

The launcher resolves both `preprocessing_num_workers: auto` and
`dataloader_num_workers: auto` from the process CPU affinity before invoking
`llamafactory-cli train`; distributed dataloader workers are divided across
ranks so the machine is used without multiplying its CPU count per GPU. For
speech-codec SFT and all GRPO scenarios, keep using Nar's native trainers.

## Dataset modes

The trainer can stream map-style or iterable Hugging Face datasets. Iterable
training requires a positive `training.max_steps`.

- `tts_tokens`: consumes the existing pretraining Parquet `input_ids`. Nar
  splits at `SOS`, decodes the transcript from the prompt, and reuses the target
  Mimi tokens as a speaker/duration reference. No raw waveform is read.
- `text`: consumes only a transcript column. This is sufficient for ASR-based
  intelligibility GRPO and follows the 4,000-text setup in the TTS paper.
- `prompt_ids`: consumes an already constructed Nar prompt plus target text.
- `voice_clone_tokens`: consumes target text, reference transcript, and flattened
  reference Mimi IDs. Nar builds the same zero-shot continuation prompt used by
  inference and can apply speaker preservation.

Set `dataset.on_invalid: drop` to discard malformed pretraining rows, or `error`
to stop at the first schema problem. `--validate-only` checks required column
mappings before allocating a model. Nar projects the source to configured
columns before streaming, so a text-only run does not decode or retain the raw
audio column.

## Reward configuration

`rewards.weights` combines the active fused components:

- `intelligibility`: CER/WER and optional ASR-NLL harmonic mean.
- `speaker`: cosine similarity between normalized WavLM speaker embeddings.
- `duration`: binary paper-style tolerance or a smooth log-ratio reward.
- `speed`: a monotonic fast/slow style direction, anchored by the other rewards.
- `format`: structural validity; constrained generation normally makes this
  constant, while invalid/truncated samples still receive `invalid_reward`.

Use `asr.metric: auto` with a populated language column to select WER for English
and CER for other languages. `nll_reduction: sum` follows the TTS paper's stated
equation; `mean` is less sensitive to transcript length and is the practical
single-GPU default. Reward components and raw error/NLL values are logged
separately even though decoding and inference are fused.

## Performance and memory guidance

For one GPU, start with the checked-in G=4 LoRA recipe. It generates one group
in a batch but optimizes microbatches of one through gradient accumulation.
Reduce `generation.max_audio_seconds` first when KV-cache or completion-logit
memory is too high, then reduce generation/ASR batch sizes or disable the
optional speaker reward. Do not replace Qwen3-ASR-1.7B with a smaller reward
model. Keeping `beta > 0` with PEFT costs no second policy copy; a full-model
GRPO run with KL does require a separate frozen reference.

On multiple GPUs, the group can span ranks. TRL requires
`world_size * per_device_train_batch_size * gradient_accumulation_steps` to be
divisible by `num_generations`; Nar validates this before loading weights. Each
rank loads its own frozen reward models, avoiding reward-server contention.
Change `num_processes` and `runtime.expected_world_size` together.

For maximum rollout throughput, use vLLM colocate/server or a dedicated SGLang
server. Both now apply Nar's custom position-dependent logits processor.
Transformers continuous batching remains disabled because it does not expose
the same request-state hook. Regular Transformers generation remains fully
batched and distributed and is the lowest-complexity one-GPU path.

## Evaluation cautions

An ASR reward can be exploited by audio that is easy for the reward model but
unnatural to people. Evaluate with a different held-out ASR model, measure
speaker similarity independently, inspect duration distributions, and run MOS
or preference tests. The WavLM-Large + ECAPA checkpoint is trained from
VoxCeleb data, so cross-lingual speaker results still need extra scrutiny.
Always retain the pre-GRPO checkpoint and compare it on a fixed multilingual
suite.
