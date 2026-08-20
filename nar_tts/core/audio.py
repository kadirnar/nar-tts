import io
import math
import threading

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel

from nar_tts.core.tokens import NUM_CODEBOOKS

MIMI_MODEL_ID = "kyutai/mimi"
_RESAMPLER_CACHE = threading.local()


class AudioLengthError(ValueError):
    """Raised internally when a clip is outside the configured duration range."""


def _resample_waveform(wav, source_sr, target_sr):
    """Resample with one cached sinc kernel per worker thread and sample-rate pair."""
    if source_sr == target_sr:
        return np.ascontiguousarray(wav, dtype=np.float32)

    cache = getattr(_RESAMPLER_CACHE, "transforms", None)
    if cache is None:
        cache = _RESAMPLER_CACHE.transforms = {}
    key = (int(source_sr), int(target_sr))
    resampler = cache.get(key)
    if resampler is None:
        # torchaudio.functional.resample rebuilds this relatively expensive sinc
        # kernel on every call. The transform stores it and is safe to reuse in
        # this thread for every clip with the same source sample rate.
        resampler = cache[key] = torchaudio.transforms.Resample(*key)

    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        return resampler(tensor).squeeze(0).numpy()


def decode_audio_bytes(audio_bytes, target_sr, min_samples=None, max_samples=None):
    """Encoded audio bytes -> mono float32 waveform resampled to ``target_sr``.

    Optional sample bounds allow callers to reject an invalid-duration clip from
    its header before decoding it. This matters for large preprocessing jobs,
    where decoding a corrupt or multi-hour row just to discard it is expensive.
    """
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as stream:
            sr = stream.samplerate
            output_frames = math.ceil(len(stream) * target_sr / sr)
            if min_samples is not None and output_frames < min_samples:
                raise AudioLengthError("audio is shorter than the configured minimum")
            if max_samples is not None and output_frames > max_samples:
                raise AudioLengthError("audio is longer than the configured maximum")
            wav = stream.read(dtype="float32", always_2d=False)
            wav = np.asarray(wav, dtype=np.float32)
    except AudioLengthError:
        raise
    except Exception:           # noqa: BLE001 - try the independent fallback backend
        t, sr = torchaudio.load(io.BytesIO(audio_bytes))
        output_frames = math.ceil(t.shape[-1] * target_sr / sr)
        if min_samples is not None and output_frames < min_samples:
            raise AudioLengthError("audio is shorter than the configured minimum")
        if max_samples is not None and output_frames > max_samples:
            raise AudioLengthError("audio is longer than the configured maximum")
        wav = t.mean(0).numpy().astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _resample_waveform(wav, sr, target_sr)
    if min_samples is not None and wav.size < min_samples:
        raise AudioLengthError("audio is shorter than the configured minimum")
    if max_samples is not None and wav.size > max_samples:
        raise AudioLengthError("audio is longer than the configured maximum")
    return np.ascontiguousarray(wav, dtype=np.float32)


def decode_audio_record(audio, text, target_sr, min_seconds=0.2, max_seconds=30.0):
    """Decode one HF audio/text row, returning ``(wave, text)`` or ``None``.

    This top-level function is deliberately pickleable so the dataset pipeline
    can run it in either a thread pool (the fast default) or a process pool.
    """
    if not audio or not audio.get("bytes") or not text:
        return None
    lo = int(min_seconds * target_sr)
    hi = int(max_seconds * target_sr)
    try:
        wav = decode_audio_bytes(audio["bytes"], target_sr, lo, hi)
    except Exception:           # noqa: BLE001 - malformed rows are intentionally skipped
        return None
    return wav, text


def load_clips(audios, texts, target_sr, min_seconds=0.2, max_seconds=30.0):
    """Decode a batch of HF audio dicts, dropping unusable / too-short / too-long.

    Returns (waves, texts) kept in parallel. `audios` are HF `{"bytes": ...}`
    dicts; rows with no audio/text or a decode error are silently skipped.
    """
    waves, kept_text = [], []
    for au, tx in zip(audios, texts):
        decoded = decode_audio_record(
            au, tx, target_sr, min_seconds=min_seconds, max_seconds=max_seconds)
        if decoded is not None:
            wav, text = decoded
            waves.append(wav)
            kept_text.append(text)
    return waves, kept_text


class MimiCodec:
    """Mimi codec with low-overhead, asynchronously transferable encode batches."""

    def __init__(self, device, model_id=MIMI_MODEL_ID, num_codebooks=NUM_CODEBOOKS,
                 dtype=None, compile_model=False, compile_mode="default",
                 allow_tf32=None, pad_to_multiple_of=None, revision=None,
                 token=None):
        self.device = torch.device(device)
        self.num_codebooks = num_codebooks
        model_kwargs = {}
        if dtype is not None:
            if isinstance(dtype, str):
                try:
                    dtype = getattr(torch, dtype)
                except AttributeError as exc:
                    raise ValueError(f"unknown torch dtype: {dtype}") from exc
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"expected a torch dtype, got {dtype!r}")
            model_kwargs["dtype"] = dtype
        if revision is not None:
            model_kwargs["revision"] = revision
        if token is not None:
            model_kwargs["token"] = token
        self.model = MimiModel.from_pretrained(model_id, **model_kwargs).to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(
            model_id, revision=revision, token=token)
        self.sampling_rate = self.fe.sampling_rate
        self.hop = self.sampling_rate / self.model.config.frame_rate   # 24000/12.5 = 1920
        self.pad_to_multiple_of = pad_to_multiple_of
        self.input_dtype = next(self.model.parameters()).dtype
        self._encode_model = self.model.encode

        if self.device.type == "cuda" and allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if compile_model:
            self.compile(mode=compile_mode)

    def compile(self, mode="default", dynamic=True):
        """Compile Mimi's neural encoder blocks for long jobs; returns ``self``.

        Compilation has a noticeable warm-up cost, so it is opt-in. ``dynamic``
        prevents a new graph for every duration bucket. The residual quantizer
        remains eager: compiling its differently shaped codebooks causes repeated
        graph recompilation and is slower than eager execution.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        for name in ("encoder", "encoder_transformer", "downsample"):
            module = getattr(self.model, name, None)
            if module is not None:
                setattr(self.model, name, torch.compile(
                    module, mode=mode, dynamic=dynamic))
        return self

    def prepare_batch(self, waves):
        """Pack variable-length mono waves into pinned CPU tensors.

        Mimi's feature extractor only converts to float32, right-pads with zero,
        and creates a padding mask. Doing that directly avoids several Python and
        NumPy conversions and permits a non-blocking host-to-device copy.
        """
        if not waves:
            raise ValueError("cannot encode an empty audio batch")
        arrays = []
        for wave in waves:
            array = np.asarray(wave, dtype=np.float32)
            if array.ndim != 1:
                raise ValueError(
                    f"expected a mono waveform, got shape {array.shape!r}")
            if not array.size:
                raise ValueError("cannot encode an empty waveform")
            arrays.append(np.ascontiguousarray(array))
        lengths = [a.size for a in arrays]
        max_length = max(lengths)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        pin = self.device.type == "cuda"
        input_values = torch.zeros(
            (len(arrays), 1, max_length), dtype=torch.float32, pin_memory=pin)
        padding_mask = torch.zeros(
            (len(arrays), max_length), dtype=torch.bool, pin_memory=pin)
        for i, (array, length) in enumerate(zip(arrays, lengths)):
            input_values[i, 0, :length].copy_(torch.from_numpy(array))
            padding_mask[i, :length] = True
        return input_values, padding_mask, lengths

    @torch.inference_mode()
    def encode_to_device(self, waves):
        """Enqueue a batch encode and return device codes plus valid frame counts.

        CUDA execution remains asynchronous. A caller can therefore batch-tokenize
        the corresponding text on CPU before transferring these codes back, which
        overlaps useful CPU work with the tail of the Mimi forward pass.
        """
        input_values, padding_mask, lengths = self.prepare_batch(waves)
        non_blocking = self.device.type == "cuda"
        input_values = input_values.to(
            self.device, dtype=self.input_dtype, non_blocking=non_blocking)
        padding_mask = padding_mask.to(self.device, non_blocking=non_blocking)
        output = self._encode_model(
            input_values, padding_mask=padding_mask, num_quantizers=self.num_codebooks)
        codes = output.audio_codes
        n_frames = [min(math.ceil(length / self.hop), codes.shape[-1])
                    for length in lengths]
        return codes, n_frames

    @staticmethod
    def codes_to_numpy(codes, dtype=None):
        """Synchronize a pending encode and copy its codes to host memory.

        Dataset writers can request ``torch.int32`` to halve device-to-host
        traffic; the default preserves the codec's native integer dtype.
        """
        codes = codes.detach()
        if dtype is not None:
            codes = codes.to(dtype=dtype)
        return codes.cpu().numpy()

    @torch.inference_mode()
    def encode(self, waves):
        """List of mono waveforms -> (codes (B, nq, Tmax), valid n_frames per clip).

        Clips are zero-padded to a common length; use the returned `n_frames` to
        trim each clip's codes back to its true duration.
        """
        codes, n_frames = self.encode_to_device(waves)
        return self.codes_to_numpy(codes), n_frames

    @torch.inference_mode()
    def encode_file(self, path):
        """Read a wav file from disk and encode it -> trimmed codes (nq, T)."""
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0),
                sr, self.sampling_rate).squeeze(0).numpy()
        codes, n_frames = self.encode([wav])
        return codes[0][:, :n_frames[0]]

    @torch.inference_mode()
    def decode(self, codes):
        """Mimi codes (nq, T) -> mono float32 waveform."""
        arr = torch.as_tensor(np.asarray(codes), dtype=torch.long, device=self.device)
        wav = self.model.decode(arr.unsqueeze(0)).audio_values
        return wav.squeeze().cpu().to(torch.float32).numpy()

    @torch.inference_mode()
    def decode_batch(self, codes):
        """Decode variable-length Mimi code arrays in one padded GPU batch.

        Mimi's decoder has no padding-mask argument. Codes are therefore
        right-padded for the batched forward and each waveform is trimmed back
        to its exact frame-derived length. The decoder is causal, so padded tail
        codes do not alter the retained prefix.
        """
        arrays = [np.asarray(item) for item in codes]
        if not arrays:
            return []
        for array in arrays:
            if array.ndim != 2 or array.shape[0] != self.num_codebooks:
                raise ValueError(
                    f"expected ({self.num_codebooks}, frames) codes, "
                    f"got {array.shape!r}")
            if array.shape[1] < 1:
                raise ValueError("cannot decode an empty code sequence")

        frame_counts = [array.shape[1] for array in arrays]
        batch = torch.zeros(
            (len(arrays), self.num_codebooks, max(frame_counts)),
            dtype=torch.long, device=self.device)
        for index, array in enumerate(arrays):
            frames = array.shape[1]
            batch[index, :, :frames] = torch.as_tensor(
                array, dtype=torch.long, device=self.device)

        audio = self.model.decode(batch).audio_values
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        elif audio.ndim != 2:
            raise RuntimeError(
                f"unexpected Mimi decoder output shape: {tuple(audio.shape)}")
        audio = audio.cpu().to(torch.float32).numpy()
        return [
            np.ascontiguousarray(
                audio[index, :min(round(frames * self.hop), audio.shape[-1])])
            for index, frames in enumerate(frame_counts)
        ]
