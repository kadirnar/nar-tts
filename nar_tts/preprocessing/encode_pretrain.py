"""Stream a Hub audio dataset through a selected tokenizer and Mimi.

For the intended one-GPU run, three bounded stages overlap continuously:

    download next shard -> CPU decode + text/Mimi tokenize -> upload prior shards

No stage needs the complete source or encoded dataset on local disk. Source
downloads use an owned ``local_dir`` instead of the global Hugging Face cache,
and encoded files are retired only after a verified, atomic Hub commit.
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import HfApi
from packaging.version import Version
from transformers import AutoTokenizer

from nar_tts.core.audio import MIMI_MODEL_ID, MimiCodec
from nar_tts.core.tokens import TokenLayout
from nar_tts.preprocessing.fast_pipeline import (
    ParallelAudioDecoder,
    encode_audio_parquet,
)
from nar_tts.preprocessing.hub_pipeline import (
    HubShardDownloader,
    StreamingHubUploader,
    cleanup_stale_uploads,
    exclusive_file_lock,
    sharded_repo_path,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "train" / "preprocess.yaml"
)
REMOTE_STATE_PATH = "nar-tts-run-state.json"
ACTIVE_CONFIG_PATH = None


def available_cpu_count():
    """Return the logical CPUs this process is actually allowed to use."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, count)


def _section(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"config section {name!r} must be a mapping")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value, name):
    if value is None or value == "":
        return None
    return _string(value, name)


def _boolean(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be true or false")
    return value


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if parsed < minimum or (maximum is not None and parsed > maximum):
        limit = (f"between {minimum} and {maximum}" if maximum is not None
                 else f"at least {minimum}")
        raise ValueError(f"{name} must be {limit}")
    return parsed


def _floating(value, name, minimum=0.0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _auto_integer(value, automatic, name, minimum=1):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return max(minimum, int(automatic))
    return _integer(value, name, minimum=minimum)


def _path(value, name, config_directory, optional=False):
    if value is None or value == "":
        if optional:
            return None
        raise ValueError(f"{name} must be a path")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return str(path.resolve())


def load_config(path=DEFAULT_CONFIG_PATH):
    """Load and validate the outer YAML document."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except OSError as exc:
        raise RuntimeError(f"cannot read preprocessing config {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in preprocessing config {config_path}") from exc
    if not isinstance(config, dict):
        raise TypeError("preprocessing config must contain a YAML mapping")
    return config_path, config


def configure(path=DEFAULT_CONFIG_PATH):
    """Load one preprocessing config and publish its validated run settings."""
    global ACCEPT_REMOTE_STATE_CHANGE, ACTIVE_CONFIG_PATH, ADOPT_LEGACY_TARGET
    global ALLOW_TF32, BATCH_SIZE, BUCKET_SIZE, CPU_COUNT, CPU_WORKERS
    global DECODE_BACKEND, DECODE_PREFETCH, DELETE_AFTER_UPLOAD
    global DELETE_DOWNLOADED_SOURCE, DOWNLOAD_PREFETCH, DOWNLOAD_ROOT
    global HUB_PREFIX_CHARS, KEEP_FAILED_DOWNLOADS, LOCAL_SRC_ROOT, LOG_PATH
    global MAX_AUDIO_SECONDS, MAX_BATCH_SECONDS, MAX_UPLOAD_STAGING_GB
    global MIMI_COMPILE, MIMI_COMPILE_MODE, MIMI_DTYPE, MIMI_MODEL
    global MIMI_REVISION, MIN_AUDIO_SECONDS, NUM_GPUS, OUTPUT_DIR, OUTPUT_ROOT
    global PUSH_TO_HUB, READ_BATCH_SIZE, RESET_RUN_STATE, ROW_GROUP_SIZE
    global RUN_STATE_PATH, SOURCE_REVISION, SRC_REPO, TARGET_PRIVATE, TARGET_REPO
    global TOKEN, TOKENIZER_NAME, TOKENIZER_REVISION, TOKENIZER_THREADS
    global TRANSFER_ATTEMPTS, UPLOAD_BATCH_FILES, UPLOAD_BATCH_GB
    global UPLOAD_FLUSH_SECONDS, UPLOAD_STAGING_ROOT

    config_path, config = load_config(path)
    source = _section(config, "source")
    target = _section(config, "target")
    hub = _section(config, "hub")
    runtime = _section(config, "runtime")
    tokenizer = _section(config, "tokenizer")
    mimi = _section(config, "mimi")
    tokenization = _section(config, "tokenization")
    transfer = _section(config, "transfer")
    resume = _section(config, "resume")

    SRC_REPO = _string(source.get("repo"), "source.repo")
    SOURCE_REVISION = _optional_string(
        source.get("revision"), "source.revision")
    LOCAL_SRC_ROOT = _path(
        source.get("local_root"), "source.local_root", config_path.parent,
        optional=True)

    TARGET_REPO = _string(target.get("repo"), "target.repo")
    PUSH_TO_HUB = _boolean(target.get("push_to_hub"), "target.push_to_hub")
    TARGET_PRIVATE = _boolean(target.get("private"), "target.private")
    OUTPUT_ROOT = _path(
        target.get("output_root"), "target.output_root", config_path.parent)
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "data")
    configured_download_root = target.get("download_root")
    if configured_download_root is None or configured_download_root == "":
        DOWNLOAD_ROOT = os.path.join(OUTPUT_ROOT, ".downloads")
    else:
        DOWNLOAD_ROOT = _path(
            configured_download_root, "target.download_root", config_path.parent)
    UPLOAD_STAGING_ROOT = os.path.join(OUTPUT_ROOT, ".upload_batches")
    LOG_PATH = os.path.join(OUTPUT_ROOT, "encode.log")
    RUN_STATE_PATH = os.path.join(OUTPUT_ROOT, "run-state.json")

    TOKEN = _optional_string(hub.get("token"), "hub.token")

    CPU_COUNT = available_cpu_count()
    NUM_GPUS = _integer(runtime.get("num_gpus"), "runtime.num_gpus")
    CPU_WORKERS = _auto_integer(
        runtime.get("cpu_workers"), CPU_COUNT, "runtime.cpu_workers")
    TOKENIZER_THREADS = _integer(
        runtime.get("tokenizer_threads"), "runtime.tokenizer_threads")
    DECODE_BACKEND = _string(
        runtime.get("decode_backend"), "runtime.decode_backend").lower()
    if DECODE_BACKEND not in {"thread", "process"}:
        raise ValueError("runtime.decode_backend must be 'thread' or 'process'")
    DECODE_PREFETCH = _auto_integer(
        runtime.get("decode_prefetch"), max(4, CPU_WORKERS * 4),
        "runtime.decode_prefetch")
    DOWNLOAD_PREFETCH = _integer(
        runtime.get("download_prefetch"), "runtime.download_prefetch")

    TOKENIZER_NAME = _string(tokenizer.get("model"), "tokenizer.model")
    TOKENIZER_REVISION = _optional_string(
        tokenizer.get("revision"), "tokenizer.revision")
    MIMI_MODEL = _string(mimi.get("model", MIMI_MODEL_ID), "mimi.model")
    MIMI_REVISION = _optional_string(mimi.get("revision"), "mimi.revision")
    MIMI_DTYPE = _optional_string(mimi.get("dtype"), "mimi.dtype")
    MIMI_COMPILE = _boolean(mimi.get("compile"), "mimi.compile")
    MIMI_COMPILE_MODE = _string(
        mimi.get("compile_mode"), "mimi.compile_mode")
    ALLOW_TF32 = _boolean(mimi.get("allow_tf32"), "mimi.allow_tf32")

    BATCH_SIZE = _integer(
        tokenization.get("gpu_batch_size"), "tokenization.gpu_batch_size")
    BUCKET_SIZE = _integer(
        tokenization.get("bucket_size"), "tokenization.bucket_size")
    MAX_BATCH_SECONDS = _floating(
        tokenization.get("max_batch_seconds"),
        "tokenization.max_batch_seconds")
    READ_BATCH_SIZE = _integer(
        tokenization.get("read_batch_size"), "tokenization.read_batch_size")
    ROW_GROUP_SIZE = _integer(
        tokenization.get("row_group_size"), "tokenization.row_group_size")
    MIN_AUDIO_SECONDS = _floating(
        tokenization.get("min_audio_seconds"),
        "tokenization.min_audio_seconds")
    MAX_AUDIO_SECONDS = _floating(
        tokenization.get("max_audio_seconds"),
        "tokenization.max_audio_seconds")
    if MAX_AUDIO_SECONDS <= MIN_AUDIO_SECONDS:
        raise ValueError(
            "tokenization.max_audio_seconds must exceed min_audio_seconds")

    TRANSFER_ATTEMPTS = _integer(
        transfer.get("attempts"), "transfer.attempts")
    DELETE_DOWNLOADED_SOURCE = _boolean(
        transfer.get("delete_downloaded_source"),
        "transfer.delete_downloaded_source")
    KEEP_FAILED_DOWNLOADS = _boolean(
        transfer.get("keep_failed_downloads"),
        "transfer.keep_failed_downloads")
    DELETE_AFTER_UPLOAD = _boolean(
        transfer.get("delete_after_upload"), "transfer.delete_after_upload")
    HUB_PREFIX_CHARS = _integer(
        transfer.get("hub_prefix_chars"), "transfer.hub_prefix_chars",
        minimum=0, maximum=64)
    UPLOAD_BATCH_FILES = _integer(
        transfer.get("upload_batch_files"), "transfer.upload_batch_files")
    UPLOAD_BATCH_GB = _floating(
        transfer.get("upload_batch_gb"), "transfer.upload_batch_gb",
        minimum=0.001)
    UPLOAD_FLUSH_SECONDS = _floating(
        transfer.get("upload_flush_seconds"), "transfer.upload_flush_seconds")
    staging_gb = transfer.get("max_upload_staging_gb")
    if isinstance(staging_gb, str) and staging_gb.strip().lower() == "auto":
        staging_gb = max(24.0, UPLOAD_BATCH_GB * 3)
    MAX_UPLOAD_STAGING_GB = _floating(
        staging_gb, "transfer.max_upload_staging_gb", minimum=0.001)
    if MAX_UPLOAD_STAGING_GB < UPLOAD_BATCH_GB:
        raise ValueError(
            "transfer.max_upload_staging_gb must be at least upload_batch_gb")

    RESET_RUN_STATE = _boolean(
        resume.get("reset_run_state"), "resume.reset_run_state")
    ACCEPT_REMOTE_STATE_CHANGE = _boolean(
        resume.get("accept_remote_state_change"),
        "resume.accept_remote_state_change")
    ADOPT_LEGACY_TARGET = _boolean(
        resume.get("adopt_legacy_target"), "resume.adopt_legacy_target")
    ACTIVE_CONFIG_PATH = str(config_path)


configure()

_LOG_LOCK = threading.Lock()


def _log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + msg
    with _LOG_LOCK:
        print(line, flush=True)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _output_identity():
    """Settings that can change token IDs or deterministic output names."""
    return {
        "source_repo": SRC_REPO,
        "tokenizer": TOKENIZER_NAME,
        "mimi_model": MIMI_MODEL,
        "mimi_dtype": MIMI_DTYPE or "float32",
        "mimi_compile": MIMI_COMPILE,
        "mimi_compile_mode": MIMI_COMPILE_MODE,
        "allow_tf32": ALLOW_TF32,
        "batch_size": BATCH_SIZE,
        "bucket_size": BUCKET_SIZE,
        "max_batch_seconds": MAX_BATCH_SECONDS,
        "min_audio_seconds": MIN_AUDIO_SECONDS,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "hub_prefix_chars": HUB_PREFIX_CHARS,
    }


def _read_run_state():
    if RESET_RUN_STATE or not os.path.isfile(RUN_STATE_PATH):
        return None
    try:
        with open(RUN_STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read run state {RUN_STATE_PATH}") from exc


def _write_run_state(state):
    tmp_path = RUN_STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, RUN_STATE_PATH)


def _make_run_state(source_revision, tokenizer_revision, mimi_revision):
    return {
        "version": 2,
        "source_revision": source_revision,
        "tokenizer_revision": tokenizer_revision,
        "mimi_revision": mimi_revision,
        "output_identity": _output_identity(),
    }


def _validate_remote_state(remote_state, current_state):
    if remote_state == current_state:
        return
    if ACCEPT_REMOTE_STATE_CHANGE:
        _log(
            "WARNING: accepting an intentional target preprocessing-state "
            "change; existing remote shard paths will still be treated as complete")
        return
    raise RuntimeError(
        f"{TARGET_REPO}/{REMOTE_STATE_PATH} does not match this run. Use a "
        "new target repo (safest), or set "
        "resume.accept_remote_state_change: true in the config only when "
        "reusing existing shards is intentional")


def list_shards():
    """Return durable source/model snapshots and all Parquet shard names."""
    identity = _output_identity()
    state = _read_run_state()
    if state is not None:
        previous_identity = state.get("output_identity")
        if previous_identity != identity:
            changed = sorted(
                key for key in set(previous_identity or {}) | set(identity)
                if (previous_identity or {}).get(key) != identity.get(key))
            raise RuntimeError(
                "output-affecting settings changed since this run started "
                f"({', '.join(changed)}). Restore them, use a new output root, "
                "or set resume.reset_run_state: true in the config to accept "
                "an intentional change")

    api = HfApi(token=TOKEN)

    def resolve(repo_id, repo_type, requested_revision, state_key):
        if requested_revision is None and state is not None:
            requested_revision = state.get(state_key)
        info = api.repo_info(
            repo_id, repo_type=repo_type, revision=requested_revision,
            token=TOKEN)
        revision = info.sha
        previous = None if state is None else state.get(state_key)
        if previous is not None and previous != revision:
            raise RuntimeError(
                f"the requested {state_key} differs from the durable run "
                f"snapshot {previous}; use a new output root or set "
                "resume.reset_run_state: true in the config for an intentional "
                "snapshot update")
        return revision

    source_revision = resolve(
        SRC_REPO, "dataset", SOURCE_REVISION, "source_revision")
    tokenizer_revision = resolve(
        TOKENIZER_NAME, "model", TOKENIZER_REVISION, "tokenizer_revision")
    mimi_revision = resolve(
        MIMI_MODEL, "model", MIMI_REVISION, "mimi_revision")
    _write_run_state(_make_run_state(
        source_revision, tokenizer_revision, mimi_revision))
    files = api.list_repo_files(
        SRC_REPO, repo_type="dataset", revision=source_revision, token=TOKEN)
    shards = sorted(name for name in files if name.endswith(".parquet"))
    return source_revision, tokenizer_revision, mimi_revision, shards


def remote_output_path(filename):
    return sharded_repo_path(
        filename, root="data", prefix_chars=HUB_PREFIX_CHARS)


def output_path(filename):
    # Keep local output flat for the training loader. The hash prefix/full hash
    # in the name eliminates collisions between equal source basenames.
    return os.path.join(OUTPUT_DIR, os.path.basename(remote_output_path(filename)))


def legacy_output_path(filename):
    """Path used by older runs, retained for zero-copy resume/migration."""
    return os.path.join(OUTPUT_DIR, filename.replace("/", "-"))


def legacy_remote_output_path(filename):
    return "data/" + filename.replace("/", "-")


def find_local_output(filename):
    for path in (output_path(filename), legacy_output_path(filename)):
        if os.path.isfile(path):
            return path
    return None


def dataset_card(source_revision=None, tokenizer_revision=None,
                 mimi_revision=None):
    return (
        "---\nlicense: apache-2.0\ntask_categories:\n- text-to-speech\n"
        "configs:\n- config_name: default\n"
        "  data_files:\n  - split: train\n    path: data/**/*.parquet\n"
        "---\n\n# Nar TTS encoded dataset\n\n"
        f"`{SRC_REPO}` encoded for Nar TTS: `{TOKENIZER_NAME}` text tokens "
        "+ Mimi audio tokens in the Orpheus prompt layout. Single column "
        "`input_ids`.\n\n"
        f"Source revision: `{source_revision or 'default'}`  \n"
        f"Tokenizer revision: `{tokenizer_revision or 'default'}`  \n"
        f"Mimi revision: `{mimi_revision or 'default'}`\n"
    ).encode()


def prepare_target_repo(source_revision, tokenizer_revision, mimi_revision):
    """Create the destination, refresh its card, and load its resume manifest."""
    import huggingface_hub

    if Version(huggingface_hub.__version__) < Version("1.26.1"):
        raise RuntimeError(
            "streaming upload requires huggingface-hub>=1.26.1; "
            f"found {huggingface_hub.__version__}")
    try:
        import hf_xet  # noqa: F401 - verifies the fast transfer backend exists
    except ImportError as exc:
        raise RuntimeError(
            "streaming upload requires hf-xet (installed by current "
            "huggingface-hub on supported platforms)") from exc

    api = HfApi(token=TOKEN)
    api.create_repo(
        TARGET_REPO, repo_type="dataset", token=TOKEN, private=TARGET_PRIVATE,
        exist_ok=True)
    remote_files = set(api.list_repo_files(
        TARGET_REPO, repo_type="dataset", token=TOKEN))
    run_state = _make_run_state(
        source_revision, tokenizer_revision, mimi_revision)
    if REMOTE_STATE_PATH in remote_files:
        from huggingface_hub import hf_hub_download

        state_path = hf_hub_download(
            repo_id=TARGET_REPO, filename=REMOTE_STATE_PATH,
            repo_type="dataset", token=TOKEN)
        try:
            with open(state_path, encoding="utf-8") as fh:
                remote_state = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot read {TARGET_REPO}/{REMOTE_STATE_PATH}") from exc
        _validate_remote_state(remote_state, run_state)
    elif any(path.endswith(".parquet") for path in remote_files):
        if not ADOPT_LEGACY_TARGET:
            raise RuntimeError(
                f"{TARGET_REPO} already contains Parquet shards but no "
                f"{REMOTE_STATE_PATH}. Set resume.adopt_legacy_target: true "
                "in the config only after confirming they use this "
                "tokenizer/Mimi configuration")
        _log(
            f"WARNING: adopting legacy target {TARGET_REPO} without a prior "
            "preprocessing-state manifest")
    state_bytes = (json.dumps(
        run_state, indent=2, sort_keys=True) + "\n").encode()
    api.upload_file(
        path_or_fileobj=state_bytes, path_in_repo=REMOTE_STATE_PATH,
        repo_id=TARGET_REPO, repo_type="dataset", token=TOKEN,
        commit_message="Pin tokenized dataset preprocessing state")
    api.upload_file(
        path_or_fileobj=dataset_card(
            source_revision, tokenizer_revision, mimi_revision),
        path_in_repo="README.md",
        repo_id=TARGET_REPO, repo_type="dataset", token=TOKEN,
        commit_message="Update tokenized dataset card")
    return api, remote_files


def make_uploader(api):
    return StreamingHubUploader(
        api=api, repo_id=TARGET_REPO, repo_type="dataset", token=TOKEN,
        staging_root=UPLOAD_STAGING_ROOT,
        batch_files=UPLOAD_BATCH_FILES,
        batch_bytes=int(UPLOAD_BATCH_GB * 1024 ** 3),
        flush_seconds=UPLOAD_FLUSH_SECONDS,
        max_outstanding_bytes=int(MAX_UPLOAD_STAGING_GB * 1024 ** 3),
        attempts=TRANSFER_ATTEMPTS,
        delete_after_upload=DELETE_AFTER_UPLOAD,
        verify=True, on_event=_log)


def encode_shard(staged, tokenizer, layout, codec, decoder):
    """Stream one local source shard into one atomic tokenized Parquet shard."""
    out_path = output_path(staged.name)
    stats = encode_audio_parquet(
        staged.path, out_path, tokenizer, layout, codec, decoder=decoder,
        batch_size=BATCH_SIZE, bucket_size=BUCKET_SIZE,
        max_batch_seconds=MAX_BATCH_SECONDS or None,
        read_batch_size=READ_BATCH_SIZE,
        min_seconds=MIN_AUDIO_SECONDS, max_seconds=MAX_AUDIO_SECONDS,
        row_group_size=ROW_GROUP_SIZE)
    if DELETE_DOWNLOADED_SOURCE:
        staged.discard()
    return out_path, stats


def worker(gpu_id, shards, cpu_workers, decode_prefetch, source_revision,
           tokenizer_revision, mimi_revision, uploader=None, config_path=None):
    """Own one GPU while CPU decode, next download, and prior upload overlap."""
    if config_path is not None:
        normalized_config = str(Path(config_path).expanduser().resolve())
        if normalized_config != ACTIVE_CONFIG_PATH:
            configure(normalized_config)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(TOKENIZER_THREADS)
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Start the first network transfer before loading tokenizer/Mimi weights.
    download_root = os.path.join(DOWNLOAD_ROOT, f"gpu{gpu_id}")
    downloads = HubShardDownloader(
        shards, repo_id=SRC_REPO, revision=source_revision,
        staging_root=download_root, local_source_root=LOCAL_SRC_ROOT,
        repo_type="dataset", token=TOKEN, prefetch=DOWNLOAD_PREFETCH,
        attempts=TRANSFER_ATTEMPTS, on_event=_log)

    done = rows = dropped = 0
    audio_samples = padded_samples = 0
    failures = []
    t0 = time.time()
    with downloads:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME, revision=tokenizer_revision, token=TOKEN)
        layout = TokenLayout.from_tokenizer(tokenizer)
        codec = MimiCodec(
            device, num_codebooks=layout.num_codebooks, dtype=MIMI_DTYPE,
            compile_model=MIMI_COMPILE, compile_mode=MIMI_COMPILE_MODE,
            allow_tf32=ALLOW_TF32, model_id=MIMI_MODEL,
            revision=mimi_revision, token=TOKEN)

        _log(
            f"gpu{gpu_id}: start, {len(shards)} shards | {cpu_workers} decode "
            f"workers | source={source_revision[:12]} "
            f"tokenizer={tokenizer_revision[:12]} mimi={mimi_revision[:12]} | "
            f"base_vocab={layout.base}")
        with ParallelAudioDecoder(
                cpu_workers, prefetch=decode_prefetch,
                backend=DECODE_BACKEND) as decoder:
            for staged in downloads:
                if uploader is not None:
                    uploader.check()
                try:
                    out_path, stats = encode_shard(
                        staged, tokenizer, layout, codec, decoder)
                except Exception as exc:  # noqa: BLE001 - log shard and continue
                    failures.append((staged.name, repr(exc)))
                    if staged.owned and not KEEP_FAILED_DOWNLOADS:
                        try:
                            staged.discard()
                        except OSError as cleanup_exc:
                            _log(
                                f"gpu{gpu_id} could not retire failed download "
                                f"{staged.name}: {cleanup_exc!r}")
                    _log(f"gpu{gpu_id} ERROR {staged.name}: {exc!r}")
                else:
                    rows += stats.output_rows
                    dropped += stats.dropped_rows
                    audio_samples += stats.audio_samples
                    padded_samples += stats.padded_samples
                    # This call is outside the encoding error handler. A failed
                    # uploader is fatal and stops wasting GPU work immediately.
                    if uploader is not None:
                        uploader.add(out_path, remote_output_path(staged.name))
                done += 1
                if done % 10 == 0 or done == len(shards):
                    efficiency = (audio_samples / padded_samples
                                  if padded_samples else 1.0)
                    elapsed = max(time.time() - t0, 1e-9)
                    _log(
                        f"gpu{gpu_id}: {done}/{len(shards)} shards | {rows} rows | "
                        f"{dropped} dropped | {rows / elapsed:.1f} rows/s | "
                        f"{efficiency * 100:.1f}% useful audio")

    _log(f"gpu{gpu_id}: DONE {done} shards | {rows} rows | {dropped} dropped | "
         f"downloaded {downloads.downloaded_files} shards "
         f"({downloads.downloaded_bytes / 1024 ** 3:.2f} GiB)")
    if failures:
        raise RuntimeError(
            f"gpu{gpu_id} failed to encode {len(failures)} shard(s); "
            "successful shards remain resumable")
    return {"shards": done, "rows": rows, "dropped": dropped}


def _cleanup_incomplete_outputs():
    removed = 0
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".parquet.tmp"):
            continue
        os.remove(os.path.join(OUTPUT_DIR, name))
        removed += 1
    if removed:
        _log(f"removed {removed} incomplete local Parquet output(s)")


def _run_locked():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_ROOT, exist_ok=True)
    cleanup_stale_uploads(UPLOAD_STAGING_ROOT)
    _cleanup_incomplete_outputs()
    if NUM_GPUS < 1:
        raise ValueError("runtime.num_gpus must be at least 1")
    if PUSH_TO_HUB and NUM_GPUS != 1:
        raise ValueError(
            "the bounded download/tokenize/upload mode is optimized for exactly "
            "one GPU; set runtime.num_gpus: 1 when target.push_to_hub is true")

    (source_revision, tokenizer_revision,
     mimi_revision, all_shards) = list_shards()
    if PUSH_TO_HUB and len(all_shards) > 10_000 and HUB_PREFIX_CHARS == 0:
        raise ValueError(
            "more than 10,000 outputs cannot share one Hub folder; leave "
            "transfer.hub_prefix_chars at its default or set it to at least 1")
    if len(all_shards) > 100_000:
        _log(
            f"WARNING: {len(all_shards)} output files exceeds the Hub's "
            "recommended per-repository count; consider larger source shards")
    api = None
    remote_files = set()
    if PUSH_TO_HUB:
        api, remote_files = prepare_target_repo(
            source_revision, tokenizer_revision, mimi_revision)

    to_encode = []
    local_backlog = []
    remote_complete = local_complete = 0
    for filename in all_shards:
        local = find_local_output(filename)
        new_remote = remote_output_path(filename)
        legacy_remote = legacy_remote_output_path(filename)
        if new_remote in remote_files:
            matched_remote = new_remote
        elif legacy_remote in remote_files:
            matched_remote = legacy_remote
        else:
            matched_remote = None
        if matched_remote is not None:
            remote_complete += 1
            # Do not delete merely because a path was listed. Re-submit the
            # local file (Xet deduplicates it), verify remote byte size, then
            # let the uploader retire it through the same integrity path.
            if local is not None and DELETE_AFTER_UPLOAD:
                local_backlog.append((filename, local, matched_remote))
        elif local is not None:
            local_complete += 1
            if PUSH_TO_HUB:
                local_backlog.append((filename, local, new_remote))
        else:
            to_encode.append(filename)

    if not to_encode and not local_backlog:
        _log(f"all {len(all_shards)} source shards are complete "
             f"({remote_complete} remote, {local_complete} local)")
        return

    if to_encode:
        available = torch.cuda.device_count()
        if NUM_GPUS > available:
            raise RuntimeError(
                f"requested {NUM_GPUS} GPU(s), but PyTorch sees {available}")

    workers_per_gpu = max(1, CPU_WORKERS // NUM_GPUS)
    prefetch_per_gpu = max(workers_per_gpu, DECODE_PREFETCH // NUM_GPUS)
    _log(
        f"streaming {len(to_encode)} shards ({remote_complete} remote, "
        f"{local_complete} local, {len(local_backlog)} awaiting upload) | "
        f"source={SRC_REPO}@{source_revision[:12]} -> {TARGET_REPO} | "
        f"{NUM_GPUS} GPU | {CPU_WORKERS} total CPU workers | "
        f"download_prefetch={DOWNLOAD_PREFETCH} batch={BATCH_SIZE} "
        f"bucket={BUCKET_SIZE} padded_budget={MAX_BATCH_SECONDS:.0f}s")

    uploader = make_uploader(api) if PUSH_TO_HUB else None
    upload_context = uploader if uploader is not None else contextlib.nullcontext()
    run_stats = {"shards": 0, "rows": 0, "dropped": 0}
    with upload_context as upload_queue:
        if upload_queue is not None:
            for _, path, path_in_repo in local_backlog:
                upload_queue.add(path, path_in_repo)

        if to_encode:
            assignments = [to_encode[g::NUM_GPUS] for g in range(NUM_GPUS)]
            if NUM_GPUS == 1:
                run_stats = worker(
                    0, assignments[0], workers_per_gpu, prefetch_per_gpu,
                    source_revision, tokenizer_revision, mimi_revision,
                    uploader=upload_queue, config_path=ACTIVE_CONFIG_PATH)
            else:
                ctx = mp.get_context("spawn")
                processes = [ctx.Process(
                    target=worker,
                    args=(gpu_id, assignments[gpu_id], workers_per_gpu,
                          prefetch_per_gpu, source_revision,
                          tokenizer_revision, mimi_revision, None,
                          ACTIVE_CONFIG_PATH))
                    for gpu_id in range(NUM_GPUS)]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()
                failed = [process.pid for process in processes if process.exitcode]
                if failed:
                    raise RuntimeError(
                        f"dataset workers failed (pids: {failed})")

    if uploader is not None:
        _log(
            f"DONE | {run_stats['shards']} newly encoded shards | "
            f"{run_stats['rows']} new rows | {uploader.uploaded_files} uploaded "
            f"in {uploader.uploaded_batches} batches "
            f"({uploader.uploaded_bytes / 1024 ** 3:.2f} GiB) -> {TARGET_REPO}")
    else:
        names = [name for name in os.listdir(OUTPUT_DIR)
                 if name.endswith(".parquet")]
        total_rows = sum(
            pq.read_metadata(os.path.join(OUTPUT_DIR, name)).num_rows
            for name in names)
        _log(f"DONE | {len(names)} local output shards | {total_rows} total rows")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stream a speech dataset through text and Mimi tokenization")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="YAML run config (default: %(default)s)")
    args = parser.parse_args(argv)
    configure(args.config)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with exclusive_file_lock(os.path.join(OUTPUT_ROOT, ".encode.lock")):
        _log(
            f"config={ACTIVE_CONFIG_PATH} | detected {CPU_COUNT} available "
            f"logical CPUs; using {CPU_WORKERS} decode workers")
        _run_locked()


if __name__ == "__main__":
    main()
