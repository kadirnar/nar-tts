"""Bounded Hub download/upload stages for multi-terabyte preprocessing runs.

The GPU/CPU tokenizers should never wait for avoidable network I/O, but a slow
network must also never fill the local disk.  This module supplies the two
outer stages around the tokenization pipeline:

    Hub download -> tokenize -> Hub upload

Downloads are prefetched into an owned ``local_dir`` and uploads are batched on
a background thread.  Both sides apply backpressure.  Completed local outputs
are deleted only after their Hub commit is visible and its byte size matches.
"""

from __future__ import annotations

import hashlib
import os
import queue
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath

_END = object()


class _WorkerError:
    def __init__(self, exception):
        self.exception = exception


def _repo_path(path):
    """Normalize a relative Hub path and reject traversal/absolute paths."""
    normalized = os.fspath(path).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if (not parts or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError(f"unsafe repository path: {path!r}")
    return PurePosixPath(*parts).as_posix()


def _safe_local_path(root, relative):
    root = os.path.abspath(os.fspath(root))
    relative = _repo_path(relative)
    path = os.path.abspath(os.path.join(root, *PurePosixPath(relative).parts))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(f"path escapes staging root: {relative!r}")
    return path


def _prune_empty_parents(path, boundary):
    boundary = os.path.abspath(os.fspath(boundary))
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    while parent != boundary and os.path.commonpath((boundary, parent)) == boundary:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def _remove_owned_tree(root):
    """Unlink one known temporary tree without following directory symlinks."""
    root = os.path.abspath(os.fspath(root))
    if not os.path.lexists(root):
        return
    if os.path.islink(root) or not os.path.isdir(root):
        os.unlink(root)
        return
    for current, directories, files in os.walk(root, topdown=False,
                                                followlinks=False):
        for name in files:
            os.unlink(os.path.join(current, name))
        for name in directories:
            path = os.path.join(current, name)
            if os.path.islink(path):
                os.unlink(path)
            else:
                os.rmdir(path)
    os.rmdir(root)


def cleanup_stale_uploads(staging_root):
    """Remove hardlinks/symlinks left in the pipeline-owned upload scratch dir."""
    staging_root = os.path.abspath(os.fspath(staging_root))
    if not os.path.isdir(staging_root):
        return
    for name in os.listdir(staging_root):
        _remove_owned_tree(os.path.join(staging_root, name))


@contextmanager
def exclusive_file_lock(path):
    """Prevent concurrent preprocessors from sharing one output/staging root."""
    import fcntl

    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another preprocessing process holds {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def sharded_repo_path(source_key, basename=None, *, root="data",
                      prefix_chars=2):
    """Return a stable, collision-resistant path spread across Hub folders.

    The full SHA-256 makes source paths with the same basename unambiguous.  A
    two-hex-character prefix distributes 100k shards over 256 directories,
    keeping each directory far below the Hub's entry limit.
    """
    key = _repo_path(source_key)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    basename = os.path.basename(key) if basename is None else os.fspath(basename)
    basename = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-")
    basename = (basename or "shard.parquet")[-120:]
    filename = f"{digest}-{basename}"
    root = _repo_path(root)
    if prefix_chars < 0 or prefix_chars > len(digest):
        raise ValueError("prefix_chars must be between 0 and 64")
    if prefix_chars:
        return f"{root}/{digest[:prefix_chars]}/{filename}"
    return f"{root}/{filename}"


def _retry(call, attempts, on_retry=None):
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(30.0, float(2 ** (attempt - 1)))
            if on_retry is not None:
                on_retry(attempt, attempts, delay, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class StagedShard:
    """A local source shard, optionally owned by the download staging area."""

    name: str
    path: str
    owned: bool = False
    staging_root: str | None = None

    def discard(self):
        """Delete only a successfully-consumed, pipeline-owned download."""
        if not self.owned:
            return False
        try:
            os.remove(self.path)
        except FileNotFoundError:
            return False
        if self.staging_root is not None:
            _prune_empty_parents(self.path, self.staging_root)
        return True


class HubShardDownloader:
    """Prefetch individual Hub shards into a bounded, disposable local folder.

    ``prefetch=1`` means at most one ready/download-in-progress shard ahead of
    the shard currently being tokenized.  Existing files under
    ``local_source_root`` are borrowed and are never deleted.
    """

    def __init__(self, filenames, *, repo_id, revision, staging_root,
                 local_source_root=None, repo_type="dataset", token=None,
                 prefetch=1, attempts=5, download_fn=None, on_event=None):
        self.filenames = tuple(filenames)
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.token = token
        self.staging_root = os.path.abspath(os.fspath(staging_root))
        self.local_source_root = (None if local_source_root is None else
                                  os.path.abspath(os.fspath(local_source_root)))
        self.prefetch = max(1, int(prefetch))
        self.attempts = max(1, int(attempts))
        self.download_fn = download_fn
        self.on_event = on_event
        self.downloaded_files = 0
        self.downloaded_bytes = 0
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self._thread is not None:
            return self
        os.makedirs(self.staging_root, exist_ok=True)
        self._thread = threading.Thread(
            target=self._produce, name="hub-shard-downloader", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _wait_for_slot(self):
        # Reserve disk capacity before starting another network transfer.  A
        # plain bounded queue alone permits one additional in-progress download.
        while self._queue.full() and not self._stop.is_set():
            self._stop.wait(0.1)
        return not self._stop.is_set()

    def _resolve(self, filename):
        filename = _repo_path(filename)
        if self.local_source_root is not None:
            local_path = _safe_local_path(self.local_source_root, filename)
            if os.path.isfile(local_path):
                return StagedShard(filename, local_path, owned=False)

        if self.download_fn is None:
            from huggingface_hub import hf_hub_download

            download_fn = hf_hub_download
        else:
            download_fn = self.download_fn

        def download():
            return download_fn(
                repo_id=self.repo_id, filename=filename,
                repo_type=self.repo_type, revision=self.revision,
                token=self.token, local_dir=self.staging_root)

        def retry_message(attempt, total, delay, exc):
            if self.on_event is not None:
                self.on_event(
                    f"download retry {attempt}/{total - 1} for {filename} "
                    f"in {delay:.0f}s: {exc!r}")

        path = os.path.abspath(os.fspath(_retry(
            download, self.attempts, retry_message)))
        if os.path.commonpath((self.staging_root, path)) != self.staging_root:
            raise RuntimeError(
                f"download returned a path outside staging_root: {path}")
        size = os.path.getsize(path)
        self.downloaded_files += 1
        self.downloaded_bytes += size
        return StagedShard(
            filename, path, owned=True, staging_root=self.staging_root)

    def _produce(self):
        try:
            for filename in self.filenames:
                if not self._wait_for_slot():
                    return
                if not self._put(self._resolve(filename)):
                    return
        except Exception as exc:  # noqa: BLE001 - transfer to consumer thread
            self._put(_WorkerError(exc))
        finally:
            self._put(_END)

    def __iter__(self):
        if self._thread is None:
            raise RuntimeError(
                "HubShardDownloader must be used as a context manager")
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, _WorkerError):
                raise item.exception
            yield item


@dataclass(frozen=True)
class UploadItem:
    local_path: str
    path_in_repo: str
    size: int


class StreamingHubUploader:
    """Batch, verify, and retire encoded files on a background upload thread.

    A batch is exposed through a temporary hardlink tree matching its desired
    repository layout.  ``HfApi.upload_folder`` can then use its parallel Xet
    pipeline without copying the encoded data or scanning unrelated outputs.
    """

    def __init__(self, *, api, repo_id, staging_root, repo_type="dataset",
                 token=None, batch_files=64, batch_bytes=8 * 1024 ** 3,
                 flush_seconds=30, max_outstanding_files=None,
                 max_outstanding_bytes=24 * 1024 ** 3, attempts=5,
                 delete_after_upload=True, verify=True, on_event=None):
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token
        self.staging_root = os.path.abspath(os.fspath(staging_root))
        self.batch_files = max(1, int(batch_files))
        self.batch_bytes = max(1, int(batch_bytes))
        self.flush_seconds = max(0.0, float(flush_seconds))
        if max_outstanding_files is None:
            max_outstanding_files = self.batch_files * 3
        self.max_outstanding_files = max(1, int(max_outstanding_files))
        self.max_outstanding_bytes = max(1, int(max_outstanding_bytes))
        self.attempts = max(1, int(attempts))
        self.delete_after_upload = bool(delete_after_upload)
        self.verify = bool(verify)
        self.on_event = on_event

        self.uploaded_files = 0
        self.uploaded_bytes = 0
        self.uploaded_batches = 0
        self._outstanding_files = 0
        self._outstanding_bytes = 0
        self._condition = threading.Condition()
        self._queue = queue.Queue(maxsize=self.max_outstanding_files)
        self._error = None
        self._thread = None
        self._run_root = None
        self._closing = False
        self._seen_paths = set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.finish()
        except Exception as upload_exc:
            if exc_type is None:
                raise
            if self.on_event is not None:
                self.on_event(
                    f"uploader also failed while handling {exc!r}: "
                    f"{upload_exc!r}")
        return False

    def start(self):
        with self._condition:
            if self._closing:
                raise RuntimeError("cannot restart a finished uploader")
            self._raise_if_error()
            if self._thread is not None:
                return
            os.makedirs(self.staging_root, exist_ok=True)
            self._run_root = tempfile.mkdtemp(
                prefix="run-", dir=self.staging_root)
            self._thread = threading.Thread(
                target=self._run, name="hub-shard-uploader", daemon=True)
            self._thread.start()

    def _raise_if_error(self):
        if self._error is not None:
            raise RuntimeError("background Hub upload failed") from self._error

    def check(self):
        """Raise promptly if the background uploader has failed."""
        with self._condition:
            self._raise_if_error()

    def add(self, local_path, path_in_repo):
        """Queue one durable local output, blocking when the disk budget is full."""
        self.start()
        local_path = os.path.abspath(os.fspath(local_path))
        path_in_repo = _repo_path(path_in_repo)
        size = os.path.getsize(local_path)
        with self._condition:
            if self._closing:
                raise RuntimeError("cannot add files after uploader.finish()")
            if path_in_repo in self._seen_paths:
                raise ValueError(
                    f"duplicate upload path in this run: {path_in_repo}")
            while True:
                self._raise_if_error()
                too_many = self._outstanding_files >= self.max_outstanding_files
                too_large = (self._outstanding_files > 0 and
                             self._outstanding_bytes + size >
                             self.max_outstanding_bytes)
                if not too_many and not too_large:
                    break
                self._condition.wait(timeout=0.25)
            self._seen_paths.add(path_in_repo)
            self._outstanding_files += 1
            self._outstanding_bytes += size
        self._queue.put(UploadItem(local_path, path_in_repo, size))

    def _get_batch(self, first):
        batch = [first]
        batch_bytes = first.size
        deadline = time.monotonic() + self.flush_seconds
        carry = None
        finishing = False
        while len(batch) < self.batch_files and batch_bytes < self.batch_bytes:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                break
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is _END:
                finishing = True
                break
            if batch and batch_bytes + item.size > self.batch_bytes:
                carry = item
                break
            batch.append(item)
            batch_bytes += item.size
        return batch, carry, finishing

    def _make_batch_tree(self, batch):
        batch_root = tempfile.mkdtemp(prefix="batch-", dir=self._run_root)
        try:
            for item in batch:
                target = _safe_local_path(batch_root, item.path_in_repo)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                try:
                    os.link(item.local_path, target)
                except OSError:
                    # Symlinks also avoid a second multi-GB copy and are followed
                    # by CommitOperationAdd/hf_xet as ordinary file content.
                    os.symlink(item.local_path, target)
            return batch_root
        except Exception:
            _remove_owned_tree(batch_root)
            raise

    def _verify_batch(self, batch):
        infos = self.api.get_paths_info(
            repo_id=self.repo_id,
            paths=[item.path_in_repo for item in batch],
            repo_type=self.repo_type, token=self.token)
        remote = {info.path: info for info in infos}
        problems = []
        for item in batch:
            info = remote.get(item.path_in_repo)
            if info is None:
                problems.append(f"missing {item.path_in_repo}")
            elif getattr(info, "size", None) != item.size:
                problems.append(
                    f"size mismatch for {item.path_in_repo}: "
                    f"local={item.size}, remote={getattr(info, 'size', None)}")
        if problems:
            raise RuntimeError("Hub upload verification failed: " + "; ".join(problems))

    def _upload_batch(self, batch):
        batch_root = self._make_batch_tree(batch)
        total_bytes = sum(item.size for item in batch)

        def upload_and_verify():
            self.api.upload_folder(
                repo_id=self.repo_id, repo_type=self.repo_type,
                folder_path=batch_root, path_in_repo="", token=self.token,
                commit_message=f"Upload {len(batch)} tokenized shard(s)")
            if self.verify:
                self._verify_batch(batch)

        def retry_message(attempt, total, delay, exc):
            if self.on_event is not None:
                self.on_event(
                    f"upload retry {attempt}/{total - 1} for {len(batch)} "
                    f"shards in {delay:.0f}s: {exc!r}")

        try:
            _retry(upload_and_verify, self.attempts, retry_message)
            if self.delete_after_upload:
                for item in batch:
                    try:
                        os.remove(item.local_path)
                    except FileNotFoundError:
                        pass
            self.uploaded_files += len(batch)
            self.uploaded_bytes += total_bytes
            self.uploaded_batches += 1
            if self.on_event is not None:
                self.on_event(
                    f"uploaded+verified {len(batch)} shards "
                    f"({total_bytes / 1024 ** 3:.2f} GiB)")
        finally:
            _remove_owned_tree(batch_root)

    def _complete(self, batch):
        with self._condition:
            self._outstanding_files -= len(batch)
            self._outstanding_bytes -= sum(item.size for item in batch)
            self._condition.notify_all()

    def _run(self):
        carry = None
        try:
            while True:
                if carry is None:
                    first = self._queue.get()
                else:
                    first, carry = carry, None
                if first is _END:
                    return
                batch, carry, finishing = self._get_batch(first)
                self._upload_batch(batch)
                self._complete(batch)
                if finishing:
                    return
        except Exception as exc:  # noqa: BLE001 - relay to producer thread
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if self._run_root is not None:
                try:
                    _remove_owned_tree(self._run_root)
                except OSError:
                    pass

    def finish(self):
        """Flush all accepted files, wait for verification, and stop the worker."""
        if self._thread is None:
            return
        with self._condition:
            self._closing = True
        while self._thread.is_alive():
            with self._condition:
                if self._error is not None:
                    break
            try:
                self._queue.put(_END, timeout=0.1)
                break
            except queue.Full:
                pass
        self._thread.join()
        self._thread = None
        with self._condition:
            self._raise_if_error()
