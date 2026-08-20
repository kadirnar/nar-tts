import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nar_tts.preprocessing.hub_pipeline import (
    HubShardDownloader,
    StreamingHubUploader,
    cleanup_stale_uploads,
    exclusive_file_lock,
    sharded_repo_path,
)


class FakeHubApi:
    def __init__(self, *, wrong_size=False, upload_gate=None):
        self.remote = {}
        self.upload_calls = []
        self.wrong_size = wrong_size
        self.upload_gate = upload_gate

    def upload_folder(self, *, folder_path, **kwargs):
        if self.upload_gate is not None:
            self.upload_gate.wait(timeout=5)
        uploaded = []
        root = Path(folder_path)
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                self.remote[relative] = path.read_bytes()
                uploaded.append(relative)
        self.upload_calls.append(uploaded)
        return SimpleNamespace(oid=str(len(self.upload_calls)))

    def get_paths_info(self, *, paths, **kwargs):
        result = []
        for path in paths:
            if path in self.remote:
                size = len(self.remote[path])
                if self.wrong_size:
                    size += 1
                result.append(SimpleNamespace(path=path, size=size))
        return result


class HubPipelineTest(unittest.TestCase):
    def test_repo_paths_are_stable_unique_and_fanned_out(self):
        first = sharded_repo_path("data/a/train.parquet")
        again = sharded_repo_path("data/a/train.parquet")
        second = sharded_repo_path("data/b/train.parquet")

        self.assertEqual(first, again)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^data/[0-9a-f]{2}/[0-9a-f]{64}-train\.parquet$")
        with self.assertRaises(ValueError):
            sharded_repo_path("../outside.parquet")

    def test_downloader_borrows_local_and_prefetches_only_one_ahead(self):
        with tempfile.TemporaryDirectory() as directory:
            local_root = os.path.join(directory, "local")
            staging_root = os.path.join(directory, "staging")
            local_path = os.path.join(local_root, "data", "local.parquet")
            os.makedirs(os.path.dirname(local_path))
            Path(local_path).write_bytes(b"local")
            downloaded = []
            one_downloaded = threading.Event()

            def fake_download(*, filename, local_dir, **kwargs):
                path = os.path.join(local_dir, *filename.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Path(path).write_bytes(filename.encode())
                downloaded.append(filename)
                one_downloaded.set()
                return path

            downloader = HubShardDownloader(
                ["data/local.parquet", "data/one.parquet", "data/two.parquet"],
                repo_id="owner/source", revision="abc", prefetch=1,
                local_source_root=local_root, staging_root=staging_root,
                download_fn=fake_download, attempts=1)
            with downloader:
                iterator = iter(downloader)
                local = next(iterator)
                self.assertFalse(local.owned)
                self.assertFalse(local.discard())

                self.assertTrue(one_downloaded.wait(timeout=2))
                time.sleep(0.05)
                self.assertEqual(downloaded, ["data/one.parquet"])

                first = next(iterator)
                self.assertTrue(first.owned)
                deadline = time.monotonic() + 2
                while len(downloaded) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(downloaded[-1], "data/two.parquet")
                second = next(iterator)
                with self.assertRaises(StopIteration):
                    next(iterator)

                self.assertTrue(first.discard())
                self.assertTrue(second.discard())
                self.assertTrue(os.path.isfile(local_path))

            self.assertEqual(downloader.downloaded_files, 2)
            self.assertFalse(os.path.exists(first.path))
            self.assertFalse(os.path.exists(second.path))

    def test_uploader_batches_verifies_then_deletes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "output")
            staging = os.path.join(directory, "staging")
            os.makedirs(output)
            paths = []
            for index, content in enumerate((b"one", b"two2", b"three33")):
                path = os.path.join(output, f"{index}.parquet")
                Path(path).write_bytes(content)
                paths.append(path)

            api = FakeHubApi()
            uploader = StreamingHubUploader(
                api=api, repo_id="owner/target", staging_root=staging,
                batch_files=2, batch_bytes=1024, flush_seconds=0.2,
                max_outstanding_bytes=1024, attempts=1,
                delete_after_upload=True)
            with uploader:
                for index, path in enumerate(paths):
                    uploader.add(path, f"data/aa/{index}.parquet")

            self.assertEqual(uploader.uploaded_files, 3)
            self.assertEqual(uploader.uploaded_batches, 2)
            self.assertEqual(set(api.remote), {
                "data/aa/0.parquet", "data/aa/1.parquet", "data/aa/2.parquet",
            })
            self.assertTrue(all(not os.path.exists(path) for path in paths))
            self.assertEqual(os.listdir(staging), [])

    def test_failed_verification_keeps_the_only_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "output.parquet")
            staging = os.path.join(directory, "staging")
            Path(output).write_bytes(b"keep me")
            uploader = StreamingHubUploader(
                api=FakeHubApi(wrong_size=True), repo_id="owner/target",
                staging_root=staging, flush_seconds=0, attempts=1,
                delete_after_upload=True)

            uploader.add(output, "data/aa/output.parquet")
            with self.assertRaisesRegex(RuntimeError, "background Hub upload"):
                uploader.finish()

            self.assertTrue(os.path.isfile(output))
            self.assertEqual(Path(output).read_bytes(), b"keep me")
            self.assertEqual(os.listdir(staging), [])

    def test_byte_budget_backpressures_producer_until_upload_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.parquet")
            second = os.path.join(directory, "second.parquet")
            Path(first).write_bytes(b"1234")
            Path(second).write_bytes(b"5678")
            gate = threading.Event()
            uploader = StreamingHubUploader(
                api=FakeHubApi(upload_gate=gate), repo_id="owner/target",
                staging_root=os.path.join(directory, "staging"),
                batch_files=1, batch_bytes=4, flush_seconds=0,
                max_outstanding_bytes=4, attempts=1)
            uploader.start()
            uploader.add(first, "data/aa/first.parquet")
            second_added = threading.Event()

            def add_second():
                uploader.add(second, "data/aa/second.parquet")
                second_added.set()

            producer = threading.Thread(target=add_second)
            producer.start()
            time.sleep(0.1)
            self.assertFalse(second_added.is_set())
            gate.set()
            self.assertTrue(second_added.wait(timeout=2))
            producer.join()
            uploader.finish()
            self.assertEqual(uploader.uploaded_files, 2)

    def test_stale_upload_cleanup_removes_only_children(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = os.path.join(directory, "run-old", "data")
            os.makedirs(stale)
            Path(os.path.join(stale, "file.parquet")).write_bytes(b"x")
            cleanup_stale_uploads(directory)
            self.assertTrue(os.path.isdir(directory))
            self.assertEqual(os.listdir(directory), [])

    def test_output_lock_rejects_a_concurrent_run(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = os.path.join(directory, ".encode.lock")
            with (exclusive_file_lock(lock_path),
                  self.assertRaisesRegex(RuntimeError, "another preprocessing"),
                  exclusive_file_lock(lock_path)):
                self.fail("the second lock must not be acquired")


if __name__ == "__main__":
    unittest.main()
