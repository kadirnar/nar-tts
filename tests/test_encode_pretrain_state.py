import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml

from nar_tts.preprocessing import encode_pretrain


class FakeManifestApi:
    def __init__(self):
        self.heads = {
            "dataset": "a" * 40,
            "tokenizer": "b" * 40,
            "mimi": "c" * 40,
        }
        self.requested_revisions = []

    def repo_info(self, repo_id, *, revision=None, repo_type=None, **kwargs):
        self.requested_revisions.append(revision)
        if repo_type == "dataset":
            key = "dataset"
        elif repo_id == encode_pretrain.TOKENIZER_NAME:
            key = "tokenizer"
        else:
            key = "mimi"
        return SimpleNamespace(sha=revision or self.heads[key])

    @staticmethod
    def list_repo_files(repo_id, *, revision=None, **kwargs):
        return ["README.md", "data/z.parquet", "data/a.parquet"]


class EncodePretrainStateTest(unittest.TestCase):
    def test_auto_cpu_workers_use_the_complete_affinity_mask(self):
        with open(encode_pretrain.DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        with tempfile.TemporaryDirectory() as directory:
            config["source"]["local_root"] = None
            config["target"]["output_root"] = "encoded"
            config["target"]["download_root"] = None
            config["runtime"]["cpu_workers"] = "auto"
            config["runtime"]["decode_prefetch"] = "auto"
            path = os.path.join(directory, "preprocess.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh)

            try:
                with mock.patch.object(
                        encode_pretrain.os, "sched_getaffinity",
                        return_value=set(range(7))):
                    encode_pretrain.configure(path)
                self.assertEqual(encode_pretrain.CPU_COUNT, 7)
                self.assertEqual(encode_pretrain.CPU_WORKERS, 7)
                self.assertEqual(encode_pretrain.DECODE_PREFETCH, 28)
                self.assertIsNone(encode_pretrain.LOCAL_SRC_ROOT)
                self.assertIsNone(encode_pretrain.TOKEN)
                self.assertEqual(
                    encode_pretrain.OUTPUT_ROOT,
                    os.path.join(directory, "encoded"))
                self.assertEqual(
                    encode_pretrain.DOWNLOAD_ROOT,
                    os.path.join(directory, "encoded", ".downloads"))
            finally:
                encode_pretrain.configure()

    def test_config_rejects_a_disabled_cpu_pool(self):
        with open(encode_pretrain.DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        config["runtime"]["cpu_workers"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "preprocess.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh)
            try:
                with self.assertRaisesRegex(
                        ValueError, "runtime.cpu_workers must be at least 1"):
                    encode_pretrain.configure(path)
            finally:
                encode_pretrain.configure()

    def test_incompatible_remote_state_requires_explicit_acceptance(self):
        current = {
            "version": 2,
            "source_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
        }
        encode_pretrain._validate_remote_state(current, current.copy())
        changed = {**current, "tokenizer_revision": "c" * 40}
        with (mock.patch.object(
                encode_pretrain, "ACCEPT_REMOTE_STATE_CHANGE", False),
              self.assertRaisesRegex(RuntimeError, "does not match this run")):
            encode_pretrain._validate_remote_state(changed, current)

        with (mock.patch.object(
                encode_pretrain, "ACCEPT_REMOTE_STATE_CHANGE", True),
              mock.patch.object(encode_pretrain, "_log") as log):
            encode_pretrain._validate_remote_state(changed, current)
        log.assert_called_once()

    def test_restart_reuses_the_persisted_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = os.path.join(directory, "run-state.json")
            api = FakeManifestApi()
            with mock.patch.multiple(
                    encode_pretrain, RUN_STATE_PATH=state_path,
                    SOURCE_REVISION=None, RESET_RUN_STATE=False,
                    HfApi=lambda token=None: api):
                (revision, tokenizer_revision,
                 mimi_revision, shards) = encode_pretrain.list_shards()
                self.assertEqual(revision, "a" * 40)
                self.assertEqual(tokenizer_revision, "b" * 40)
                self.assertEqual(mimi_revision, "c" * 40)
                self.assertEqual(shards, ["data/a.parquet", "data/z.parquet"])

                api.heads = {key: "d" * 40 for key in api.heads}
                (restarted_revision, restarted_tokenizer,
                 restarted_mimi, _) = encode_pretrain.list_shards()

            self.assertEqual(restarted_revision, revision)
            self.assertEqual(restarted_tokenizer, tokenizer_revision)
            self.assertEqual(restarted_mimi, mimi_revision)
            self.assertEqual(api.requested_revisions, [
                None, None, None,
                revision, tokenizer_revision, mimi_revision,
            ])
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            self.assertEqual(state["source_revision"], revision)
            self.assertEqual(state["tokenizer_revision"], tokenizer_revision)
            self.assertEqual(state["mimi_revision"], mimi_revision)

    def test_restart_rejects_output_affecting_setting_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = os.path.join(directory, "run-state.json")
            api = FakeManifestApi()
            with mock.patch.multiple(
                    encode_pretrain, RUN_STATE_PATH=state_path,
                    SOURCE_REVISION=None, RESET_RUN_STATE=False,
                    HfApi=lambda token=None: api):
                encode_pretrain.list_shards()
                with (mock.patch.object(
                        encode_pretrain, "BATCH_SIZE",
                        encode_pretrain.BATCH_SIZE + 1),
                      self.assertRaisesRegex(
                          RuntimeError, "output-affecting settings changed")):
                    encode_pretrain.list_shards()


if __name__ == "__main__":
    unittest.main()
