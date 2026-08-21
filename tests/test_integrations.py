import copy
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from nar_tts.core.config import (
    apply_user_token_ids,
    configure_reporting,
    user_token_ids,
)
from nar_tts.inference.infer import load_inference_config
from nar_tts.integrations.sglang_server import server_command
from nar_tts.training.finetune import _load_config as load_finetune_config
from nar_tts.training.finetune import _padding_token_id
from nar_tts.training.grpo_config import (
    GRPOConfigError,
    available_cpu_count,
    load_grpo_config,
    validate_grpo_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "nar_tts" / "configs"


class IntegrationConfigTest(unittest.TestCase):
    @staticmethod
    def _grpo_config():
        config = load_grpo_config(CONFIGS / "train" / "grpo.yaml")
        config["tokens"] = {"text_eos_token_id": 2, "pad_token_id": 2}
        return config

    def test_train_and_inference_configs_are_separated(self):
        launch = CONFIGS / "train" / "launch"
        self.assertEqual(
            {path.name for path in launch.glob("*.yaml")},
            {"single_gpu.yaml", "fsdp.yaml"},
        )
        self.assertEqual(
            {path.name for path in (CONFIGS / "train").glob("*.yaml")},
            {"preprocess.yaml", "pretrain.yaml", "finetune.yaml", "grpo.yaml"},
        )
        self.assertEqual(
            {path.name for path in (CONFIGS / "inference").glob("*.yaml")},
            {"override.yaml"},
        )
        self.assertFalse(list(CONFIGS.glob("*.yaml")))

    def test_fsdp_wrap_is_model_neutral(self):
        path = CONFIGS / "train" / "launch" / "fsdp.yaml"
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        fsdp = config["fsdp_config"]
        self.assertEqual(fsdp["fsdp_auto_wrap_policy"], "SIZE_BASED_WRAP")
        self.assertNotIn("fsdp_transformer_layer_cls_to_wrap", fsdp)

    def test_sglang_server_is_fully_derived_from_grpo_yaml(self):
        config = copy.deepcopy(self._grpo_config())
        config["rollout"] = {
            "backend": "sglang",
            "sglang": {
                "base_url": "http://127.0.0.1:30000",
                "adapter_sync_dir": "/tmp/nar-sglang-adapter",
                "tensor_parallel_size": 1,
            },
        }
        command = server_command(config)
        self.assertIn("sglang.launch_server", command)
        self.assertIn("--enable-custom-logit-processor", command)
        self.assertIn("--enable-lora", command)
        self.assertEqual(command[command.index("--tp-size") + 1], "1")

    def test_llama_factory_needs_no_repository_config(self):
        self.assertFalse((CONFIGS / "llama_factory").exists())
        self.assertFalse((ROOT / "data" / "llama_factory").exists())

    def test_cpu_count_falls_back_when_affinity_is_unavailable(self):
        with (
            patch("os.sched_getaffinity", side_effect=OSError),
            patch("os.cpu_count", return_value=12),
        ):
            self.assertEqual(available_cpu_count(), 12)

    def test_one_finetune_config_supports_both_loaders(self):
        config = load_finetune_config(CONFIGS / "train" / "finetune.yaml")
        self.assertEqual(config["model"]["loader"], "transformers")
        self.assertIn("unsloth", config["model"])
        self.assertFalse(config["model"]["peft"]["enabled"])
        self.assertEqual(config["runtime"]["dataloader_workers"], "auto")

    def test_user_tokens_and_wandb_are_explicit(self):
        for name in ("pretrain.yaml", "finetune.yaml", "grpo.yaml"):
            path = CONFIGS / "train" / name
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertIsNone(config["tokens"]["text_eos_token_id"])
            self.assertIsNone(config["tokens"]["pad_token_id"])
            self.assertEqual(config["logging"]["report_to"], "wandb")
        with self.assertRaisesRegex(ValueError, "written by the user"):
            user_token_ids({"tokens": {}})

        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 3

            def __init__(self):
                self.init_kwargs = {}

            def __len__(self):
                return 4

            def get_vocab(self):
                return {"<unk>": 0, "a": 1, "</s>": 2, "<pad>": 3}

        tokenizer = Tokenizer()
        apply_user_token_ids(
            tokenizer,
            {"tokens": {"text_eos_token_id": 2, "pad_token_id": 3}},
        )
        self.assertEqual(tokenizer.init_kwargs["nar_text_eos_token_id"], 2)

        with patch.dict("os.environ", {}, clear=True):
            target = configure_reporting(
                {"enabled": True, "report_to": "wandb", "project": "nar-test"}
            )
            self.assertEqual(target, "wandb")
            self.assertEqual(os.environ["WANDB_PROJECT"], "nar-test")

    def test_inference_defaults_are_embedded_and_overridable(self):
        defaults = load_inference_config()
        self.assertIsNone(defaults["_config_path"])
        self.assertEqual(defaults["best_of_n"], {"initial": 2, "maximum": 4})
        override = load_inference_config(CONFIGS / "inference" / "override.yaml")
        self.assertEqual(override["model"]["checkpoint"], "checkpoints/latest")
        self.assertTrue(override["verification"]["asr"]["enabled"])

    def test_supervised_handoff_preserves_checkpoint_pad_token(self):
        tokenizer = SimpleNamespace(pad_token_id=17, eos_token_id=18)
        self.assertEqual(_padding_token_id(tokenizer), 17)
        tokenizer.pad_token_id = None
        self.assertEqual(_padding_token_id(tokenizer), 18)

    def test_grpo_rejects_nonpositive_world_size(self):
        config = self._grpo_config()
        with self.assertRaisesRegex(GRPOConfigError, "world_size must be positive"):
            validate_grpo_config(config, world_size=0)


if __name__ == "__main__":
    unittest.main()
