import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nar_tts.integrations.llama_factory import (
    load_llama_factory_config,
    resolve_runtime_values,
    validate_llama_factory_config,
)
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
    def test_launch_config_names_are_backend_neutral(self):
        launch = CONFIGS / "launch"
        self.assertEqual(
            {path.name for path in launch.glob("*.yaml")},
            {"single_gpu.yaml", "multi_gpu.yaml", "fsdp.yaml"},
        )
        self.assertFalse((CONFIGS / "accelerate_grpo_1gpu.yaml").exists())

    def test_sglang_server_is_fully_derived_from_grpo_yaml(self):
        command = server_command(load_grpo_config(CONFIGS / "grpo_sglang.yaml"))
        self.assertIn("sglang.launch_server", command)
        self.assertIn("--enable-custom-logit-processor", command)
        self.assertIn("--enable-lora", command)
        self.assertEqual(command[command.index("--tp-size") + 1], "1")

    def test_llama_factory_resolves_all_cpu_worker_fields(self):
        config = load_llama_factory_config(
            CONFIGS / "llama_factory" / "language_retention_sft.yaml"
        )
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            resolved = resolve_runtime_values(config)
        self.assertEqual(resolved["preprocessing_num_workers"], available_cpu_count())
        self.assertEqual(
            resolved["dataloader_num_workers"],
            max(1, available_cpu_count() // 2),
        )
        self.assertEqual(resolved["stage"], "sft")
        self.assertTrue(resolved["use_unsloth"])

    def test_cpu_count_falls_back_when_affinity_is_unavailable(self):
        with (
            patch("os.sched_getaffinity", side_effect=OSError),
            patch("os.cpu_count", return_value=12),
        ):
            self.assertEqual(available_cpu_count(), 12)

    def test_llama_factory_does_not_claim_an_unsupported_grpo_stage(self):
        config = load_llama_factory_config(
            CONFIGS / "llama_factory" / "language_retention_sft.yaml"
        )
        config["stage"] = "grpo"
        with self.assertRaisesRegex(ValueError, "does not provide native GRPO"):
            validate_llama_factory_config(config)

    def test_unsloth_has_a_dedicated_supervised_scenario(self):
        config = load_finetune_config(CONFIGS / "finetune_unsloth.yaml")
        self.assertEqual(config["model"]["loader"], "unsloth")
        self.assertTrue(config["model"]["peft"]["enabled"])
        self.assertEqual(config["runtime"]["dataloader_workers"], "auto")

    def test_supervised_handoff_preserves_checkpoint_pad_token(self):
        tokenizer = SimpleNamespace(pad_token_id=17, eos_token_id=18)
        self.assertEqual(_padding_token_id(tokenizer), 17)
        tokenizer.pad_token_id = None
        self.assertEqual(_padding_token_id(tokenizer), 18)

    def test_grpo_rejects_nonpositive_world_size(self):
        config = load_grpo_config(CONFIGS / "grpo_intelligibility.yaml")
        with self.assertRaisesRegex(GRPOConfigError, "world_size must be positive"):
            validate_grpo_config(config, world_size=0)


if __name__ == "__main__":
    unittest.main()
