from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL = ROOT / "neutral"
if str(NEUTRAL) not in sys.path:
    sys.path.insert(0, str(NEUTRAL))

from model_profiles import DENSE_CAUSAL_LM, GPT_OSS, detect_model_profile


class ModelProfileTests(unittest.TestCase):
    def test_gpt_oss_hub_id(self) -> None:
        profile = detect_model_profile("openai/gpt-oss-20b")
        self.assertEqual(profile, GPT_OSS)
        self.assertEqual(profile.native_quantization, "mxfp4")
        self.assertTrue(profile.requires_chat_template)
        self.assertEqual(
            profile.lora_target_modules,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )

    def test_gpt_oss_local_path(self) -> None:
        self.assertEqual(
            detect_model_profile("/models/gpt-oss-20b"),
            GPT_OSS,
        )

    def test_dense_default(self) -> None:
        profile = detect_model_profile("legacy-model/legacy-model-2.5-7B-Instruct")
        self.assertEqual(profile, DENSE_CAUSAL_LM)
        self.assertIsNone(profile.native_quantization)
        self.assertFalse(profile.requires_chat_template)
        self.assertIn("gate_proj", profile.lora_target_modules)


if __name__ == "__main__":
    unittest.main()
