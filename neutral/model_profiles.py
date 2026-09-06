from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    name: str
    lora_target_modules: tuple[str, ...]
    native_quantization: str | None
    requires_chat_template: bool
    supports_bitsandbytes_qlora: bool


GPT_OSS = ModelProfile(
    name="gpt-oss",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    native_quantization="mxfp4",
    requires_chat_template=True,
    supports_bitsandbytes_qlora=False,
)

DENSE_CAUSAL_LM = ModelProfile(
    name="dense-causal-lm",
    lora_target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    native_quantization=None,
    requires_chat_template=False,
    supports_bitsandbytes_qlora=True,
)


def detect_model_profile(model_ref: str) -> ModelProfile:
    normalized = str(model_ref).replace("\\", "/").lower().rstrip("/")
    basename = Path(normalized).name

    if "gpt-oss" in normalized or basename.startswith("gpt_oss"):
        return GPT_OSS

    return DENSE_CAUSAL_LM
