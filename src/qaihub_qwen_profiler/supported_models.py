from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class SupportedModel:
    display_name: str
    qaihub_slug: str
    model_type: str
    default_precision: str
    notes: str

SUPPORTED_MODELS: Dict[str, SupportedModel] = {
    "Qwen2.5-VL-7B-Instruct": SupportedModel(
        display_name="Qwen2.5-VL-7B-Instruct",
        qaihub_slug="qwen2_5_vl_7b_instruct",
        model_type="vlm",
        default_precision="w4a16",
        notes="Supported Qwen vision-language baseline for image+prompt profiling.",
    ),
    "Qwen2.5-7B-Instruct": SupportedModel(
        display_name="Qwen2.5-7B-Instruct",
        qaihub_slug="qwen2_5_7b_instruct",
        model_type="llm",
        default_precision="w4a16",
        notes="Text-only LLM. Image size is ignored unless the model recipe adds image inputs.",
    ),
    "Qwen2-7B-Instruct": SupportedModel(
        display_name="Qwen2-7B-Instruct",
        qaihub_slug="qwen2_7b_instruct",
        model_type="llm",
        default_precision="w4a16",
        notes="Text-only LLM. Image size is ignored unless the model recipe adds image inputs.",
    ),
    "Qwen3.4": SupportedModel(
        display_name="Qwen3.4",
        qaihub_slug="qwen3_4b",
        model_type="llm",
        default_precision="w4a16",
        notes="Text-only Qwen3-family model from AI Hub Models.",
    ),
    "Qwen3-4B-Instruct-2507": SupportedModel(
        display_name="Qwen3-4B-Instruct-2507",
        qaihub_slug="qwen3_4b_instruct_2507",
        model_type="llm",
        default_precision="w4a16",
        notes="Text-only Qwen3-family instruct model from AI Hub Models.",
    ),
}

def list_models() -> str:
    lines = []
    for name, meta in SUPPORTED_MODELS.items():
        lines.append(f"{name} ({meta.model_type}, default {meta.default_precision}) - {meta.notes}")
    return "\n".join(lines)

def resolve_supported_model(name: str) -> SupportedModel:
    if name not in SUPPORTED_MODELS:
        valid = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{name}'. Valid names: {valid}")
    return SUPPORTED_MODELS[name]
