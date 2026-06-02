from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .cache import get_or_download_hf_snapshot


def build_qwen_vl_inputs(model_id: str, prompt: str, image_path: Path, cache_dir: Path) -> Dict[str, Any]:
    """Build processor inputs for Qwen-VL/Qwen2.5-VL-style models.

    This function intentionally imports transformers lazily because the base CLI can be used
    without heavyweight export dependencies.
    """
    try:
        from PIL import Image
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("Install export extras first: pip install '.[export]'") from exc

    local_model = get_or_download_hf_snapshot(model_id, cache_dir)
    processor = AutoProcessor.from_pretrained(local_model, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {"local_model_path": str(local_model), "input_keys": list(inputs.keys()), "inputs": inputs}


def export_qwen_vl_to_onnx(model_id: str, prompt: str, image_path: Path, cache_dir: Path, output_dir: Path) -> Path:
    """Export a custom Qwen-VL model to ONNX with external data.

    Large VLM exports can exceed 2 GB; external data keeps weights next to the .onnx file.
    The exact tensor set may vary by model revision, so this creates a manifest that records
    the generated processor inputs for troubleshooting.
    """
    try:
        import torch
        from transformers import AutoModelForVision2Seq
    except Exception as exc:
        raise RuntimeError("Install export extras first: pip install '.[export]'") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    built = build_qwen_vl_inputs(model_id, prompt, image_path, cache_dir)
    local_model = Path(built["local_model_path"])
    inputs = built["inputs"]
    model = AutoModelForVision2Seq.from_pretrained(
        local_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    onnx_path = output_dir / "custom_qwen_vl.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            str(onnx_path),
            input_names=list(inputs.keys()),
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            external_data=True,
        )
    manifest = {
        "model_id": model_id,
        "local_model_path": str(local_model),
        "prompt": prompt,
        "image_path": str(image_path),
        "input_keys": built["input_keys"],
        "onnx_path": str(onnx_path),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return onnx_path
