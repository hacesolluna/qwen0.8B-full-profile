from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aihub_runner import compile_and_profile_model_artifact, explain_supported_model_run, profile_compiled_model_id
from .defaults import DEFAULT_CACHE_DIR, DEFAULT_CUSTOM_MODEL, DEFAULT_DEVICE, DEFAULT_IMAGE_SIZE, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPT, DEFAULT_SUPPORTED_MODEL
from .export_custom import export_qwen_vl_to_onnx
from .image_utils import prepare_image
from .supported_models import list_models, resolve_supported_model


def _print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def add_common_args(parser):
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt. Default: %(default)r")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Square image size in pixels. Default: %(default)s")
    parser.add_argument("--image", default=None, help="Optional input image. If omitted, a deterministic random-noise dummy image is used.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="AI Hub device name. Default: %(default)r")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Run output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved actions without calling AI Hub.")


def supported_cmd(args):
    if args.list_models:
        print(list_models())
        return
    model = resolve_supported_model(args.model)
    run_dir = args.output_dir / "supported" / model.qaihub_slug
    image_path = prepare_image(args.image, run_dir, args.image_size)
    payload = explain_supported_model_run(
        model_meta=model,
        prompt=args.prompt,
        image_path=image_path,
        image_size=args.image_size,
        device_name=args.device,
        precision=args.precision or model.default_precision,
    )
    _print_json(payload)
    if not args.dry_run:
        print("\nUse the QAI Hub Models recipe for the printed slug, or pass a compiled model ID to:")
        print("qaihub-qwen-profile profile-id --compiled-model-id <MODEL_ID>")


def profile_id_cmd(args):
    payload = profile_compiled_model_id(args.compiled_model_id, args.device, args.output_dir, dry_run=args.dry_run)
    _print_json(payload)


def custom_cmd(args):
    run_dir = args.output_dir / "custom" / args.model.replace("/", "__")
    image_path = prepare_image(args.image, run_dir, args.image_size)
    if args.compiled_model_id:
        payload = profile_compiled_model_id(args.compiled_model_id, args.device, run_dir, dry_run=args.dry_run)
        _print_json(payload)
        return
    if args.model_artifact:
        payload = compile_and_profile_model_artifact(Path(args.model_artifact), args.device, run_dir, options=args.options, dry_run=args.dry_run)
        _print_json(payload)
        return
    if args.dry_run:
        _print_json({
            "dry_run": True,
            "mode": "custom",
            "model": args.model,
            "prompt": args.prompt,
            "image_size": args.image_size,
            "image_path": str(image_path),
            "cache_dir": str(args.cache_dir),
            "planned_export": str(run_dir / "onnx" / "custom_qwen_vl.onnx"),
            "planned_steps": [
                "check Hugging Face cache",
                "download missing weights only if needed",
                "build processor inputs from prompt and resized image",
                "export ONNX with external data",
                "compile and profile on AI Hub",
            ],
        })
        return
    onnx_path = export_qwen_vl_to_onnx(args.model, args.prompt, image_path, args.cache_dir, run_dir / "onnx")
    payload = compile_and_profile_model_artifact(onnx_path, args.device, run_dir, options=args.options, dry_run=args.dry_run)
    _print_json(payload)


def build_parser():
    parser = argparse.ArgumentParser(description="Profile Qwen/Qwen-VL models on Qualcomm AI Hub with simple prompt/image/model knobs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_supported = sub.add_parser("supported", help="Resolve and prepare a supported AI Hub Qwen model run.")
    add_common_args(p_supported)
    p_supported.add_argument("--model", default=DEFAULT_SUPPORTED_MODEL, help="Supported model display name. Default: %(default)r")
    p_supported.add_argument("--precision", default=None, help="Quantization/profile precision, e.g. w4a16.")
    p_supported.add_argument("--list-models", action="store_true", help="Print supported names and exit.")
    p_supported.set_defaults(func=supported_cmd)

    p_custom = sub.add_parser("custom", help="Export/compile/profile a custom Hugging Face or local artifact model.")
    add_common_args(p_custom)
    p_custom.add_argument("--model", default=DEFAULT_CUSTOM_MODEL, help="Hugging Face repo ID or local model path. Default: %(default)r")
    p_custom.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="HF and export cache directory.")
    p_custom.add_argument("--model-artifact", default=None, help="Existing ONNX/TorchScript artifact to compile/profile instead of exporting.")
    p_custom.add_argument("--compiled-model-id", default=None, help="Existing compiled AI Hub model ID to profile only.")
    p_custom.add_argument("--options", default="", help="Optional AI Hub compile options string.")
    p_custom.set_defaults(func=custom_cmd)

    p_profile = sub.add_parser("profile-id", help="Profile an existing compiled AI Hub model ID.")
    p_profile.add_argument("--compiled-model-id", required=True)
    p_profile.add_argument("--device", default=DEFAULT_DEVICE)
    p_profile.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_profile.add_argument("--dry-run", action="store_true")
    p_profile.set_defaults(func=profile_id_cmd)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
