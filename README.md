# Qualcomm AI Hub On-Device Simulation

## 0. Logistics

Device Name: `QCS8550 (Proxy)`

This codebase abstracts profiling behind the same three knobs requested in the report:

1. input text prompt, default chair prompt shown below;
2. input image size, default `256`;
3. model used.

Default chair prompt:

```text
Look at the chairs in this image.
Are all visible chairs pushed in and neatly tucked under the table?
Return exactly one valid JSON object and no markdown or explanation. The first character must be "{" and the last character must be "}".
Schema: {"answer":"yes"} or {"answer":"no"}
```

Default image behavior:

- If `--image` is not supplied, the wrapper creates a deterministic random-noise dummy image at the requested `--image-size`.
- If `--image /path/to/image.jpg` is supplied, the wrapper converts that image to RGB, resizes it to `--image-size x --image-size`, and uses it for the run.

## 1. Set-Up

On browser:

1. Set up an account on Qualcomm AI Hub.
2. Acquire a personal API token.

On device:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[aihub]'
qai-hub configure --api_token YOUR_API_TOKEN
```

For custom Hugging Face export/profile workflows:

```bash
pip install -e '.[export]'
```

For the full workflow:

```bash
pip install -e '.[aihub,export]'
```

## 2. Profiling Supported Models

Supported Qwen Models on AI Hub:

- Qwen 3.5 is not available as a supported AI Hub model.
- Only Qwen VLM available: `Qwen2.5-VL-7B-Instruct`; all other listed Qwen models are LLMs.
- Custom models can still be exported from Hugging Face and sent through the custom path in Section 3.

Available quantization:

- `Qwen2.5-7B-Instruct`: `w4a16` + `w8a16` for a few layers.
- `Qwen2.5-VL-7B-Instruct`: `w4a16`.
- `Qwen2-7B-Instruct`: `w4a16` + `w8a16` for a few layers.
- `Qwen3.4`: `w4a16`.
- `Qwen3-4B-Instruct-2507`: `w4a16`.

`w4a16` means 4-bit integer weights and 16-bit float activations.

### Instructions for variable prompt, image size, model, and optional image

Use the supported-model wrapper when the target model exists in Qualcomm AI Hub Models:

```bash
qaihub-qwen-profile supported \
  --model Qwen2.5-VL-7B-Instruct \
  --image-size 256
```

This uses the default chair prompt and a default random-noise dummy image.

To provide a custom prompt:

```bash
qaihub-qwen-profile supported \
  --prompt "What object is in the image?" \
  --image-size 512
```

To provide a custom image:

```bash
qaihub-qwen-profile supported \
  --image /path/to/chairs.jpg \
  --image-size 256
```

To list supported names:

```bash
qaihub-qwen-profile supported --list-models
```

Shell shortcut:

```bash
./scripts/profile_supported.sh
MODEL=Qwen2.5-VL-7B-Instruct IMAGE_SIZE=512 ./scripts/profile_supported.sh
IMAGE=/path/to/chairs.jpg ./scripts/profile_supported.sh
```

What the wrapper does:

- It validates the model name in `src/qaihub_qwen_profiler/supported_models.py`.
- It creates `runs/supported/<model_slug>/default_noise_<size>.png` unless `--image` is supplied.
- It prints the resolved AI Hub model slug, device, precision, prompt, image path, and next action.
- It keeps most experiments limited to changing `--prompt`, `--image-size`, `--model`, and optionally `--image`.

## 3. Profiling Custom Models

Use this path when the target model comes from Hugging Face or from your own exported artifact.

The default custom model is:

```text
Qwen/Qwen3.5-0.8B
```

Basic command:

```bash
qaihub-qwen-profile custom \
  --image-size 256
```

This uses:

- default prompt: the chair JSON prompt shown in Section 0;
- default image: deterministic random-noise dummy image of size `256 x 256`;
- default custom model: `Qwen/Qwen3.5-0.8B`.

To change all three core parameters:

```bash
qaihub-qwen-profile custom \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Look at the chairs in this image. Are all visible chairs pushed in? Return JSON only." \
  --image-size 512
```

To use a real custom image while keeping the same model and prompt:

```bash
qaihub-qwen-profile custom \
  --image /path/to/chairs.jpg \
  --image-size 256
```

Shell shortcut:

```bash
./scripts/profile_custom.sh
MODEL=Qwen/Qwen3.5-0.8B IMAGE_SIZE=512 ./scripts/profile_custom.sh
IMAGE=/path/to/chairs.jpg ./scripts/profile_custom.sh
```

The custom path dynamically reuses cached model weights:

- It checks `.cache/qaihub-qwen-profiler` for an existing Hugging Face snapshot first.
- It downloads weights only when the snapshot is missing.
- It builds processor inputs from the prompt and prepared image.
- It exports ONNX with external data into `runs/custom/<model>/onnx/`.
- It compiles and profiles the artifact on Qualcomm AI Hub.

For an already exported ONNX/TorchScript artifact:

```bash
qaihub-qwen-profile custom \
  --model-artifact /path/to/model.onnx \
  --image-size 256
```

For an already compiled AI Hub model, skip compile and only profile:

```bash
qaihub-qwen-profile profile-id --compiled-model-id mqerzgk7n
```

Dry runs are recommended before launching long compile/profile jobs:

```bash
qaihub-qwen-profile supported --dry-run
qaihub-qwen-profile custom --dry-run
qaihub-qwen-profile profile-id --compiled-model-id mqerzgk7n --dry-run
```

## 4. Project Layout

```text
qaihub_qwen_profiler/
  README.md
  pyproject.toml
  src/qaihub_qwen_profiler/
    aihub_runner.py      # QAI Hub compile/profile calls
    cache.py             # Hugging Face cache reuse/download logic
    cli.py               # command line interface
    defaults.py          # default prompt, image size, device, model names, directories
    export_custom.py     # custom Qwen/Qwen-VL ONNX export path
    image_utils.py       # default noise image + custom image resizing
    supported_models.py  # supported Qwen model registry
  scripts/
    profile_supported.sh
    profile_custom.sh
  examples/
    custom_qwen_vl_dry_run.sh
  docs/
    Qualcomm_AI_Hub_Report_modified.pdf
```

## 5. Notes from the Previous Qwen / QAI Hub Work

The custom path is intentionally conservative because large VLM exports can hit artifact-size and external-weight packaging issues.

- Use `--dry-run` first to confirm prompt, image size, model, image path, cache path, and output paths.
- Use `profile-id` when you already have a compiled model ID, such as `mqerzgk7n`.
- Keep ONNX external data files beside the `.onnx` file when uploading or compiling.
- If a full end-to-end VLM is too large to profile, split the pipeline into smaller artifacts, such as vision encoder, prompt processor, and language model blocks.
- If your processor produces `int64` inputs, AI Hub compilation may require 64-bit truncation options depending on runtime.
