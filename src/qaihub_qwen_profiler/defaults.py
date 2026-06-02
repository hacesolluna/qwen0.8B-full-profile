from pathlib import Path

DEFAULT_PROMPT = 'Look at the chairs in this image.\nAre all visible chairs pushed in and neatly tucked under the table?\nReturn exactly one valid JSON object and no markdown or explanation. The first character must be "{" and the last character must be "}".\nSchema: {"answer":"yes"} or {"answer":"no"}'
DEFAULT_IMAGE_SIZE = 256
DEFAULT_DEVICE = "QCS8550 (Proxy)"
DEFAULT_SUPPORTED_MODEL = "Qwen2.5-VL-7B-Instruct"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_CACHE_DIR = Path(".cache/qaihub-qwen-profiler")
DEFAULT_OUTPUT_DIR = Path("runs")
