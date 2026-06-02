from pathlib import Path
from huggingface_hub import snapshot_download


def get_or_download_hf_snapshot(model_id: str, cache_dir: Path, revision: str | None = None) -> Path:
    """Reuse a local Hugging Face snapshot when present; otherwise download it once."""
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        return Path(path)
    except Exception:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        return Path(path)
