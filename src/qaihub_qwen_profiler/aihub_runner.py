from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


def _import_qai_hub():
    try:
        import qai_hub as hub  # type: ignore
        return hub
    except Exception as exc:
        raise RuntimeError(
            "qai_hub is not installed or not configured. Run: pip install '.[aihub]' and "
            "qai-hub configure --api_token YOUR_API_TOKEN"
        ) from exc


def profile_compiled_model_id(compiled_model_id: str, device_name: str, output_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    payload = {
        "compiled_model_id": compiled_model_id,
        "device": device_name,
        "output_dir": str(output_dir),
    }
    if dry_run:
        return {"dry_run": True, **payload}
    hub = _import_qai_hub()
    device = hub.Device(name=device_name)
    compiled_model = hub.get_model(compiled_model_id)
    job = hub.submit_profile_job(model=compiled_model, device=device)
    job.wait()
    return {"profile_job_id": job.job_id, "profile_job_url": getattr(job, "url", None), **payload}


def compile_and_profile_model_artifact(
    model_artifact: Path,
    device_name: str,
    output_dir: Path,
    input_specs: Dict[str, Any] | None = None,
    options: str | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    payload = {
        "model_artifact": str(model_artifact),
        "device": device_name,
        "input_specs": input_specs or {},
        "options": options or "",
        "output_dir": str(output_dir),
    }
    if dry_run:
        return {"dry_run": True, **payload}
    hub = _import_qai_hub()
    device = hub.Device(name=device_name)
    compile_job = hub.submit_compile_job(
        model=str(model_artifact),
        device=device,
        input_specs=input_specs,
        options=options or "",
    )
    compile_job.wait()
    target_model = compile_job.get_target_model()
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    profile_job.wait()
    return {
        "compile_job_id": compile_job.job_id,
        "compile_job_url": getattr(compile_job, "url", None),
        "profile_job_id": profile_job.job_id,
        "profile_job_url": getattr(profile_job, "url", None),
        **payload,
    }


def explain_supported_model_run(model_meta: Any, prompt: str, image_path: Path, image_size: int, device_name: str, precision: str) -> Dict[str, Any]:
    return {
        "mode": "supported",
        "model": asdict(model_meta),
        "prompt": prompt,
        "image_path": str(image_path),
        "image_size": image_size,
        "device": device_name,
        "precision": precision,
        "next_step": (
            "Use the matching Qualcomm AI Hub Models recipe for the slug above. "
            "For many LLM/VLM recipes, Qualcomm keeps export/profile entry points inside "
            "qai_hub_models.models.<slug>. The wrapper prints this resolved configuration so "
            "only --prompt, --image-size, and --model need to change between runs."
        ),
    }
