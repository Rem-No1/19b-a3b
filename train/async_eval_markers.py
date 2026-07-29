"""Atomic checkpoint markers consumed by the asynchronous vLLM evaluator."""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback


CHECKPOINT_READY_NAME = ".async_eval_ready.json"
CHECKPOINT_DONE_NAME = ".async_eval_done.json"
CHECKPOINT_FAILED_NAME = ".async_eval_failed.json"
FINAL_READY_NAME = ".async_eval_final_ready.json"
FINAL_DONE_NAME = ".async_eval_final_done.json"
FINAL_FAILED_NAME = ".async_eval_final_failed.json"
TRAINING_COMPLETE_NAME = ".async_eval_training_complete.json"
CHECKPOINT_METADATA_NAMES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "expert_prune_metadata.json",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_huggingface_weight_files(model_dir):
    model_dir = Path(model_dir)
    candidates = []
    for pattern in (
        "model.safetensors",
        "model-*-of-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*-of-*.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ):
        candidates.extend(model_dir.glob(pattern))
    return sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file() and path.stat().st_size > 0
        }
    )


def copy_checkpoint_metadata(metadata_source, checkpoint_dir):
    """Copy non-weight model metadata that Trainer does not save per checkpoint."""
    if metadata_source is None:
        return []
    source = Path(metadata_source)
    checkpoint_dir = Path(checkpoint_dir)
    copied = []
    for name in CHECKPOINT_METADATA_NAMES:
        source_file = source / name
        if source_file.is_file():
            target_file = checkpoint_dir / name
            shutil.copy2(source_file, target_file)
            copied.append(target_file)
    return copied


def build_ready_payload(model_dir, global_step, run_name, kind):
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.json"
    weight_files = find_huggingface_weight_files(model_dir)
    if not config_path.is_file():
        raise RuntimeError(f"异步验证模型缺少 config.json: {model_dir}")
    if not weight_files:
        raise RuntimeError(f"异步验证模型缺少 Hugging Face 权重: {model_dir}")
    return {
        "schema_version": 1,
        "kind": kind,
        "global_step": int(global_step),
        "run_name": str(run_name),
        "model_path": str(model_dir),
        "ready_at": utc_now(),
        "weight_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
            }
            for path in weight_files
        ],
    }


def mark_checkpoint_ready(output_dir, global_step, run_name):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{int(global_step)}"
    payload = build_ready_payload(
        checkpoint_dir,
        global_step=global_step,
        run_name=run_name,
        kind="checkpoint",
    )
    marker_path = checkpoint_dir / CHECKPOINT_READY_NAME
    atomic_write_json(marker_path, payload)
    return marker_path


def mark_final_model_ready(output_dir, global_step, run_name):
    output_dir = Path(output_dir)
    checkpoint_marker = (
        output_dir / f"checkpoint-{int(global_step)}" / CHECKPOINT_READY_NAME
    )
    if checkpoint_marker.is_file():
        return None
    payload = build_ready_payload(
        output_dir,
        global_step=global_step,
        run_name=run_name,
        kind="final",
    )
    marker_path = output_dir / FINAL_READY_NAME
    atomic_write_json(marker_path, payload)
    return marker_path


def mark_training_complete(output_dir, global_step, run_name):
    marker_path = Path(output_dir) / TRAINING_COMPLETE_NAME
    atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "global_step": int(global_step),
            "run_name": str(run_name),
            "completed_at": utc_now(),
        },
    )
    return marker_path


class AsyncEvalMarkerCallback(TrainerCallback):
    """Publish a checkpoint only after Trainer has completely saved it."""

    def __init__(self, metadata_source=None):
        self.metadata_source = (
            str(Path(metadata_source).resolve())
            if metadata_source is not None
            else None
        )

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        if not getattr(state, "is_world_process_zero", False):
            return control
        try:
            checkpoint_dir = (
                Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            )
            copied = copy_checkpoint_metadata(
                self.metadata_source,
                checkpoint_dir,
            )
            marker_path = mark_checkpoint_ready(
                args.output_dir,
                global_step=state.global_step,
                run_name=args.run_name,
            )
            if copied:
                print(
                    "[async-eval] checkpoint metadata: "
                    + ", ".join(path.name for path in copied),
                    flush=True,
                )
            print(f"[async-eval] checkpoint ready: {marker_path}", flush=True)
        except Exception as exc:  # Do not strand the other distributed ranks.
            print(
                f"[async-eval][error] checkpoint-{state.global_step} "
                f"未发布 ready marker: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return control
