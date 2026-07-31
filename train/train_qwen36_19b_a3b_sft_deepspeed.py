"""DeepSpeed ZeRO-3 SFT for the pruned Qwen3.6/Qwen3.5-35B-A3B checkpoint.

The input checkpoint is a complete ``Qwen3_5MoeForConditionalGeneration``
model with 128 routed experts (about 19B parameters). This trainer performs
text/tool-chat SFT, supervises assistant tokens only, and freezes the unused
vision tower by default. Full-parameter SFT is the default so the saved Hugging
Face checkpoint can be passed directly to the later GRPO launcher.
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolchat_data import (  # noqa: E402
    collect_data_files,
    encode_messages,
    labels_from_assistant_masks,
    sample_data_files,
    think_marker_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/model"
DEFAULT_OUTPUT_DIR = "/output"
DEFAULT_DEEPSPEED = REPO_ROOT / "train" / "ds_config" / "qwen36_19b_a3b_zero3.json"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EXPECTED_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
IGNORE_INDEX = -100
DEFAULT_DDP_TIMEOUT_SECONDS = 86400
MANIFEST_BATCH_SIZE = 10000
MANIFEST_COLUMNS = (
    "source_file_index",
    "last_assistant_only",
    "n_tokens",
    "has_loss",
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Qwen3.6-19B-A3B mixed-thinking tool-chat SFT with DeepSpeed ZeRO-3."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-files", nargs="+", required=True)
    parser.add_argument(
        "--eval-data-files",
        nargs="+",
        default=None,
        help="Optional validation JSON/JSONL files. Enables step-based eval_loss.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deepspeed", default=str(DEFAULT_DEEPSPEED))
    parser.add_argument("--expected-num-experts", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=16000)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument(
        "--max-samples-per-file",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-eval-samples-per-file",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument("--dataset-num-proc", type=int, default=8)
    parser.add_argument(
        "--ddp-timeout",
        type=int,
        default=DEFAULT_DDP_TIMEOUT_SECONDS,
        help=(
            "Distributed operation timeout in seconds. Large dataset preprocessing "
            "runs under main_process_first, so non-main ranks may wait for hours."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--attn-implementation", default="flash_attention_2")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=70,
        help="Evaluate every N optimizer/global steps when --eval-data-files is set.",
    )
    parser.add_argument(
        "--eval-metric",
        choices=("loss", "pass_rate"),
        default="loss",
        help="loss uses teacher forcing; pass_rate generates final answers and scores them.",
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--eval-generation-enable-thinking",
        type=parse_bool,
        default=False,
        help="Enable Qwen thinking mode only for pass_rate generation.",
    )
    parser.add_argument("--save-steps", type=int, default=70)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--save-only-model", type=parse_bool, default=False)
    parser.add_argument(
        "--async-eval-markers",
        type=parse_bool,
        default=False,
        help=(
            "Publish atomic checkpoint-ready markers for a separate vLLM worker. "
            "Cannot be combined with inline --eval-data-files."
        ),
    )
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--freeze-vision-tower", type=parse_bool, default=True)

    parser.add_argument("--use-lora", type=parse_bool, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--use-rslora", type=parse_bool, default=False)

    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="qwen36-19b-a3b-sft")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--enable-thinking", type=parse_bool, default=True)
    parser.add_argument(
        "--enable-thinking-per-file",
        nargs="+",
        type=parse_bool,
        default=None,
    )
    parser.add_argument(
        "--eval-enable-thinking-per-file",
        nargs="+",
        type=parse_bool,
        default=None,
    )
    parser.add_argument(
        "--last-assistant-only",
        type=parse_bool,
        default=False,
        help=(
            "Global fallback: False supervises every assistant turn, including TIR "
            "reasoning and tool calls; True keeps only the final assistant turn."
        ),
    )
    parser.add_argument(
        "--eval-last-assistant-only-per-file",
        nargs="+",
        type=parse_bool,
        default=None,
        help=(
            "Optional boolean for each --eval-data-files entry. Defaults to "
            "--last-assistant-only when omitted."
        ),
    )
    parser.add_argument(
        "--last-assistant-only-per-file",
        nargs="+",
        type=parse_bool,
        default=None,
        help=(
            "Optional boolean for each --data-files entry. Each value overrides "
            "--last-assistant-only for that file. A row-level last_assistant_only "
            "field has the highest priority."
        ),
    )
    parser.add_argument("--mask-empty-think", type=parse_bool, default=True)
    parser.add_argument("--preflight-only", type=parse_bool, default=False)
    parser.add_argument("--skip-gpu-check", type=parse_bool, default=False)
    return parser


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def normalize_args(args):
    if args.ddp_timeout <= 0:
        raise ValueError("--ddp-timeout 必须大于 0")
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        args.max_eval_samples = None
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.use_lora else 5e-6
    return args


def validate_checkpoint_config(model_path, expected_num_experts=128):
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError(f"模型目录不存在: {model_path}")
    required = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise ValueError(f"模型目录缺少文件: {missing}")

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    if architectures != [EXPECTED_ARCHITECTURE]:
        raise ValueError(
            f"architectures={architectures}，预期 [{EXPECTED_ARCHITECTURE!r}]"
        )
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError(f"不支持的 model_type={config.get('model_type')}")
    text_config = config.get("text_config", {})
    num_experts = int(text_config.get("num_experts", 0))
    experts_per_token = int(text_config.get("num_experts_per_tok", 0))
    if num_experts != int(expected_num_experts):
        raise ValueError(f"num_experts={num_experts}，预期 {expected_num_experts}")
    if not 0 < experts_per_token <= num_experts:
        raise ValueError(
            f"非法专家路由配置: num_experts_per_tok={experts_per_token}, num_experts={num_experts}"
        )

    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map", {})
    if any(key.startswith("mtp.") or ".mtp." in key for key in weight_map):
        raise ValueError("检查点仍包含 MTP 权重，与当前剪枝模型元数据不一致")
    missing_shards = sorted(
        {filename for filename in weight_map.values() if not (model_path / filename).is_file()}
    )
    if missing_shards:
        raise ValueError(f"模型分片缺失: {missing_shards}")
    total_parameters = int(index.get("metadata", {}).get("total_parameters", 0))
    return {
        "architecture": architectures[0],
        "num_experts": num_experts,
        "num_experts_per_tok": experts_per_token,
        "num_hidden_layers": int(text_config.get("num_hidden_layers", 0)),
        "total_parameters": total_parameters,
    }


def validate_data_arguments(args):
    missing = [path for path in args.data_files if not Path(path).is_file()]
    if missing:
        raise ValueError(f"训练数据文件不存在: {missing}")
    if (
        args.max_samples_per_file is not None
        and len(args.max_samples_per_file) != len(args.data_files)
    ):
        raise ValueError("--max-samples-per-file 的数量必须与 --data-files 相同")
    if (
        args.enable_thinking_per_file is not None
        and len(args.enable_thinking_per_file) != len(args.data_files)
    ):
        raise ValueError("--enable-thinking-per-file 的数量必须与 --data-files 相同")
    if (
        args.last_assistant_only_per_file is not None
        and len(args.last_assistant_only_per_file) != len(args.data_files)
    ):
        raise ValueError("--last-assistant-only-per-file 的数量必须与 --data-files 相同")

    if args.eval_data_files is not None:
        if args.async_eval_markers:
            raise ValueError(
                "--async-eval-markers true 不能与训练进程内的 "
                "--eval-data-files 同时使用"
            )
        missing_eval = [path for path in args.eval_data_files if not Path(path).is_file()]
        if missing_eval:
            raise ValueError(f"验证数据文件不存在: {missing_eval}")
        if args.eval_steps <= 0:
            raise ValueError("--eval-steps 必须大于 0")
        if args.per_device_eval_batch_size <= 0:
            raise ValueError("--per-device-eval-batch-size 必须大于 0")
        if args.eval_metric == "pass_rate" and args.eval_max_new_tokens <= 0:
            raise ValueError("--eval-max-new-tokens 必须大于 0")
        if (
            args.eval_metric == "pass_rate"
            and args.eval_max_new_tokens >= args.max_seq_length
        ):
            raise ValueError("--eval-max-new-tokens 必须小于 --max-seq-length")
        if (
            args.max_eval_samples_per_file is not None
            and len(args.max_eval_samples_per_file) != len(args.eval_data_files)
        ):
            raise ValueError(
                "--max-eval-samples-per-file 的数量必须与 --eval-data-files 相同"
            )
        if (
            args.eval_enable_thinking_per_file is not None
            and len(args.eval_enable_thinking_per_file) != len(args.eval_data_files)
        ):
            raise ValueError(
                "--eval-enable-thinking-per-file 的数量必须与 --eval-data-files 相同"
            )
        if (
            args.eval_last_assistant_only_per_file is not None
            and len(args.eval_last_assistant_only_per_file) != len(args.eval_data_files)
        ):
            raise ValueError(
                "--eval-last-assistant-only-per-file 的数量必须与 --eval-data-files 相同"
            )
        train_paths = {Path(path).resolve() for path in args.data_files}
        eval_paths = {Path(path).resolve() for path in args.eval_data_files}
        overlap = sorted(str(path) for path in train_paths & eval_paths)
        if overlap:
            raise ValueError(f"训练集和验证集不能使用相同文件: {overlap}")
    elif any(
        value is not None
        for value in (
            args.max_eval_samples,
            args.max_eval_samples_per_file,
            args.eval_enable_thinking_per_file,
            args.eval_last_assistant_only_per_file,
        )
    ):
        raise ValueError("验证集参数需要同时提供 --eval-data-files")

    if not Path(args.deepspeed).is_file():
        raise ValueError(f"DeepSpeed 配置不存在: {args.deepspeed}")


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def runtime_preflight(args):
    summary = validate_checkpoint_config(args.model_path, args.expected_num_experts)
    validate_data_arguments(args)
    from packaging.version import Version

    versions = {name: package_version(name) for name in ("deepspeed", "datasets", "accelerate")}
    problems = [f"{name} 未安装" for name, installed in versions.items() if installed is None]
    if versions["datasets"] is not None and Version(versions["datasets"]) < Version("2.15.0"):
        problems.append(f"datasets=={versions['datasets']} 过旧")
    if problems:
        raise RuntimeError(
            f"训练环境依赖不满足: {problems}。请在 verl-qwen36 环境执行: "
            "UV_LINK_MODE=copy uv pip install --upgrade 'datasets==5.0.0' deepspeed"
        )
    try:
        import datasets
        import deepspeed
        import transformers
        from transformers import Qwen3_5MoeForConditionalGeneration

        del Qwen3_5MoeForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            f"训练依赖导入失败: {type(exc).__name__}: {exc}。若出现 PyExtensionType，执行: "
            "UV_LINK_MODE=copy uv pip install --upgrade 'datasets==5.0.0'"
        ) from exc
    if args.attn_implementation == "flash_attention_2":
        try:
            import flash_attn

            del flash_attn
        except Exception as exc:
            raise RuntimeError(f"FlashAttention 导入失败: {exc}") from exc
    if not args.skip_gpu_check:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；静态检查可传 --skip-gpu-check true")
        if torch.cuda.device_count() < int(os.environ.get("LOCAL_WORLD_SIZE", "1")):
            raise RuntimeError("可见 GPU 数少于 LOCAL_WORLD_SIZE")
    rank0_print(
        "[preflight OK]",
        f"architecture={summary['architecture']}",
        f"parameters={summary['total_parameters'] / 1e9:.3f}B",
        f"experts={summary['num_experts']}",
        f"datasets={datasets.__version__}",
        f"deepspeed={deepspeed.__version__}",
        f"transformers={transformers.__version__}",
    )
    return summary


def freeze_vision_tower(model):
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise RuntimeError("没有找到 model.visual，无法安全冻结视觉塔")
    frozen = 0
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
        frozen += getattr(parameter, "ds_numel", None) or parameter.numel()
    return frozen


def apply_lora_if_requested(args, model):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("--use-lora true 需要安装 peft") from exc
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=args.use_rslora,
    )
    model = get_peft_model(model, config, autocast_adapter_dtype=False)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if is_main_process():
        model.print_trainable_parameters()
    return model


def normalize_enable_thinking_per_file(data_files, flags, default):
    if flags is None:
        return [bool(default)] * len(data_files)
    if len(flags) != len(data_files):
        raise ValueError("thinking flags 与 data files 数量不一致")
    return [parse_bool(value) for value in flags]


def normalize_last_assistant_only_per_file(data_files, flags, default):
    if flags is None:
        return [bool(default)] * len(data_files)
    if len(flags) != len(data_files):
        raise ValueError("last-assistant-only flags 与 data files 数量不一致")
    return [parse_bool(value) for value in flags]


def dataset_options(args, split):
    if split == "train":
        return {
            "split": split,
            "data_files": args.data_files,
            "max_samples": args.max_samples,
            "max_samples_per_file": args.max_samples_per_file,
            "enable_thinking_per_file": args.enable_thinking_per_file,
            "last_assistant_only_per_file": args.last_assistant_only_per_file,
        }
    if split == "eval" and args.eval_data_files is not None:
        return {
            "split": split,
            "data_files": args.eval_data_files,
            "max_samples": args.max_eval_samples,
            "max_samples_per_file": args.max_eval_samples_per_file,
            "enable_thinking_per_file": args.eval_enable_thinking_per_file,
            "last_assistant_only_per_file": args.eval_last_assistant_only_per_file,
        }
    raise ValueError(f"无法为 split={split!r} 构建数据选项")


def sample_rows_with_thinking(args, options):
    data_files = collect_data_files(options["data_files"])
    thinking_flags = normalize_enable_thinking_per_file(
        data_files, options["enable_thinking_per_file"], args.enable_thinking
    )
    last_assistant_only_flags = normalize_last_assistant_only_per_file(
        data_files,
        options["last_assistant_only_per_file"],
        args.last_assistant_only,
    )
    results = sample_data_files(data_files, options["max_samples_per_file"], args.seed)
    rows = []
    for file_index, (result, thinking_flag, last_assistant_only_flag) in enumerate(
        zip(results, thinking_flags, last_assistant_only_flags)
    ):
        for sampled in result.records:
            row_thinking = sampled.row.get("enable_thinking", thinking_flag)
            row_last_assistant_only = sampled.row.get(
                "last_assistant_only", last_assistant_only_flag
            )
            rows.append(
                {
                    "raw_row": sampled.row,
                    "enable_thinking": parse_bool(row_thinking),
                    "last_assistant_only": parse_bool(row_last_assistant_only),
                    "source_file_index": file_index,
                    "source_record_number": sampled.source_record_number,
                }
            )
    if options["max_samples"] is not None:
        rows = rows[: options["max_samples"]]
    return results, rows


def build_sampling_manifest(args, options, sampling_results, tokenized_rows):
    files = []
    for result in sampling_results:
        files.append(
            {
                "path": result.source_path,
                "size_bytes": result.source_size_bytes,
                "mtime_ns": result.source_mtime_ns,
                "selection_mode": result.selection_mode,
                "configured_limit": result.configured_limit,
                "source_record_count": result.source_record_count,
                "selected_count": len(result.records),
                "kept_count": 0,
                "selected_all_assistant_turns": 0,
                "selected_last_assistant_only": 0,
                "kept_all_assistant_turns": 0,
                "kept_last_assistant_only": 0,
                "dropped_too_long": 0,
                "dropped_no_assistant_loss": 0,
            }
        )
    manifest_rows = tokenized_rows.select_columns(list(MANIFEST_COLUMNS))
    for start in range(0, len(manifest_rows), MANIFEST_BATCH_SIZE):
        batch = manifest_rows[start : start + MANIFEST_BATCH_SIZE]
        for file_index, last_only, n_tokens, has_loss in zip(
            batch["source_file_index"],
            batch["last_assistant_only"],
            batch["n_tokens"],
            batch["has_loss"],
            strict=True,
        ):
            entry = files[file_index]
            mode = "last_assistant_only" if last_only else "all_assistant_turns"
            entry[f"selected_{mode}"] += 1
            if n_tokens > args.max_seq_length:
                entry["dropped_too_long"] += 1
            elif not has_loss:
                entry["dropped_no_assistant_loss"] += 1
            else:
                entry["kept_count"] += 1
                entry[f"kept_{mode}"] += 1
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": options["split"],
        "seed": args.seed,
        "global_max_samples": options["max_samples"],
        "files": files,
    }


def filter_trainable_rows(tokenized_rows, max_seq_length, dataset_num_proc, split):
    filter_num_proc = dataset_num_proc if dataset_num_proc > 1 else None
    return tokenized_rows.filter(
        lambda has_loss, n_tokens: has_loss and n_tokens <= max_seq_length,
        input_columns=["has_loss", "n_tokens"],
        num_proc=filter_num_proc,
        desc=f"Filtering long/empty {split} examples",
    )


@dataclass
class AssistantLMCollator:
    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            target = list(feature["labels"])
            padding = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            labels.append(target + [self.label_pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(args, tokenizer, split="train"):
    from datasets import Dataset

    options = dataset_options(args, split)
    sampling_results, sampled_rows = sample_rows_with_thinking(args, options)

    def generator():
        for sampled in sampled_rows:
            yield {
                "raw": json.dumps(sampled["raw_row"], ensure_ascii=False),
                "enable_thinking": sampled["enable_thinking"],
                "last_assistant_only": sampled["last_assistant_only"],
                "source_file_index": sampled["source_file_index"],
                "source_record_number": sampled["source_record_number"],
            }

    raw = Dataset.from_generator(generator)
    think_open, _ = think_marker_ids(tokenizer)

    def tokenize(example):
        row = json.loads(example["raw"])
        if "messages" not in row:
            raise ValueError("数据行缺少 messages 字段")
        input_ids, assistant_masks = encode_messages(
            {"messages": row["messages"], "tools": row.get("tools")},
            tokenizer,
            bool(example["enable_thinking"]),
            bool(example["last_assistant_only"]),
            args.mask_empty_think,
        )
        labels = labels_from_assistant_masks(input_ids, assistant_masks, IGNORE_INDEX)
        first_loss = next((i for i, flag in enumerate(assistant_masks) if flag), None)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "enable_thinking": bool(example["enable_thinking"]),
            "last_assistant_only": bool(example["last_assistant_only"]),
            "trains_thinking": first_loss is not None and input_ids[first_loss] == think_open,
            "n_tokens": len(input_ids),
            "has_loss": any(label != IGNORE_INDEX for label in labels),
            "source_file_index": example["source_file_index"],
            "source_record_number": example["source_record_number"],
        }

    tokenized = raw.map(
        tokenize,
        remove_columns=raw.column_names,
        num_proc=max(args.dataset_num_proc, 1),
        desc=f"Tokenizing {split} tool-chat data",
    )
    rank0_print(f"Building {split} sampling manifest from metadata columns...")
    manifest = build_sampling_manifest(args, options, sampling_results, tokenized)
    total = len(tokenized)
    kept = filter_trainable_rows(
        tokenized,
        args.max_seq_length,
        args.dataset_num_proc,
        split,
    )
    thinking = sum(bool(value) for value in kept["trains_thinking"])
    last_assistant_only = sum(bool(value) for value in kept["last_assistant_only"])
    stats = {
        "total": total,
        "kept": len(kept),
        "thinking": thinking,
        "non_thinking": len(kept) - thinking,
        "last_assistant_only": last_assistant_only,
        "all_assistant_turns": len(kept) - last_assistant_only,
        "dropped_too_long": sum(item["dropped_too_long"] for item in manifest["files"]),
        "dropped_no_assistant": sum(
            item["dropped_no_assistant_loss"] for item in manifest["files"]
        ),
    }
    kept = kept.remove_columns(
        [
            "last_assistant_only",
            "trains_thinking",
            "n_tokens",
            "has_loss",
            "source_file_index",
            "source_record_number",
        ]
    )
    return kept, stats, manifest


def build_training_arguments(args):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=args.output_dir,
        deepspeed=args.deepspeed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_data_files is not None else "no",
        eval_steps=args.eval_steps if args.eval_data_files is not None else None,
        prediction_loss_only=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        ddp_timeout=args.ddp_timeout,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )


def build_model_and_tokenizer(args):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
    )
    if model.__class__.__name__ != EXPECTED_ARCHITECTURE:
        raise RuntimeError(f"实际加载模型类为 {model.__class__.__name__}，预期 {EXPECTED_ARCHITECTURE}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.text_config.pad_token_id = tokenizer.pad_token_id
    model.config.text_config.use_cache = False
    if args.freeze_vision_tower:
        frozen = freeze_vision_tower(model)
        rank0_print(f"已冻结视觉塔参数: {frozen:,}")
    return model, tokenizer


def preview_sample(dataset, tokenizer, split="training"):
    if len(dataset) == 0:
        return
    sample = dataset[0]
    input_ids = list(sample["input_ids"])
    labels = list(sample["labels"])
    if len(input_ids) != len(labels):
        raise ValueError(
            "预览样本的 input_ids/labels 长度不一致: "
            f"{len(input_ids)} != {len(labels)}"
        )
    loss_mask = [int(label != IGNORE_INDEX) for label in labels]
    rendered = tokenizer.decode(input_ids)
    rank0_print(
        f"\n===== first {split} sample: full rendered chat template =====\n"
        f"{rendered}\n"
        "===== loss_mask =====\n"
        f"{loss_mask}\n"
        "===== sample stats =====\n"
        f"tokens={len(input_ids)}, loss_tokens={sum(loss_mask)}\n"
        "===== end sample =====\n"
    )


def preserve_checkpoint_metadata(model_path, output_dir):
    source = Path(model_path)
    target = Path(output_dir)
    for name in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "expert_prune_metadata.json",
    ):
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, target / name)


def train(args):
    from transformers import Trainer

    args = normalize_args(args)
    runtime_preflight(args)
    training_args = build_training_arguments(args)
    model, tokenizer = build_model_and_tokenizer(args)
    model = apply_lora_if_requested(args, model)

    trainable = sum(
        (getattr(parameter, "ds_numel", None) or parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    rank0_print(f"训练模式={'LoRA' if args.use_lora else '全参数'}，可训练参数={trainable:,}")

    with training_args.main_process_first(desc="train dataset map"):
        dataset, stats, manifest = build_dataset(args, tokenizer, split="train")
    eval_dataset = None
    eval_stats = None
    eval_manifest = None
    pass_rate_examples = None
    if args.eval_data_files is not None:
        with training_args.main_process_first(desc="eval dataset map"):
            eval_dataset, eval_stats, eval_manifest = build_dataset(
                args, tokenizer, split="eval"
            )
        if args.eval_metric == "pass_rate":
            from pass_rate_eval import build_pass_rate_examples

            with training_args.main_process_first(desc="pass-rate prompt map"):
                pass_rate_examples = build_pass_rate_examples(args, tokenizer)
    if is_main_process():
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "data_sampling_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if eval_manifest is not None:
            (output_path / "eval_data_sampling_manifest.json").write_text(
                json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    rank0_print(
        f"数据总数={stats['total']}，参与训练={stats['kept']}，"
        f"过长丢弃={stats['dropped_too_long']}，无 assistant loss={stats['dropped_no_assistant']}，"
        f"thinking={stats['thinking']}，non-thinking={stats['non_thinking']}，"
        f"all-assistant-turns={stats['all_assistant_turns']}，"
        f"last-assistant-only={stats['last_assistant_only']}"
    )
    if not len(dataset):
        raise RuntimeError("过滤后没有训练样本")
    preview_sample(dataset, tokenizer, split="training")
    if eval_dataset is not None:
        rank0_print(
            f"验证数据总数={eval_stats['total']}，参与验证={eval_stats['kept']}，"
            f"过长丢弃={eval_stats['dropped_too_long']}，"
            f"无 assistant loss={eval_stats['dropped_no_assistant']}，"
            f"thinking={eval_stats['thinking']}，"
            f"non-thinking={eval_stats['non_thinking']}，"
            f"all-assistant-turns={eval_stats['all_assistant_turns']}，"
            f"last-assistant-only={eval_stats['last_assistant_only']}，"
            f"评测指标={args.eval_metric}，"
            f"每 {args.eval_steps} 个 optimizer/global steps 验证一次"
        )
        if not len(eval_dataset):
            raise RuntimeError("过滤后没有验证样本")
        preview_sample(eval_dataset, tokenizer, split="validation")

    trainer_class = Trainer
    trainer_kwargs = {}
    if args.eval_metric == "pass_rate":
        from pass_rate_eval import PassRateTrainer

        trainer_class = PassRateTrainer
        trainer_kwargs = {
            "pass_rate_examples": pass_rate_examples,
            "pass_rate_tokenizer": tokenizer,
            "pass_rate_max_new_tokens": args.eval_max_new_tokens,
        }

    callbacks = []
    if args.async_eval_markers:
        from async_eval_markers import AsyncEvalMarkerCallback

        callbacks.append(AsyncEvalMarkerCallback(metadata_source=args.model_path))

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)
        preserve_checkpoint_metadata(args.model_path, args.output_dir)
        if args.async_eval_markers:
            from async_eval_markers import (
                mark_final_model_ready,
                mark_training_complete,
            )

            final_marker = mark_final_model_ready(
                args.output_dir,
                global_step=trainer.state.global_step,
                run_name=args.run_name,
            )
            if final_marker is not None:
                rank0_print(f"[async-eval] final model ready: {final_marker}")
            complete_marker = mark_training_complete(
                args.output_dir,
                global_step=trainer.state.global_step,
                run_name=args.run_name,
            )
            rank0_print(f"[async-eval] training complete: {complete_marker}")


def main():
    args = normalize_args(build_arg_parser().parse_args())
    if args.preflight_only:
        runtime_preflight(args)
        rank0_print("[info] preflight-only 完成，未加载模型权重、未开始训练。")
        return
    train(args)


if __name__ == "__main__":
    main()
