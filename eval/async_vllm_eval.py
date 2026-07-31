"""Watch saved checkpoints and evaluate math pass rate through a vLLM server."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "train"))

from async_eval_markers import (  # noqa: E402
    CHECKPOINT_DONE_NAME,
    CHECKPOINT_FAILED_NAME,
    CHECKPOINT_READY_NAME,
    FINAL_DONE_NAME,
    FINAL_FAILED_NAME,
    FINAL_READY_NAME,
    TRAINING_COMPLETE_NAME,
    atomic_write_json,
    copy_checkpoint_metadata,
    find_huggingface_weight_files,
)
from pass_rate_eval import answers_match, extract_final_answer, normalize_answer  # noqa: E402


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
        description="Asynchronously score ready Hugging Face checkpoints with vLLM."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=None,
        help=(
            "Original model directory used to restore processor metadata that "
            "Transformers may omit from periodic checkpoints."
        ),
    )
    parser.add_argument("--gpus", default="0,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--vllm-bin",
        type=Path,
        default=Path(os.environ.get("VLLM_BIN", Path(sys.executable).parent / "vllm")),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--served-model-name", default="qwen36-async-eval")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-model-len", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=18000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking", type=parse_bool, default=True)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--server-start-timeout", type=float, default=1800.0)
    parser.add_argument("--server-stop-timeout", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-task-attempts", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N validation examples (for smoke tests).",
    )
    parser.add_argument("--retry-failed", type=parse_bool, default=False)
    parser.add_argument("--exit-when-training-complete", type=parse_bool, default=True)
    parser.add_argument(
        "--once",
        type=parse_bool,
        default=False,
        help="Evaluate currently ready tasks and exit without waiting for new checkpoints.",
    )
    return parser


def validate_args(args):
    args.run_dir = args.run_dir.resolve()
    args.dataset_file = args.dataset_file.resolve()
    if not args.dataset_file.is_file():
        raise FileNotFoundError(f"验证数据不存在: {args.dataset_file}")
    if args.metadata_source is not None:
        args.metadata_source = args.metadata_source.resolve()
        if not args.metadata_source.is_dir():
            raise FileNotFoundError(
                f"模型 metadata 目录不存在: {args.metadata_source}"
            )
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size 必须大于 0")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != args.tensor_parallel_size:
        raise ValueError(
            f"--gpus 选择了 {len(gpu_ids)} 张卡，但 "
            f"--tensor-parallel-size={args.tensor_parallel_size}"
        )
    if args.max_tokens <= 0 or args.max_tokens >= args.max_model_len:
        raise ValueError("--max-tokens 必须大于 0 且小于 --max-model-len")
    if args.concurrency <= 0:
        raise ValueError("--concurrency 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")
    if not args.vllm_bin.is_file():
        raise FileNotFoundError(f"找不到 vLLM CLI: {args.vllm_bin}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    return args


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "".join(pieces)
    return str(content or "")


def load_validation_examples(path, limit=None):
    examples = []
    with Path(path).open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number} 缺少 messages 列表")
            assistant_indices = [
                index
                for index, message in enumerate(messages)
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if not assistant_indices:
                raise ValueError(f"{path}:{line_number} 缺少 assistant 标准答案")
            answer_index = assistant_indices[-1]
            reference = content_to_text(messages[answer_index].get("content")).strip()
            prompt_messages = messages[:answer_index]
            if not reference:
                raise ValueError(f"{path}:{line_number} 的标准答案为空")
            if not prompt_messages:
                raise ValueError(f"{path}:{line_number} 的生成 prompt 为空")
            examples.append(
                {
                    "index": len(examples),
                    "source_line": line_number,
                    "messages": prompt_messages,
                    "reference": reference,
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"验证集为空: {path}")
    return examples


@dataclass(frozen=True)
class EvalTask:
    kind: str
    step: int
    model_dir: Path
    ready_marker: Path
    done_marker: Path
    failed_marker: Path

    @property
    def key(self):
        return f"{self.kind}-{self.step:08d}"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def discover_tasks(run_dir, max_attempts, retry_failed):
    run_dir = Path(run_dir)
    tasks = []
    for checkpoint_dir in run_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        ready = checkpoint_dir / CHECKPOINT_READY_NAME
        if not ready.is_file() or (checkpoint_dir / CHECKPOINT_DONE_NAME).is_file():
            continue
        failed = checkpoint_dir / CHECKPOINT_FAILED_NAME
        failure = read_json(failed, {}) or {}
        if (
            failed.is_file()
            and not retry_failed
            and int(failure.get("attempts", 0)) >= max_attempts
        ):
            continue
        tasks.append(
            EvalTask(
                kind="checkpoint",
                step=step,
                model_dir=checkpoint_dir.resolve(),
                ready_marker=ready,
                done_marker=checkpoint_dir / CHECKPOINT_DONE_NAME,
                failed_marker=failed,
            )
        )

    final_ready = run_dir / FINAL_READY_NAME
    if final_ready.is_file() and not (run_dir / FINAL_DONE_NAME).is_file():
        payload = read_json(final_ready, {}) or {}
        step = int(payload.get("global_step", -1))
        failed = run_dir / FINAL_FAILED_NAME
        failure = read_json(failed, {}) or {}
        if retry_failed or int(failure.get("attempts", 0)) < max_attempts:
            tasks.append(
                EvalTask(
                    kind="final",
                    step=step,
                    model_dir=run_dir.resolve(),
                    ready_marker=final_ready,
                    done_marker=run_dir / FINAL_DONE_NAME,
                    failed_marker=failed,
                )
            )
    return sorted(tasks, key=lambda item: (item.step, item.kind == "final"))


def validate_ready_task(task):
    payload = read_json(task.ready_marker)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ready marker 无效: {task.ready_marker}")
    if int(payload.get("global_step", -1)) != task.step:
        raise RuntimeError(f"ready marker step 不匹配: {task.ready_marker}")
    if not (task.model_dir / "config.json").is_file():
        raise RuntimeError(f"checkpoint 缺少 config.json: {task.model_dir}")
    if not find_huggingface_weight_files(task.model_dir):
        raise RuntimeError(f"checkpoint 缺少 Hugging Face 权重: {task.model_dir}")
    return payload


def restore_task_metadata(args, task):
    metadata_source = getattr(args, "metadata_source", None)
    copied = copy_checkpoint_metadata(metadata_source, task.model_dir)
    if copied:
        print(
            f"[async-eval] restored checkpoint metadata step={task.step}: "
            + ", ".join(path.name for path in copied),
            flush=True,
        )
    return copied


def port_is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match the reusable-listener behavior used by vLLM/Uvicorn. Without
        # this, recently closed client connections can make a fully stopped
        # server look as if it still owns the port until TCP cleanup finishes.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_port_free(host, port, timeout, poll_seconds=0.5):
    deadline = time.monotonic() + timeout
    while True:
        if port_is_free(host, port):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"等待 vLLM 端口释放超时 ({timeout:g}s): {host}:{port}"
            )
        time.sleep(min(poll_seconds, remaining))


def tail_text(path, max_bytes=12000):
    path = Path(path)
    if not path.is_file():
        return ""
    with path.open("rb") as reader:
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        reader.seek(max(0, size - max_bytes))
        return reader.read().decode("utf-8", errors="replace")


def wait_for_server(process, health_url, log_path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM server 提前退出，returncode={return_code}\n"
                f"{tail_text(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError(f"等待 vLLM server 超时 ({timeout}s): {health_url}")


def stop_process_group(process, timeout):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=30)


def launch_server(args, task, server_log_path):
    if not port_is_free(args.host, args.port):
        raise RuntimeError(f"vLLM 端口已被占用: {args.host}:{args.port}")
    command = [
        str(args.vllm_bin),
        "serve",
        str(task.model_dir),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--api-key",
        args.api_key,
        "--gdn-prefill-backend",
        args.gdn_prefill_backend,
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    environment["PYTHONUNBUFFERED"] = "1"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    process._async_eval_log_handle = log_handle  # Keep the descriptor alive.
    print(
        f"[async-eval] vLLM PID={process.pid}, GPUs={args.gpus}, "
        f"model={task.model_dir}",
        flush=True,
    )
    return process, command


class ApiGenerator:
    def __init__(self, args):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key=args.api_key,
            timeout=args.request_timeout,
            max_retries=0,
        )
        self.model = args.served_model_name
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.thinking = args.thinking
        self.request_retries = args.request_retries

    def generate(self, messages):
        last_error = None
        for attempt in range(self.request_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.thinking,
                        }
                    },
                )
                message = response.choices[0].message
                content = message.content or ""
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                    or ""
                )
                raw_prediction = content if content.strip() else reasoning
                usage = response.usage
                return {
                    "raw_prediction": raw_prediction,
                    "reasoning": reasoning,
                    "content": content,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "finish_reason": response.choices[0].finish_reason,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < self.request_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(
            f"请求在 {self.request_retries} 次尝试后失败: {last_error}"
        )


def atomic_write_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False))
            writer.write("\n")
    os.replace(temporary, path)


def evaluate_examples(args, task, examples, task_output_dir):
    generator = ApiGenerator(args)
    records_by_index = {}
    errors = 0
    passed = 0
    progress = tqdm(
        total=len(examples),
        desc=f"vLLM验证(step={task.step})",
        unit="题",
        dynamic_ncols=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            future_to_example = {
                pool.submit(generator.generate, example["messages"]): example
                for example in examples
            }
            for future in as_completed(future_to_example):
                example = future_to_example[future]
                try:
                    generated = future.result()
                    error = None
                except Exception as exc:  # noqa: BLE001
                    generated = {
                        "raw_prediction": "",
                        "reasoning": "",
                        "content": "",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "finish_reason": None,
                    }
                    error = f"{type(exc).__name__}: {exc}"
                    errors += 1
                raw_prediction = generated["raw_prediction"]
                is_passed = bool(
                    error is None
                    and answers_match(raw_prediction, example["reference"])
                )
                passed += int(is_passed)
                records_by_index[example["index"]] = {
                    "index": example["index"],
                    "source_line": example["source_line"],
                    "messages": example["messages"],
                    "reference": example["reference"],
                    "prediction": raw_prediction,
                    "reasoning": generated["reasoning"],
                    "content": generated["content"],
                    "extracted_prediction": extract_final_answer(raw_prediction),
                    "normalized_reference": normalize_answer(example["reference"]),
                    "normalized_prediction": normalize_answer(raw_prediction),
                    "passed": is_passed,
                    "error": error,
                    "prompt_tokens": generated["prompt_tokens"],
                    "completion_tokens": generated["completion_tokens"],
                    "finish_reason": generated["finish_reason"],
                }
                progress.update(1)
                completed = len(records_by_index)
                progress.set_postfix(
                    pass_rate=f"{passed / completed:.4f}",
                    errors=errors,
                )
    finally:
        progress.close()

    records = [records_by_index[index] for index in sorted(records_by_index)]
    atomic_write_jsonl(task_output_dir / "predictions.jsonl", records)
    return records, passed, errors


def rebuild_aggregate_results(run_dir):
    async_dir = Path(run_dir) / "async_eval"
    metrics = []
    for path in async_dir.glob("*/metrics.json"):
        payload = read_json(path)
        if isinstance(payload, dict):
            metrics.append(payload)
    metrics.sort(key=lambda item: (int(item["global_step"]), item["kind"] == "final"))
    atomic_write_jsonl(async_dir / "results.jsonl", metrics)


def evaluate_task(args, task, examples):
    restore_task_metadata(args, task)
    ready_payload = validate_ready_task(task)
    task_output_dir = args.run_dir / "async_eval" / task.key
    task_output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = task_output_dir / "vllm_server.log"
    started_at = utc_now()
    started_monotonic = time.monotonic()
    process = None
    command = None
    try:
        process, command = launch_server(args, task, server_log_path)
        wait_for_server(
            process,
            health_url=f"http://{args.host}:{args.port}/health",
            log_path=server_log_path,
            timeout=args.server_start_timeout,
        )
        print(f"[async-eval] vLLM ready: step={task.step}", flush=True)
        records, passed, errors = evaluate_examples(
            args,
            task,
            examples,
            task_output_dir,
        )
        if errors:
            raise RuntimeError(f"{errors}/{len(records)} 个验证请求失败")
        runtime = time.monotonic() - started_monotonic
        total = len(records)
        metrics = {
            "schema_version": 1,
            "kind": task.kind,
            "global_step": task.step,
            "checkpoint_path": str(task.model_dir),
            "dataset_file": str(args.dataset_file),
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
            "runtime_seconds": runtime,
            "started_at": started_at,
            "finished_at": utc_now(),
            "gpus": args.gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "ready_marker": ready_payload,
            "server_command": command,
            "predictions_path": str(task_output_dir / "predictions.jsonl"),
            "server_log_path": str(server_log_path),
        }
        atomic_write_json(task_output_dir / "metrics.json", metrics)
        atomic_write_json(task.done_marker, metrics)
        if task.failed_marker.exists():
            task.failed_marker.unlink()
        rebuild_aggregate_results(args.run_dir)
        print(
            f"[async-eval] completed step={task.step}: "
            f"{passed}/{total}, pass_rate={metrics['pass_rate']:.4f}, "
            f"runtime={runtime:.1f}s",
            flush=True,
        )
        return metrics
    finally:
        if process is not None:
            try:
                stop_process_group(process, args.server_stop_timeout)
                wait_for_port_free(
                    args.host,
                    args.port,
                    timeout=args.server_stop_timeout,
                )
            finally:
                log_handle = getattr(process, "_async_eval_log_handle", None)
                if log_handle is not None:
                    log_handle.close()


def record_task_failure(task, error, max_attempts):
    previous = read_json(task.failed_marker, {}) or {}
    attempts = int(previous.get("attempts", 0)) + 1
    payload = {
        "schema_version": 1,
        "kind": task.kind,
        "global_step": task.step,
        "checkpoint_path": str(task.model_dir),
        "attempts": attempts,
        "max_attempts": max_attempts,
        "failed_at": utc_now(),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    atomic_write_json(task.failed_marker, payload)
    print(
        f"[async-eval][error] step={task.step}, attempt={attempts}/"
        f"{max_attempts}: {payload['error']}",
        flush=True,
    )
    return attempts


def acquire_worker_lock(run_dir):
    lock_path = Path(run_dir) / "async_eval" / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"该 run 已有异步验证 worker: {lock_path}") from exc
    handle.write(f"pid={os.getpid()}\nstarted_at={utc_now()}\n")
    handle.flush()
    return handle


def reset_failed_markers(run_dir):
    reset = []
    for checkpoint_dir in Path(run_dir).glob("checkpoint-*"):
        marker = checkpoint_dir / CHECKPOINT_FAILED_NAME
        if marker.is_file():
            marker.unlink()
            reset.append(marker)
    final_marker = Path(run_dir) / FINAL_FAILED_NAME
    if final_marker.is_file():
        final_marker.unlink()
        reset.append(final_marker)
    for marker in reset:
        print(f"[async-eval] retry reset: {marker}", flush=True)


def has_permanent_failures(run_dir, max_attempts):
    markers = list(Path(run_dir).glob(f"checkpoint-*/{CHECKPOINT_FAILED_NAME}"))
    markers.append(Path(run_dir) / FINAL_FAILED_NAME)
    return any(
        marker.is_file()
        and int((read_json(marker, {}) or {}).get("attempts", 0)) >= max_attempts
        for marker in markers
    )


def run_worker(args):
    examples = load_validation_examples(args.dataset_file, limit=args.limit)
    print(
        f"[async-eval] run_dir={args.run_dir}, dataset={args.dataset_file}, "
        f"examples={len(examples)}, GPUs={args.gpus}",
        flush=True,
    )
    lock_handle = acquire_worker_lock(args.run_dir)
    if args.retry_failed:
        reset_failed_markers(args.run_dir)
    had_permanent_failure = False
    try:
        while True:
            tasks = discover_tasks(
                args.run_dir,
                max_attempts=args.max_task_attempts,
                retry_failed=False,
            )
            if tasks:
                task = tasks[0]
                try:
                    evaluate_task(args, task, examples)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    attempts = record_task_failure(
                        task,
                        exc,
                        max_attempts=args.max_task_attempts,
                    )
                    had_permanent_failure |= attempts >= args.max_task_attempts
                continue

            if args.once:
                break
            training_complete = (args.run_dir / TRAINING_COMPLETE_NAME).is_file()
            if args.exit_when_training_complete and training_complete:
                all_ready = discover_tasks(
                    args.run_dir,
                    max_attempts=10**9,
                    retry_failed=True,
                )
                unfinished = [
                    task
                    for task in all_ready
                    if not task.done_marker.is_file()
                    and not (
                        task.failed_marker.is_file()
                        and int(
                            (read_json(task.failed_marker, {}) or {}).get(
                                "attempts", 0
                            )
                        )
                        >= args.max_task_attempts
                    )
                ]
                if not unfinished:
                    break
            print(
                f"[async-eval] 暂无 ready checkpoint；"
                f"{args.poll_seconds:g}s 后重试",
                flush=True,
            )
            time.sleep(args.poll_seconds)
    finally:
        lock_handle.close()
    had_permanent_failure |= has_permanent_failures(
        args.run_dir,
        max_attempts=args.max_task_attempts,
    )
    return 1 if had_permanent_failure else 0


def main():
    args = validate_args(build_arg_parser().parse_args())
    raise SystemExit(run_worker(args))


if __name__ == "__main__":
    main()
