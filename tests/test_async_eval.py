import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "train"))

from async_eval_markers import (  # noqa: E402
    AsyncEvalMarkerCallback,
    CHECKPOINT_DONE_NAME,
    CHECKPOINT_READY_NAME,
    FINAL_READY_NAME,
    TRAINING_COMPLETE_NAME,
    copy_checkpoint_metadata,
    mark_checkpoint_ready,
    mark_final_model_ready,
    mark_training_complete,
)


def load_worker_module():
    path = PACKAGE_ROOT / "eval" / "async_vllm_eval.py"
    spec = importlib.util.spec_from_file_location("async_vllm_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = load_worker_module()


def create_fake_hf_model(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fake-weights")


class AsyncEvalMarkerTests(unittest.TestCase):
    def test_checkpoint_metadata_is_copied_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model"
            checkpoint = root / "run" / "checkpoint-10"
            source.mkdir()
            checkpoint.mkdir(parents=True)
            (source / "preprocessor_config.json").write_text(
                '{"image_processor_type": "Qwen3VLImageProcessor"}\n',
                encoding="utf-8",
            )
            (source / "video_preprocessor_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )

            copied = copy_checkpoint_metadata(source, checkpoint)

            self.assertEqual(
                [path.name for path in copied],
                [
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                ],
            )
            self.assertEqual(
                (checkpoint / "preprocessor_config.json").read_text(
                    encoding="utf-8"
                ),
                '{"image_processor_type": "Qwen3VLImageProcessor"}\n',
            )

    def test_checkpoint_marker_is_atomic_and_discoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "checkpoint-10"
            create_fake_hf_model(checkpoint)

            marker = mark_checkpoint_ready(run_dir, 10, "unit-test")
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["global_step"], 10)
            self.assertEqual(payload["kind"], "checkpoint")
            self.assertEqual(payload["weight_files"][0]["size_bytes"], 12)
            self.assertFalse(any(checkpoint.glob("*.tmp-*")))

            tasks = worker.discover_tasks(run_dir, max_attempts=3, retry_failed=False)
            self.assertEqual([(task.kind, task.step) for task in tasks], [("checkpoint", 10)])

            (checkpoint / CHECKPOINT_DONE_NAME).write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                worker.discover_tasks(run_dir, max_attempts=3, retry_failed=False),
                [],
            )

    def test_final_and_training_complete_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            create_fake_hf_model(run_dir)
            final_marker = mark_final_model_ready(run_dir, 12, "unit-test")
            complete_marker = mark_training_complete(run_dir, 12, "unit-test")
            self.assertEqual(final_marker.name, FINAL_READY_NAME)
            self.assertEqual(complete_marker.name, TRAINING_COMPLETE_NAME)

    def test_final_marker_skipped_when_same_step_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            create_fake_hf_model(run_dir)
            create_fake_hf_model(run_dir / "checkpoint-20")
            mark_checkpoint_ready(run_dir, 20, "unit-test")
            self.assertIsNone(mark_final_model_ready(run_dir, 20, "unit-test"))

    def test_callback_only_publishes_on_world_process_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            create_fake_hf_model(run_dir / "checkpoint-30")
            metadata_source = run_dir / "model"
            metadata_source.mkdir()
            (metadata_source / "preprocessor_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            callback = AsyncEvalMarkerCallback(metadata_source=metadata_source)
            args = SimpleNamespace(output_dir=str(run_dir), run_name="unit-test")
            control = object()

            callback.on_save(
                args,
                SimpleNamespace(global_step=30, is_world_process_zero=False),
                control,
            )
            self.assertFalse(
                (run_dir / "checkpoint-30" / CHECKPOINT_READY_NAME).exists()
            )

            returned = callback.on_save(
                args,
                SimpleNamespace(global_step=30, is_world_process_zero=True),
                control,
            )
            self.assertIs(returned, control)
            self.assertTrue(
                (run_dir / "checkpoint-30" / CHECKPOINT_READY_NAME).is_file()
            )
            self.assertTrue(
                (run_dir / "checkpoint-30" / "preprocessor_config.json").is_file()
            )


class AsyncEvalWorkerTests(unittest.TestCase):
    def test_worker_restores_processor_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model"
            checkpoint = root / "run" / "checkpoint-10"
            source.mkdir()
            checkpoint.mkdir(parents=True)
            (source / "preprocessor_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            task = SimpleNamespace(step=10, model_dir=checkpoint)

            copied = worker.restore_task_metadata(
                SimpleNamespace(metadata_source=source),
                task,
            )

            self.assertEqual(
                [path.name for path in copied],
                ["preprocessor_config.json"],
            )
            self.assertTrue(
                (checkpoint / "preprocessor_config.json").is_file()
            )

    def test_validation_loader_removes_reference_assistant(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "val.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "1+1=?"},
                            {"role": "assistant", "content": "2"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            examples = worker.load_validation_examples(dataset)
            self.assertEqual(examples[0]["reference"], "2")
            self.assertEqual(
                examples[0]["messages"],
                [{"role": "user", "content": "1+1=?"}],
            )

    def test_worker_lock_rejects_duplicate_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = worker.acquire_worker_lock(temporary)
            try:
                with self.assertRaisesRegex(RuntimeError, "已有异步验证 worker"):
                    worker.acquire_worker_lock(temporary)
            finally:
                first.close()

    def test_evaluate_examples_writes_ordered_predictions(self):
        class FakeGenerator:
            def __init__(self, args):
                del args

            def generate(self, messages):
                answer = "2" if "1+1" in messages[0]["content"] else "4"
                return {
                    "raw_prediction": answer,
                    "reasoning": "",
                    "content": answer,
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "finish_reason": "stop",
                }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            task = SimpleNamespace(step=10)
            examples = [
                {
                    "index": 1,
                    "source_line": 2,
                    "messages": [{"role": "user", "content": "2+2=?"}],
                    "reference": "4",
                },
                {
                    "index": 0,
                    "source_line": 1,
                    "messages": [{"role": "user", "content": "1+1=?"}],
                    "reference": "2",
                },
            ]
            with mock.patch.object(worker, "ApiGenerator", FakeGenerator):
                records, passed, errors = worker.evaluate_examples(
                    SimpleNamespace(concurrency=2),
                    task,
                    examples,
                    output_dir,
                )
            self.assertEqual([record["index"] for record in records], [0, 1])
            self.assertEqual(passed, 2)
            self.assertEqual(errors, 0)
            lines = (output_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual([json.loads(line)["index"] for line in lines], [0, 1])

    def test_full_task_lifecycle_with_fake_openai_server(self):
        fake_server_source = """\
#!{python}
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        payload = {{
            "id": "fake",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen36-async-eval",
            "choices": [{{
                "index": 0,
                "message": {{"role": "assistant", "content": "2"}},
                "finish_reason": "stop"
            }}],
            "usage": {{
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6
            }}
        }}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
""".format(python=sys.executable)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            checkpoint = run_dir / "checkpoint-10"
            create_fake_hf_model(checkpoint)
            ready_marker = mark_checkpoint_ready(run_dir, 10, "unit-test")
            fake_vllm = Path(temporary) / "fake-vllm"
            fake_vllm.write_text(fake_server_source, encoding="utf-8")
            fake_vllm.chmod(0o755)
            task = worker.EvalTask(
                kind="checkpoint",
                step=10,
                model_dir=checkpoint,
                ready_marker=ready_marker,
                done_marker=checkpoint / worker.CHECKPOINT_DONE_NAME,
                failed_marker=checkpoint / worker.CHECKPOINT_FAILED_NAME,
            )
            with socket_port() as port:
                args = SimpleNamespace(
                    run_dir=run_dir,
                    dataset_file=Path(temporary) / "val.jsonl",
                    gpus="0,7",
                    tensor_parallel_size=2,
                    vllm_bin=fake_vllm,
                    host="127.0.0.1",
                    port=port,
                    served_model_name="qwen36-async-eval",
                    api_key="EMPTY",
                    max_model_len=24000,
                    max_tokens=32,
                    gpu_memory_utilization=0.9,
                    dtype="bfloat16",
                    gdn_prefill_backend="triton",
                    concurrency=2,
                    temperature=0.0,
                    top_p=1.0,
                    thinking=False,
                    request_timeout=10.0,
                    request_retries=1,
                    server_start_timeout=10.0,
                    server_stop_timeout=5.0,
                )
                metrics = worker.evaluate_task(
                    args,
                    task,
                    [
                        {
                            "index": 0,
                            "source_line": 1,
                            "messages": [{"role": "user", "content": "1+1=?"}],
                            "reference": "2",
                        }
                    ],
                )
            self.assertEqual(metrics["pass_rate"], 1.0)
            self.assertTrue(task.done_marker.is_file())
            self.assertTrue(
                (run_dir / "async_eval" / "results.jsonl").is_file()
            )


class socket_port:
    def __enter__(self):
        import socket

        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.socket.close()
        return self.port

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback


if __name__ == "__main__":
    unittest.main()
