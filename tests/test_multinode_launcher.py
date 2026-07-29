import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PACKAGE_ROOT / "train" / "run_qwen36_19b_a3b_sft_deepspeed.sh"


class MultiNodeLauncherTests(unittest.TestCase):
    def run_launcher(self, extra_env=None):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            fake_python = temporary_path / "python"
            fake_torchrun = temporary_path / "torchrun"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_torchrun.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'TORCHRUN_ARG=%s\\n' \"$@\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_torchrun.chmod(0o755)

            env = os.environ.copy()
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "GPUS",
                "MASTER_ADDR",
                "MASTER_PORT",
                "NNODES",
                "NODE_RANK",
                "NPROC_PER_NODE",
            ):
                env.pop(name, None)
            env.update(
                {
                    "PYTHON_BIN": str(fake_python),
                    "TORCHRUN_BIN": str(fake_torchrun),
                    "RUN_NAME": "launcher-unit-test",
                    "OUTPUT_DIR": str(temporary_path / "output"),
                    "SKIP_GPU_CHECK": "1",
                }
            )
            if extra_env:
                env.update(extra_env)

            return subprocess.run(
                [
                    "bash",
                    str(LAUNCHER),
                    "--gpus",
                    "0,1",
                    "--data-files",
                    "/datasets/train.jsonl",
                ],
                cwd=PACKAGE_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    @staticmethod
    def torchrun_args(result):
        prefix = "TORCHRUN_ARG="
        return [
            line[len(prefix) :]
            for line in result.stdout.splitlines()
            if line.startswith(prefix)
        ]

    def test_single_node_keeps_standalone_launch(self):
        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.torchrun_args(result)
        self.assertIn("--standalone", args)
        self.assertEqual(args[args.index("--nnodes") + 1], "1")
        self.assertEqual(args[args.index("--nproc_per_node") + 1], "2")
        self.assertNotIn("--node_rank", args)
        self.assertNotIn("--master_addr", args)
        self.assertIn("WORLD_SIZE=2", result.stdout)
        self.assertIn("MASTER_ADDR=standalone", result.stdout)

    def test_multi_node_uses_static_rendezvous(self):
        result = self.run_launcher(
            {
                "NNODES": "3",
                "NODE_RANK": "1",
                "MASTER_ADDR": "10.20.30.40",
                "MASTER_PORT": "29600",
            }
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.torchrun_args(result)
        self.assertNotIn("--standalone", args)
        self.assertEqual(args[args.index("--nnodes") + 1], "3")
        self.assertEqual(args[args.index("--nproc_per_node") + 1], "2")
        self.assertEqual(args[args.index("--node_rank") + 1], "1")
        self.assertEqual(args[args.index("--master_addr") + 1], "10.20.30.40")
        self.assertEqual(args[args.index("--master_port") + 1], "29600")
        self.assertIn("WORLD_SIZE=6", result.stdout)

    def test_multi_node_requires_master_address(self):
        result = self.run_launcher({"NNODES": "2", "NODE_RANK": "0"})

        self.assertEqual(result.returncode, 2)
        self.assertIn("多机模式要求设置 MASTER_ADDR", result.stderr)
        self.assertEqual(self.torchrun_args(result), [])

    def test_rejects_node_rank_outside_world(self):
        result = self.run_launcher(
            {
                "NNODES": "2",
                "NODE_RANK": "2",
                "MASTER_ADDR": "10.20.30.40",
            }
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("NODE_RANK 必须是 [0, NNODES)", result.stderr)
        self.assertEqual(self.torchrun_args(result), [])

    def test_rejects_invalid_master_port(self):
        result = self.run_launcher(
            {
                "NNODES": "2",
                "NODE_RANK": "0",
                "MASTER_ADDR": "10.20.30.40",
                "MASTER_PORT": "70000",
            }
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("MASTER_PORT 必须是 1-65535", result.stderr)
        self.assertEqual(self.torchrun_args(result), [])


if __name__ == "__main__":
    unittest.main()
