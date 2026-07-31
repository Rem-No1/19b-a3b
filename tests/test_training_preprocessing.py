import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PACKAGE_ROOT / "train" / "train_qwen36_19b_a3b_sft_deepspeed.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("qwen36_training", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = load_training_module()


class MetadataDataset:
    def __init__(self, rows):
        self.rows = rows
        self.selected_columns = None

    def select_columns(self, columns):
        self.selected_columns = tuple(columns)
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise AssertionError("manifest should read metadata in batches")
        selected = self.rows[item]
        return {
            column: [row[column] for row in selected]
            for column in self.selected_columns
        }

    def __iter__(self):
        raise AssertionError("manifest should not materialize full dataset rows")


class TrainingPreprocessingTests(unittest.TestCase):
    def test_filter_reads_only_loss_and_length_columns(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets is only installed in the delivery image")

        rows = Dataset.from_dict(
            {
                "input_ids": [[1, 2], [3, 4], [5, 6]],
                "labels": [[1, 2], [3, 4], [-100, -100]],
                "has_loss": [True, True, False],
                "n_tokens": [8, 11, 5],
            }
        )

        kept = training.filter_trainable_rows(
            rows,
            max_seq_length=10,
            dataset_num_proc=1,
            split="train",
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["input_ids"], [1, 2])

    def test_manifest_reads_only_small_metadata_columns_in_batches(self):
        rows = MetadataDataset(
            [
                {
                    "source_file_index": 0,
                    "last_assistant_only": False,
                    "n_tokens": 8,
                    "has_loss": True,
                },
                {
                    "source_file_index": 0,
                    "last_assistant_only": True,
                    "n_tokens": 11,
                    "has_loss": True,
                },
                {
                    "source_file_index": 1,
                    "last_assistant_only": True,
                    "n_tokens": 5,
                    "has_loss": False,
                },
                {
                    "source_file_index": 1,
                    "last_assistant_only": False,
                    "n_tokens": 9,
                    "has_loss": True,
                },
            ]
        )
        sampling_results = [
            SimpleNamespace(
                source_path=f"/data/file-{index}.jsonl",
                source_size_bytes=100,
                source_mtime_ns=123,
                selection_mode="all",
                configured_limit=None,
                source_record_count=2,
                records=(object(), object()),
            )
            for index in range(2)
        ]

        manifest = training.build_sampling_manifest(
            SimpleNamespace(max_seq_length=10, seed=3407),
            {"split": "train", "max_samples": None},
            sampling_results,
            rows,
        )

        self.assertEqual(rows.selected_columns, training.MANIFEST_COLUMNS)
        self.assertEqual(manifest["files"][0]["kept_count"], 1)
        self.assertEqual(manifest["files"][0]["dropped_too_long"], 1)
        self.assertEqual(manifest["files"][1]["kept_count"], 1)
        self.assertEqual(manifest["files"][1]["dropped_no_assistant_loss"], 1)

    def test_ddp_timeout_is_forwarded_to_training_arguments(self):
        args = training.normalize_args(
            training.build_arg_parser().parse_args(
                [
                    "--data-files",
                    "/data/train.jsonl",
                    "--ddp-timeout",
                    "7200",
                ]
            )
        )
        fake_transformers = ModuleType("transformers")

        class FakeTrainingArguments:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_transformers.TrainingArguments = FakeTrainingArguments
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            training_args = training.build_training_arguments(args)

        self.assertEqual(training_args.kwargs["ddp_timeout"], 7200)

    def test_ddp_timeout_must_be_positive(self):
        args = training.build_arg_parser().parse_args(
            ["--data-files", "/data/train.jsonl", "--ddp-timeout", "0"]
        )

        with self.assertRaisesRegex(ValueError, "--ddp-timeout 必须大于 0"):
            training.normalize_args(args)


if __name__ == "__main__":
    unittest.main()
