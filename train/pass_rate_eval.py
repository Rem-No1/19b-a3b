"""Generation-based pass-rate evaluation for final-answer math datasets."""

import json
import math
import re
import time
import unicodedata
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer
from tqdm.auto import tqdm

from toolchat_data import (
    collect_data_files,
    normalize_tool_call_arguments,
    sample_data_files,
)


def _extract_braced(text, open_brace):
    if open_brace >= len(text) or text[open_brace] != "{":
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index], index + 1
    return None


def _last_boxed(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(matches):
        parsed = _extract_braced(text, match.end() - 1)
        if parsed is not None:
            return parsed[0]
    return None


def extract_final_answer(text):
    text = str(text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        text = lines[-1]
    text = re.sub(
        r"^(?:the\s+final\s+answer\s+is|final\s+answer|answer|the\s+answer\s+is)"
        r"\s*[:：=]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _replace_text_commands(text):
    pattern = re.compile(r"\\(?:text|mathrm|textrm|operatorname)\s*\{([^{}]*)\}")
    while pattern.search(text):
        text = pattern.sub(r"\1", text)
    return text


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", extract_final_answer(text))
    text = _replace_text_commands(text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\displaystyle", "")
    text = text.replace("\\%", "%").replace("π", "\\pi")
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\\[,!;:]\s*", "", text)
    text = text.replace("$", "").replace("`", "")
    text = text.strip().lower()
    text = re.sub(r"^(?:finalanswer|answer|theansweris)[:：=]?", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.rstrip(".。")
    aliases = {"one": "1", "zero": "0"}
    return aliases.get(text, text)


def _replace_latex_fractions(text):
    for command in ("\\dfrac", "\\tfrac", "\\frac"):
        while command in text:
            start = text.find(command)
            numerator_start = start + len(command)
            while numerator_start < len(text) and text[numerator_start].isspace():
                numerator_start += 1
            numerator = _extract_braced(text, numerator_start)
            if numerator is None:
                break
            denominator_start = numerator[1]
            while denominator_start < len(text) and text[denominator_start].isspace():
                denominator_start += 1
            denominator = _extract_braced(text, denominator_start)
            if denominator is None:
                break
            replacement = (
                f"(({_replace_latex_fractions(numerator[0])})/"
                f"({_replace_latex_fractions(denominator[0])}))"
            )
            text = text[:start] + replacement + text[denominator[1] :]
    return text


def _replace_latex_sqrt(text):
    command = "\\sqrt"
    while command in text:
        start = text.find(command)
        argument_start = start + len(command)
        while argument_start < len(text) and text[argument_start].isspace():
            argument_start += 1
        argument = _extract_braced(text, argument_start)
        if argument is None:
            break
        replacement = f"sqrt({_replace_latex_sqrt(argument[0])})"
        text = text[:start] + replacement + text[argument[1] :]
    return text


def _to_symbolic_expression(text):
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    expression = normalize_answer(text)
    expression = _replace_latex_fractions(expression)
    expression = _replace_latex_sqrt(expression)
    expression = expression.replace("\\pi", "pi")
    expression = expression.replace("{", "(").replace("}", ")")
    expression = expression.replace(":", "/")
    if expression.endswith("%"):
        expression = f"({expression[:-1]})/100"
    if not expression or len(expression) > 512:
        return None
    if not re.fullmatch(r"[0-9a-z+\-*/^().,=%]+", expression):
        return None
    if re.search(r"(?<!\d)\.|\.(?!\d)", expression):
        return None
    identifiers = set(re.findall(r"[a-z]+", expression))
    if any(len(name) > 1 and name not in {"pi", "sqrt"} for name in identifiers):
        return None

    local_dict = {"pi": sympy.pi, "sqrt": sympy.sqrt}
    for name in identifiers:
        if len(name) == 1:
            local_dict[name] = sympy.Symbol(name)
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    def parse_one(value):
        return parse_expr(
            value,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )

    if expression.count("=") == 1:
        left, right = expression.split("=", 1)
        return sympy.simplify(parse_one(left) - parse_one(right))
    return parse_one(expression)


def answers_match(prediction, reference):
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if normalized_prediction == normalized_reference:
        return True
    try:
        prediction_expression = _to_symbolic_expression(prediction)
        reference_expression = _to_symbolic_expression(reference)
        if prediction_expression is None or reference_expression is None:
            return False
        if isinstance(prediction_expression, tuple) or isinstance(reference_expression, tuple):
            if not (
                isinstance(prediction_expression, tuple)
                and isinstance(reference_expression, tuple)
                and len(prediction_expression) == len(reference_expression)
            ):
                return False
            return all(
                bool((left - right).simplify() == 0)
                for left, right in zip(prediction_expression, reference_expression)
            )
        return bool((prediction_expression - reference_expression).simplify() == 0)
    except (ArithmeticError, SyntaxError, TypeError, ValueError):
        return False


def build_pass_rate_examples(args, tokenizer):
    data_files = collect_data_files(args.eval_data_files)
    sampling_results = sample_data_files(
        data_files, args.max_eval_samples_per_file, args.seed
    )
    sampled = [
        record
        for result in sampling_results
        for record in result.records
    ]
    if args.max_eval_samples is not None:
        sampled = sampled[: args.max_eval_samples]

    examples = []
    for index, record in enumerate(sampled):
        row = record.row
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                f"{record.source_path}:{record.source_record_number}: "
                "pass_rate 验证要求 messages 至少包含 user 和 assistant"
            )
        answer = messages[-1]
        if answer.get("role") != "assistant" or not isinstance(answer.get("content"), str):
            raise ValueError(
                f"{record.source_path}:{record.source_record_number}: "
                "pass_rate 验证要求最后一条 message 是含文本答案的 assistant"
            )
        prompt_messages = normalize_tool_call_arguments(messages[:-1])
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tools=row.get("tools") or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.eval_generation_enable_thinking,
        )
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        if len(prompt_ids) + args.eval_max_new_tokens > args.max_seq_length:
            raise ValueError(
                f"{record.source_path}:{record.source_record_number}: prompt "
                f"{len(prompt_ids)} + eval tokens {args.eval_max_new_tokens} "
                f"超过 max_seq_length={args.max_seq_length}"
            )
        examples.append(
            {
                "index": index,
                "source_file": record.source_path,
                "source_record_number": record.source_record_number,
                "question": messages[-2].get("content", ""),
                "reference": answer["content"],
                "prompt_ids": list(prompt_ids),
            }
        )
    if not examples:
        raise RuntimeError("pass_rate 验证集为空")
    return examples


class PassRateTrainer(Trainer):
    def __init__(
        self,
        *args,
        pass_rate_examples,
        pass_rate_tokenizer,
        pass_rate_max_new_tokens,
        **kwargs,
    ):
        self.pass_rate_examples = pass_rate_examples
        self.pass_rate_tokenizer = pass_rate_tokenizer
        self.pass_rate_max_new_tokens = int(pass_rate_max_new_tokens)
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        del eval_dataset, ignore_keys
        self._memory_tracker.start()
        start_time = time.time()
        self.accelerator.wait_for_everyone()

        wrapped_model = self.model_wrapped
        model = self.accelerator.unwrap_model(wrapped_model)
        was_training = model.training
        model.eval()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        rounds = math.ceil(len(self.pass_rate_examples) / world_size)
        local_records = []
        progress_bar = (
            tqdm(
                total=len(self.pass_rate_examples),
                desc=f"验证生成(step={self.state.global_step})",
                unit="题",
                dynamic_ncols=True,
                leave=True,
            )
            if rank == 0
            else None
        )

        try:
            for round_index in range(rounds):
                example_index = round_index * world_size + rank
                is_real_example = example_index < len(self.pass_rate_examples)
                example = self.pass_rate_examples[example_index if is_real_example else 0]
                input_ids = torch.tensor(
                    [example["prompt_ids"]],
                    dtype=torch.long,
                    device=self.args.device,
                )
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.pass_rate_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        synced_gpus=world_size > 1,
                        pad_token_id=self.pass_rate_tokenizer.pad_token_id,
                        eos_token_id=self.pass_rate_tokenizer.eos_token_id,
                    )
                if progress_bar is not None:
                    completed_before_round = round_index * world_size
                    completed_this_round = min(
                        world_size,
                        len(self.pass_rate_examples) - completed_before_round,
                    )
                    progress_bar.update(completed_this_round)
                if not is_real_example:
                    continue
                prediction = self.pass_rate_tokenizer.decode(
                    output_ids[0, input_ids.shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                passed = answers_match(prediction, example["reference"])
                local_records.append(
                    {
                        "index": example["index"],
                        "source_file": example["source_file"],
                        "source_record_number": example["source_record_number"],
                        "question": example["question"],
                        "reference": example["reference"],
                        "prediction": prediction,
                        "normalized_reference": normalize_answer(example["reference"]),
                        "normalized_prediction": normalize_answer(prediction),
                        "passed": passed,
                    }
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
            model.train(was_training)

        if dist.is_initialized():
            gathered_records = [None] * world_size
            dist.all_gather_object(gathered_records, local_records)
            records = [
                record
                for rank_records in gathered_records
                for record in rank_records
            ]
            elapsed = torch.tensor(
                time.time() - start_time,
                dtype=torch.float64,
                device=self.args.device,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            runtime = elapsed.item()
        else:
            records = local_records
            runtime = time.time() - start_time

        records.sort(key=lambda record: record["index"])
        passed = sum(bool(record["passed"]) for record in records)
        total = len(records)
        metrics = {
            f"{metric_key_prefix}_pass_rate": passed / total if total else 0.0,
            f"{metric_key_prefix}_passed": float(passed),
            f"{metric_key_prefix}_total": float(total),
            f"{metric_key_prefix}_runtime": runtime,
            f"{metric_key_prefix}_samples_per_second": total / runtime if runtime else 0.0,
        }

        if self.is_world_process_zero():
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / (
                f"pass_rate_predictions_step_{self.state.global_step:08d}.jsonl"
            )
            result_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            print(
                f"[pass_rate] step={self.state.global_step} "
                f"passed={passed}/{total} pass_rate={metrics[f'{metric_key_prefix}_pass_rate']:.4f} "
                f"details={result_path}"
            )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.accelerator.wait_for_everyone()
        return metrics
