"""Qwen ChatML tool-chat data and assistant-only loss-mask utilities."""

import copy
import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampledDataRow:
    row: dict
    source_path: str
    source_record_number: int


@dataclass(frozen=True)
class FileSamplingResult:
    source_path: str
    source_size_bytes: int
    source_mtime_ns: int
    configured_limit: int | None
    source_record_count: int
    records: tuple[SampledDataRow, ...]

    @property
    def selection_mode(self):
        return "all" if self.configured_limit is None else "random_subset"


def derive_file_sampling_seed(global_seed, source_path):
    normalized_path = str(Path(source_path).resolve())
    material = f"{int(global_seed)}\0{normalized_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def collect_data_files(paths):
    return [str(Path(path)) for path in paths if Path(path).exists()]


def iter_jsonl(handle):
    for line in handle:
        line = line.strip()
        if line:
            yield json.loads(line)


def iter_json_array(handle, chunk_size=1 << 20):
    """Stream rows from a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        buffer += chunk
        while True:
            buffer = buffer.lstrip()
            if not started:
                if buffer.startswith("["):
                    buffer = buffer[1:]
                    started = True
                    continue
                break
            if not buffer:
                break
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                return
            try:
                obj, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield obj
            buffer = buffer[index:]


def normalize_max_samples_per_file(data_files, max_samples_per_file=None):
    if max_samples_per_file is None:
        return [None] * len(data_files)
    if len(max_samples_per_file) != len(data_files):
        raise ValueError(
            "--max-samples-per-file must provide one value per data file "
            f"({len(max_samples_per_file)} values for {len(data_files)} files)."
        )
    return [None if int(limit) <= 0 else int(limit) for limit in max_samples_per_file]


def sample_data_file(data_file, limit, global_seed):
    """Reservoir-sample one file deterministically; a non-positive cap keeps all."""
    path = Path(data_file).resolve()
    normalized_limit = None if limit is None or int(limit) <= 0 else int(limit)
    rng = random.Random(derive_file_sampling_seed(global_seed, path))
    selected = []
    source_record_count = 0
    with path.open(encoding="utf-8") as handle:
        first = handle.read(64).lstrip()[:1]
        handle.seek(0)
        rows = iter_json_array(handle) if first == "[" else iter_jsonl(handle)
        for source_record_count, row in enumerate(rows, start=1):
            sampled = SampledDataRow(row, str(path), source_record_count)
            if normalized_limit is None or len(selected) < normalized_limit:
                selected.append(sampled)
                continue
            replacement = rng.randrange(source_record_count)
            if replacement < normalized_limit:
                selected[replacement] = sampled
    rng.shuffle(selected)
    stat = path.stat()
    return FileSamplingResult(
        source_path=str(path),
        source_size_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        configured_limit=normalized_limit,
        source_record_count=source_record_count,
        records=tuple(selected),
    )


def sample_data_files(data_files, max_samples_per_file=None, global_seed=3407):
    limits = normalize_max_samples_per_file(data_files, max_samples_per_file)
    return [
        sample_data_file(data_file, limit, global_seed)
        for data_file, limit in zip(data_files, limits)
    ]


def tokenizer_supports_native_assistant_mask(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if isinstance(chat_template, dict):
        chat_template = "\n".join(str(value) for value in chat_template.values())
    return "{% generation" in str(chat_template)


def chatml_marker_ids(tokenizer):
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if start_id is None or end_id is None or start_id == unk_id or end_id == unk_id:
        raise SystemExit(
            "分词器缺少 <|im_start|>/<|im_end|>，当前 assistant-only loss mask "
            "只支持 Qwen ChatML 模板。"
        )
    return start_id, end_id


def assistant_token_mask(input_ids, tokenizer):
    """Mask assistant content, including its closing ChatML token."""
    start_id, end_id = chatml_marker_ids(tokenizer)
    mask = [0] * len(input_ids)
    size = len(input_ids)
    index = 0
    while index < size:
        if input_ids[index] != start_id:
            index += 1
            continue
        header_end = index + 1
        header = []
        while header_end < size:
            header.append(input_ids[header_end])
            if "\n" in tokenizer.decode(header):
                break
            header_end += 1
        role = tokenizer.decode(header).split("\n", 1)[0].strip()
        content_start = header_end + 1
        turn_end = content_start
        while turn_end < size and input_ids[turn_end] != end_id:
            turn_end += 1
        if role == "assistant":
            for position in range(content_start, min(turn_end + 1, size)):
                mask[position] = 1
        index = turn_end + 1
    return mask


def think_marker_ids(tokenizer):
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if open_id is None or close_id is None or open_id == unk_id or close_id == unk_id:
        return None, None
    return open_id, close_id


def iter_mask_spans(assistant_masks):
    start = None
    for index, value in enumerate(assistant_masks):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index - 1
            start = None
    if start is not None:
        yield start, len(assistant_masks) - 1


def mask_empty_think_blocks(input_ids, assistant_masks, tokenizer):
    """Do not supervise template-injected empty <think></think> prefixes."""
    open_id, close_id = think_marker_ids(tokenizer)
    if open_id is None:
        return assistant_masks
    mask = list(assistant_masks)
    for start, end in iter_mask_spans(assistant_masks):
        if input_ids[start] != open_id:
            continue
        try:
            close = input_ids.index(close_id, start + 1, end + 1)
        except ValueError:
            continue
        if tokenizer.decode(input_ids[start + 1 : close]).strip():
            continue
        content_start = close + 1
        while content_start <= end and not tokenizer.decode([input_ids[content_start]]).strip():
            content_start += 1
        for position in range(start, min(content_start, end + 1)):
            mask[position] = 0
    return mask


def keep_last_assistant_span(assistant_masks):
    last_one = None
    for index, value in enumerate(assistant_masks):
        if value:
            last_one = index
    if last_one is None:
        return assistant_masks
    start = last_one
    while start > 0 and assistant_masks[start - 1]:
        start -= 1
    return [int(start <= index <= last_one) for index in range(len(assistant_masks))]


def normalize_tool_call_arguments(messages):
    """Return messages whose tool-call arguments are Qwen-template mappings.

    Nemotron-SFT-Math-v4 stores OpenAI-style ``function.arguments`` as a JSON
    string. The Qwen3.5/Qwen3.6 chat template iterates over ``arguments|items``
    and therefore requires a mapping. Existing mappings are preserved, while
    null/blank arguments become an empty mapping. The input is not mutated.
    """
    normalized = copy.deepcopy(messages)
    for message_index, message in enumerate(normalized):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {message_index}: expected a mapping")
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        if isinstance(tool_calls, (str, bytes, Mapping)):
            raise ValueError(
                f"message {message_index}: tool_calls must be a list of mappings"
            )
        for tool_call_index, tool_call in enumerate(tool_calls):
            context = f"message {message_index} tool_call {tool_call_index}"
            if not isinstance(tool_call, Mapping):
                raise ValueError(f"{context}: expected a mapping")
            function = tool_call.get("function", tool_call)
            if not isinstance(function, Mapping):
                raise ValueError(f"{context}: function must be a mapping")
            if "arguments" not in function:
                continue
            arguments = function["arguments"]
            if arguments is None:
                function["arguments"] = {}
                continue
            if isinstance(arguments, Mapping):
                function["arguments"] = dict(arguments)
                continue
            if not isinstance(arguments, str):
                raise ValueError(
                    f"{context}: arguments must be a JSON object string or mapping, "
                    f"got {type(arguments).__name__}"
                )
            if not arguments.strip():
                function["arguments"] = {}
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: arguments is not valid JSON: {exc}") from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(
                    f"{context}: arguments JSON must decode to an object, "
                    f"got {type(parsed).__name__}"
                )
            function["arguments"] = dict(parsed)
    return normalized


def encode_messages(
    example,
    tokenizer,
    enable_thinking,
    last_assistant_only=False,
    mask_empty_think=True,
):
    native = tokenizer_supports_native_assistant_mask(tokenizer)
    messages = normalize_tool_call_arguments(example["messages"])
    processed = tokenizer.apply_chat_template(
        messages,
        tools=example.get("tools") or None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=native,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    def unwrap(value):
        return value[0] if value and isinstance(value[0], list) else value

    input_ids = unwrap(processed["input_ids"])
    if native and "assistant_masks" in processed:
        assistant_masks = unwrap(processed["assistant_masks"])
    else:
        assistant_masks = assistant_token_mask(input_ids, tokenizer)
    if last_assistant_only:
        assistant_masks = keep_last_assistant_span(assistant_masks)
    if mask_empty_think:
        assistant_masks = mask_empty_think_blocks(input_ids, assistant_masks, tokenizer)
    return input_ids, assistant_masks


def labels_from_assistant_masks(input_ids, assistant_masks, ignore_index=-100):
    return [token if flag else ignore_index for token, flag in zip(input_ids, assistant_masks)]
