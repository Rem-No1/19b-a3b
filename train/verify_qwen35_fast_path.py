#!/usr/bin/env python3
"""GPU forward/backward smoke test for the Qwen3.5 linear-attention fast path."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3.5 FLA and causal-conv1d CUDA training paths."
    )
    parser.add_argument("--model-path", default="/model")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--skip-model-layer",
        action="store_true",
        help="Only test the low-level kernels; do not read a model config.",
    )
    return parser.parse_args()


def verify_low_level_kernels(sequence_length: int) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    x = torch.randn(2, 64, sequence_length, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(64, 4, device=device, dtype=dtype, requires_grad=True)
    conv_out = causal_conv1d_fn(x, weight, activation="silu")
    conv_out.float().square().mean().backward()
    assert x.grad is not None and weight.grad is not None

    shape = (2, sequence_length, 2, 64)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k_raw = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    g_raw = torch.randn(2, sequence_length, 2, device=device, dtype=dtype, requires_grad=True)
    beta_raw = torch.randn(2, sequence_length, 2, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(k_raw, p=2, dim=-1)
    g = F.logsigmoid(g_raw.float()).to(dtype)
    beta = beta_raw.sigmoid()
    fla_out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
    )
    fla_out.float().square().mean().backward()
    for tensor in (q, k_raw, v, g_raw, beta_raw):
        assert tensor.grad is not None

    return {
        "causal_conv_output": tuple(conv_out.shape),
        "fla_output": tuple(fla_out.shape),
        "fla_final_state": tuple(final_state.shape),
    }


def verify_model_layer(model_path: str, sequence_length: int) -> dict[str, object]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = getattr(config, "text_config", config)
    layer_idx = config.layer_types.index("linear_attention")
    layer = Qwen3_5MoeGatedDeltaNet(config, layer_idx=layer_idx).to(
        device="cuda", dtype=torch.bfloat16
    )
    layer.train()

    assert layer.causal_conv1d_fn.__module__.startswith("causal_conv1d")
    assert layer.chunk_gated_delta_rule.__module__.startswith("fla.")
    assert layer.recurrent_gated_delta_rule.__module__.startswith("fla.")

    hidden_states = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = layer(hidden_states)
    output.float().square().mean().backward()
    assert hidden_states.grad is not None
    assert layer.in_proj_qkv.weight.grad is not None
    assert layer.conv1d.weight.grad is not None
    assert layer.out_proj.weight.grad is not None

    return {
        "model_type": config.model_type,
        "layer_idx": layer_idx,
        "input": tuple(hidden_states.shape),
        "output": tuple(output.shape),
        "causal_conv_impl": layer.causal_conv1d_fn.__module__,
        "chunk_rule_impl": layer.chunk_gated_delta_rule.__module__,
        "recurrent_rule_impl": layer.recurrent_gated_delta_rule.__module__,
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length < 64:
        raise ValueError("--sequence-length must be at least 64")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; start the container with --gpus")
    if not is_flash_linear_attention_available():
        raise RuntimeError("Transformers cannot detect flash-linear-attention")
    if not is_causal_conv1d_available():
        raise RuntimeError("Transformers cannot detect causal-conv1d")

    torch.manual_seed(0)
    result: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": metadata.version("triton"),
        "flash_linear_attention": metadata.version("flash-linear-attention"),
        "fla_core": metadata.version("fla-core"),
        "causal_conv1d": metadata.version("causal-conv1d"),
        "tilelang": metadata.version("tilelang"),
        "low_level": verify_low_level_kernels(args.sequence_length),
    }
    if not args.skip_model_layer:
        result["qwen35_layer"] = verify_model_layer(args.model_path, args.sequence_length)
    torch.cuda.synchronize()
    result["forward_backward"] = "ok"
    print(result)


if __name__ == "__main__":
    main()
