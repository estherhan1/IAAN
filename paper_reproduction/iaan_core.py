"""Model-agnostic IAAN primitives corresponding to SLT 2026, Sec. III.

This module intentionally does not decide how an audio model encodes a waveform,
how prompts are built, or how VoxParadox predictions are parsed. Those steps
must match each model's official evaluation protocol.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn


def gaussian_reference(waveform: Tensor, seed: int) -> Tensor:
    """Fixed-seed Gaussian reference matching waveform mean and variance.

    Computes population statistics; returns a same-shape float32 waveform.
    """
    x = waveform.detach().to(device="cpu", dtype=torch.float32)
    if x.numel() == 0:
        raise ValueError("waveform must not be empty")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32)
    # Clamp handles silent/constant audio without NaNs.
    return ((noise - noise.mean()) /
            noise.std(unbiased=False).clamp_min(1e-8) *
            x.std(unbiased=False) + x.mean()).to(waveform.device)


@torch.no_grad()
def collect_means(
    encoder_forward: Callable[[], object],
    fc2_layers: Sequence[nn.Module],
) -> list[Tensor]:
    """Capture the FFN activations INPUT to each fc2 (after nonlinearity).

    Assumes the model's fc2 input has neuron dimension last. Averaging over
    all preceding dimensions is appropriate for a single unpadded clip.
    For padded batches, replace pooling with a mask-aware implementation.
    """
    captured: dict[int, Tensor] = {}
    handles = []

    def register(i: int, layer: nn.Module) -> None:
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError("fc2 hook received no inputs")
            x = inputs[0]
            if not isinstance(x, Tensor) or x.ndim < 2:
                raise ValueError("expected (..., neuron_width) activation tensor")
            captured[i] = x.detach().float().reshape(-1, x.shape[-1]).mean(0).cpu()
        handles.append(layer.register_forward_pre_hook(hook))

    for i, layer in enumerate(fc2_layers):
        register(i, layer)
    try:
        encoder_forward()
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(fc2_layers):
        missing = sorted(set(range(len(fc2_layers))) - set(captured))
        raise RuntimeError(f"encoder did not visit fc2 layer indices: {missing}")
    return [captured[i] for i in range(len(fc2_layers))]


def select_topk(
    real_means: Sequence[Tensor],
    reference_means: Sequence[Tensor],
    k: int,
) -> dict[int, Tensor]:
    """Signed mean(real) - mean(reference), top K across ALL encoder layers.

    Returns layer index -> selected local neuron indices (CPU long tensors).
    Scores are NOT normalized and NOT converted to absolute values.
    """
    if not real_means or len(real_means) != len(reference_means):
        raise ValueError("real and reference layers must be non-empty and aligned")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    sizes = []
    scores = []
    for real, ref in zip(real_means, reference_means):
        if real.ndim != 1 or ref.shape != real.shape:
            raise ValueError("each layer needs matching one-dimensional means")
        sizes.append(real.numel())
        scores.append(real.float().cpu() - ref.float().cpu())
    flat = torch.cat(scores)
    if k > flat.numel():
        raise ValueError("k exceeds number of encoder FFN neurons")
    chosen = torch.topk(flat, k=k, sorted=False).indices
    selected = {}
    offset = 0
    for layer_index, width in enumerate(sizes):
        local = chosen[(chosen >= offset) & (chosen < offset + width)] - offset
        if local.numel():
            selected[layer_index] = local
        offset += width
    return selected


@contextmanager
def amplify_fc2_inputs(
    fc2_layers: Sequence[nn.Module],
    selected: Mapping[int, Tensor],
    gain: float,
):
    """Temporarily multiply selected encoder FFN neuron activations by gain.

    Hooks only affect the context-managed real-audio generation forward pass.
    They are removed even when generation raises an exception.
    """
    if gain <= 1:
        raise ValueError("IAAN gain must be > 1")
    handles = []
    for layer_index, indices in selected.items():
        if layer_index < 0 or layer_index >= len(fc2_layers):
            raise ValueError(f"unknown encoder layer index {layer_index}")
        idx_cpu = indices.detach().long().cpu().clone()

        def make_hook(idx):
            def hook(_module, inputs):
                activation = inputs[0]
                out = activation.clone()
                local_idx = idx.to(out.device)
                out[..., local_idx] *= gain
                return (out, *inputs[1:])
            return hook

        handles.append(fc2_layers[layer_index].register_forward_pre_hook(make_hook(idx_cpu)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
