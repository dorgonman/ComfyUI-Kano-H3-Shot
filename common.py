"""Shared helpers for the Kano H3 Shot wrappers."""

from __future__ import annotations

import os
import threading

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------
# Each preset names files relative to the ComfyUI model folders. They are
# resolved by the loader nodes at execution time, so a missing file surfaces as
# a normal ComfyUI validation error instead of a silent fallback.

MODEL_PRESETS = {
    "H3 int8 (24GB / RTX 3090)": {
        "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae_name": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio_vae_name": "minimax_h3_audio_vae_fp32.safetensors",
    },
    "H3 fp16 (full precision)": {
        "unet_name": "minimax_h3_ref2va_pruned_fp16.safetensors",
        "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae_name": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae_name": "minimax_h3_audio_vae_fp32.safetensors",
    },
}

DEFAULT_PRESET = next(iter(MODEL_PRESETS))

# Resolution presets exposed on KanoH3Shot, mapped to the sampler's own enum.
QUALITY_PRESETS = {
    "Draft 0.30 MP": "Draft — 0.30 MP",
    "Balanced 0.60 MP": "Balanced — 0.60 MP",
    "Native 768": "Native 768",
}

# Aspect label -> (sampler legacy aspect, manual width, manual height).
# Widths/heights are multiples of 32 and stay near the preset's megapixel budget.
ASPECTS = {
    "Portrait 9:16": ("Portrait 9:16", 480, 832),
    "Landscape 16:9": ("Landscape 16:9", 832, 480),
    "Square 1:1": ("Square 1:1", 640, 640),
}

CONTINUITY = {
    "Balanced (22 frames)": "Balanced — 22 frames",
    "Fast (5 frames)": "Fast — 5 frames",
    "Strong (39 frames)": "Strong — 39 frames (Experimental)",
    "Auto (conservative)": "Auto — conservative",
}

# ---------------------------------------------------------------------------
# Graph expansion helpers
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_UID = 0


def next_uid(prefix: str) -> str:
    """Return a unique node id that cannot collide with a litegraph integer id."""
    global _UID
    with _LOCK:
        _UID += 1
        return f"{prefix}.{_UID}"


def node(class_type: str, inputs=None, widgets=None) -> dict:
    """Build one entry of the API-format subgraph dictionary."""
    entry = {"class_type": class_type}
    if inputs:
        entry["inputs"] = dict(inputs)
    if widgets is not None:
        entry["widgets_values"] = list(widgets)
    return entry


def link(uid: str, slot: int = 0) -> list:
    """Reference an output of another node in the same subgraph."""
    return [uid, slot]


def expansion(graph: dict, outputs, names=None) -> dict:
    """Build the return value ComfyUI Core expects from an expanding node.

    Core 0.43 reads the parent's output values from the ``result`` key of the
    returned dict (``execution.get_output_from_returns``) and then indexes that
    value **positionally** (``node_outputs[i]``), so it must be a sequence in
    ``RETURN_TYPES`` order — a mapping keyed by output name raises a KeyError
    at runtime.

    The function must return the dict itself, not a one element tuple. Core
    collects the node function's return value into a list
    (``_async_map_node_over_list``) and then checks ``isinstance(r, dict)`` on
    each element. Wrapping the payload in a tuple made that check fail, the
    subgraph was never recognised, and the node reported a single list output
    instead of its six declared outputs.

    ``visible_to_parent`` is not read by Core 0.43 and is kept only so that a
    caller inspecting the payload can tell the subgraph is internal. The
    ``outputs`` mapping is a readable alias and is ignored by Core.
    """
    ordered = tuple(outputs)
    payload = {
        "expand": graph,
        "result": ordered,
        "visible_to_parent": False,
    }
    if names:
        payload["outputs"] = dict(zip(names, ordered))
    return payload


def sage_attention_available() -> bool:
    """True when the installed sageattention exposes the API KJNodes needs.

    KJNodes' MiniMax H3 sage patch imports ``per_thread_int8_triton`` and
    ``get_cuda_arch_versions`` from ``sageattention.core``. Older releases
    (1.0.6 on this machine) do not have them, and the patch node then raises at
    execution time.
    """
    try:
        from sageattention.core import (  # type: ignore
            get_cuda_arch_versions,
            per_thread_int8_triton,
        )
    except Exception:
        return False
    return callable(get_cuda_arch_versions) and callable(per_thread_int8_triton)


def model_exists(name: str, folder_key: str) -> bool:
    """True when the named file is present in the given model folder."""
    try:
        import folder_paths

        path = folder_paths.get_full_path_or_raise(folder_key, name)
    except Exception:
        return False
    return bool(path) and os.path.exists(path)
