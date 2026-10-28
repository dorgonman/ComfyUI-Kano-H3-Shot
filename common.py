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


def model_exists(name: str, folder_key: str) -> bool:
    """True when the named file is present in the given model folder."""
    try:
        import folder_paths

        path = folder_paths.get_full_path_or_raise(folder_key, name)
    except Exception:
        return False
    return bool(path) and os.path.exists(path)
