"""KanoH3Shot: the whole Continuum long-form video chain in one node.

Expands at runtime into H3ContinuumSamplerV39 -> H3DecodeCacheHelper ->
H3ContinuumAssembleSeamV35 -> CreateVideo -> SaveVideo, exposing only the
knobs that actually change a shot: chunk count, seconds, aspect, quality,
seed, and the run name used by Project/Take storage.
"""

from __future__ import annotations

from .common import (
    ASPECTS,
    CONTINUITY,
    QUALITY_PRESETS,
    link,
    next_uid,
    node,
)

MODES = ["Full Run", "Review Each Chunk"]


class KanoH3Shot:
    """Generate a Continuum sequence and write the finished video to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "sequence_prompt": ("STRING", {"multiline": True, "default": ""}),
                "chunks": ("INT", {"default": 2, "min": 1, "max": 16}),
                "chunk_seconds": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.5}),
                "aspect": (list(ASPECTS.keys()), {"default": "Portrait 9:16"}),
                "quality": (list(QUALITY_PRESETS.keys()), {"default": "Draft 0.30 MP"}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "run_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Saved-run identity. Name it per scene, e.g. scene1_chamber, "
                        "so Project/Take Manager can resume or branch it.",
                    },
                ),
                "mode": (
                    MODES,
                    {
                        "default": "Full Run",
                        "tooltip": "Review Each Chunk stops after every chunk so you can approve "
                        "before continuing.",
                    },
                ),
            },
            "optional": {
                "reference_images": ("H3_CONTINUUM_REFERENCE_IMAGES_V39",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "status", "report")
    FUNCTION = "expand"
    OUTPUT_NODE = False
    CATEGORY = "Kano/H3"
    DESCRIPTION = (
        "MiniMax H3 Continuum sampler plus decode, seam assembly and MP4 output. "
        "Prompt format is auto-detected: one block, --- separated chunks, or "
        "[0-15s] timeline headers."
    )

    def expand(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        sampler,
        sigmas,
        sequence_prompt,
        chunks,
        chunk_seconds,
        aspect,
        quality,
        base_seed,
        run_name,
        mode,
        reference_images=None,
        first_frame=None,
        last_frame=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        aspect_label, width, height = ASPECTS.get(aspect, ASPECTS["Portrait 9:16"])
        preset = QUALITY_PRESETS.get(quality, "Draft — 0.30 MP")

        sampler_uid = next_uid("sampler")
        decode_uid = next_uid("decode")
        seam_uid = next_uid("seam")
        video_uid = next_uid("video")
        save_uid = next_uid("save")

        # --- sampler -----------------------------------------------------
        sampler_inputs = {
            "model": model,
            "clip": clip,
            "video_vae": video_vae,
            "sampler": sampler,
            "sigmas": sigmas,
            "sequence_prompt": sequence_prompt,
            "audio_vae": audio_vae,
            "prompt_mode": "Auto",
            "chunks": chunks,
            "chunk_seconds": chunk_seconds,
            "aspect": aspect_label,
            "preset": preset,
            "custom_mp": 0.3,
            "continuity": CONTINUITY["Balanced (22 frames)"],
            "base_seed": base_seed,
            "audio_continuity": True,
            "diagnostics": "Detailed Report",
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "strict_compatibility": False,
            "debug": False,
            "show_preview": True,
            "run_storage": "Save + Auto Resume",
            "run_name": run_name,
            "reference_size": "Match Output",
            "project_id": "",
            "video_reference_size": "Efficient - 0.4 MP",
            "continuation_backend": "Standard",
            "generation_mode": mode,
            "review_action": "Continue / Next",
            "take_group": 0,
            "take_revision_id": "",
            "take_action": "Automatic",
            # Manual size keeps the canvas stable when there is no first frame.
            "size_source": "First Image" if first_frame is not None else "Manual",
            "width": width,
            "height": height,
        }
        if first_frame is not None:
            sampler_inputs["first_frame"] = first_frame
        if last_frame is not None:
            sampler_inputs["last_frame"] = last_frame
        if reference_images is not None:
            sampler_inputs["reference_images"] = reference_images

        sampler_widgets = [
            "Auto", chunks, chunk_seconds, aspect_label, preset, 0.3,
            CONTINUITY["Balanced (22 frames)"], base_seed, True, "Detailed Report",
            "Auto", 0, False, False, True, "Save + Auto Resume", run_name,
            "Match Output", "", "Efficient - 0.4 MP", "Standard", mode,
            "Continue / Next", 0, "", "Automatic",
            sampler_inputs["size_source"], width, height,
        ]

        graph = {
            sampler_uid: node("H3ContinuumSamplerV39", inputs=sampler_inputs, widgets=sampler_widgets
            ),
            decode_uid: node("H3DecodeCacheHelper",
                inputs={
                    "cache_mode": "Auto",
                    "ram_budget_mb": 256,
                    "disk_budget_gb": 8,
                    "reset_token": 0,
                    "video_samples": link(sampler_uid, 0),
                    "video_vae": video_vae,
                    "audio_samples": link(sampler_uid, 1),
                    "audio_vae": audio_vae,
                },
                widgets=["Auto", 256, 8, 0],
            ),
            seam_uid: node("H3ContinuumAssembleSeamV35",
                inputs={
                    "images": link(decode_uid, 0),
                    "audio": link(decode_uid, 1),
                    "assembly_plan": link(sampler_uid, 2),
                    "exact_total_duration": True,
                    "audio_seam": "Auto",
                    "video_seam": "Auto",
                    "buffer_backend": "Auto",
                    "diagnostics": "Detailed Report",
                },
                widgets=[True, "Auto", "Auto", "Auto", "Detailed Report"],
            ),
            video_uid: node("CreateVideo",
                inputs={
                    "images": link(seam_uid, 0),
                    "fps": 24.0,
                    "audio": link(seam_uid, 1),
                    "bit_depth": "8",
                    "color_space": "sRGB",
                    "codec": "none",
                },
                widgets=[24.0, 8, "sRGB", "none"],
            ),
            save_uid: node("SaveVideo",
                inputs={
                    "video": link(video_uid, 0),
                    "filename_prefix": f"video/{run_name or 'KanoH3'}",
                    "format": "auto",
                    "codec": "auto",
                },
                widgets=[f"video/{run_name or 'KanoH3'}", "auto", "auto"],
            ),
        }

        return ({
            "expand": graph,
            "visible_to_parent": False,
            "outputs": {
                "images": link(seam_uid, 0),
                "audio": link(seam_uid, 1),
                "status": link(sampler_uid, 3),
                "report": link(seam_uid, 2),
            },
        },)


NODE_CLASS_MAPPINGS = {"KanoH3Shot": KanoH3Shot}
