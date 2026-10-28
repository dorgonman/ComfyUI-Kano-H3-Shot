"""KanoH3Stack: one node that loads the whole MiniMax H3 model stack.

Replaces the UNETLoader + CLIPLoader + VAELoader x2 + KSamplerSelect +
BasicScheduler + SageAttention patch chain (7 nodes) with a single preset
dropdown.
"""

from __future__ import annotations

from .common import DEFAULT_PRESET, MODEL_PRESETS, link, next_uid, node

SAMPLERS = ["res_multistep", "euler", "dpmpp_2m", "dpmpp_3m_sde"]
SCHEDULERS = ["simple", "sgm_uniform", "karras", "normal"]


class KanoH3Stack:
    """Load MiniMax H3 weights and produce the sampler/scheduler pair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(MODEL_PRESETS.keys()), {"default": DEFAULT_PRESET}),
                "sampler": (SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sage_attention": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply the memory-efficient SageAttention patch. "
                        "Needs the sageattention package; turn off if it is missing.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "SAMPLER", "SIGMAS")
    RETURN_NAMES = (
        "model",
        "clip",
        "video_vae",
        "audio_vae",
        "sampler",
        "sigmas",
    )
    FUNCTION = "expand"
    CATEGORY = "Kano/H3"
    DESCRIPTION = (
        "Loads the MiniMax H3 diffusion model, Qwen3-VL text encoder, video VAE "
        "and audio VAE, plus the sampler and sigma schedule the Continuum sampler "
        "expects. Collapses seven loader nodes into one preset."
    )

    def expand(self, preset, sampler, scheduler, steps, denoise, sage_attention):
        cfg = MODEL_PRESETS.get(preset, MODEL_PRESETS[DEFAULT_PRESET])

        unet = next_uid("unet")
        clip = next_uid("clip")
        vvae = next_uid("vvae")
        avae = next_uid("avae")
        sel = next_uid("sel")
        sched = next_uid("sched")

        graph = {
            unet: node(
                "UNETLoader",
                widgets=[cfg["unet_name"], "default"],
                inputs={"unet_name": cfg["unet_name"], "weight_dtype": "default"},
            ),
            clip: node("CLIPLoader",
                widgets=[cfg["clip_name"], "minimax", "default"],
                inputs={"clip_name": cfg["clip_name"], "type": "minimax", "device": "default"},
            ),
            vvae: node("VAELoader",
                widgets=[cfg["video_vae_name"]],
                inputs={"vae_name": cfg["video_vae_name"]},
            ),
            avae: node("VAELoader",
                widgets=[cfg["audio_vae_name"]],
                inputs={"vae_name": cfg["audio_vae_name"]},
            ),
            sel: node("KSamplerSelect",
                widgets=[sampler],
                inputs={"sampler_name": sampler},
            ),
        }

        model_src = link(unet, 0)
        if sage_attention:
            patch = next_uid("sage")
            graph[patch] = node("MiniMaxH3MemoryEfficientSageAttentionPatch",
                inputs={"model": model_src},
            )
            model_src = link(patch, 0)

        sched_node = node("BasicScheduler",
            widgets=[scheduler, steps, denoise],
            inputs={"model": model_src, "scheduler": scheduler, "steps": steps, "denoise": denoise},
        )
        graph[sched] = sched_node

        return ({
            "expand": graph,
            "visible_to_parent": False,
            "outputs": {
                "model": model_src,
                "clip": link(clip, 0),
                "video_vae": link(vvae, 0),
                "audio_vae": link(avae, 0),
                "sampler": link(sel, 0),
                "sigmas": link(sched, 0),
            },
        },)


NODE_CLASS_MAPPINGS = {"KanoH3Stack": KanoH3Stack}
