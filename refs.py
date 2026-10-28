"""KanoH3RefCast: nine optional IMAGE inputs -> one reference-images bundle.

The Continuum Reference Images V3.9 node needs ten widget values plus nine
links. This wrapper keeps only the images and a single "use" switch, and pins
every connected image to all chunks (the behaviour a shot list normally wants,
so ``@R1`` stays valid in every chunk prompt).
"""

from __future__ import annotations

from .common import link, next_uid, node

REF_SLOTS = 9


class KanoH3RefCast:
    """Bundle up to nine reference images for the Continuum sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "reference_use": (
                ["All chunks", "Per chunk"],
                {
                    "default": "All chunks",
                    "tooltip": "All chunks keeps @R1..@R9 valid in every chunk prompt.",
                },
            ),
        }
        optional = {
            f"image_{i}": ("IMAGE", {"tooltip": f"Reference image {i} — use @R{i} in the prompt."})
            for i in range(1, REF_SLOTS + 1)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("H3_CONTINUUM_REFERENCE_IMAGES_V39",)
    RETURN_NAMES = ("reference_images",)
    FUNCTION = "expand"
    CATEGORY = "Kano/H3"
    DESCRIPTION = (
        "Connect character or set reference images here and plug the result into "
        "KanoH3Shot. Reference 1 is addressed as @R1 in the sequence prompt."
    )

    def expand(self, reference_use, **kwargs):
        refs = next_uid("refs")
        inputs = {"reference_use": reference_use}
        widgets = [reference_use]

        for i in range(1, REF_SLOTS + 1):
            chunks_key = f"reference_r{i}_chunks"
            img = kwargs.get(f"image_{i}")
            if img is None:
                widgets.append("off")
                continue
            widgets.append("on" if reference_use == "All chunks" else "all")
            inputs[f"reference_image_{i}"] = img

        graph = {
            refs: node("H3ContinuumReferenceImagesV39",
                inputs=inputs,
                widgets=widgets,
            )
        }

        return ({
            "expand": graph,
            "visible_to_parent": False,
            "outputs": {"reference_images": link(refs, 0)},
        },)


NODE_CLASS_MAPPINGS = {"KanoH3RefCast": KanoH3RefCast}
