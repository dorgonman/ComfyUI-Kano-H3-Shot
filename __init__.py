from .stack import NODE_CLASS_MAPPINGS as _STACK
from .shot import NODE_CLASS_MAPPINGS as _SHOT
from .refs import NODE_CLASS_MAPPINGS as _REFS

NODE_CLASS_MAPPINGS = {**_STACK, **_SHOT, **_REFS}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KanoH3Stack": "H3 Model Stack",
    "KanoH3Shot": "H3 Shot (Continuum)",
    "KanoH3RefCast": "H3 Reference Images",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
