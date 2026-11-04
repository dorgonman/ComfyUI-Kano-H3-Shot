"""Contract tests for the expanding nodes.

The wrappers expand into a subgraph at execution time. ComfyUI Core 0.43
(``execution.get_output_from_returns``) reads the parent's output values from
the ``result`` key of the returned dict and iterates ``expand`` as a mapping of
node id to node info. These tests pin that contract so a future refactor
cannot silently break the node at runtime.

Run from the package directory:
    D:\\Web\\ComfyUI\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "kano_h3_shot"


def _load_package():
    """Import the package under a valid module name.

    The checkout directory is ``kano-h3-shot`` and the modules use relative
    imports, so the package needs a synthetic importable name.
    """
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


_load_package()

from kano_h3_shot.common import expansion, sage_attention_available  # noqa: E402
from kano_h3_shot.refs import KanoH3RefCast  # noqa: E402
from kano_h3_shot.shot import KanoH3Shot  # noqa: E402
from kano_h3_shot.stack import KanoH3Stack  # noqa: E402


def is_link(value) -> bool:
    """Same shape test ComfyUI's ``comfy_execution.graph_utils.is_link`` uses."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], (int, float))
        and not isinstance(value[1], bool)
    )


def assert_core_contract(testcase: unittest.TestCase, payload: dict, expected_outputs: int) -> None:
    """Assert the payload matches what ComfyUI Core 0.43 expects."""
    testcase.assertIsInstance(payload, dict)
    # Core reads the parent's outputs from "result" and indexes them
    # positionally, so the value must be a sequence in RETURN_TYPES order.
    testcase.assertIn("result", payload, "Core 0.43 reads parent outputs from the 'result' key")
    testcase.assertIsInstance(payload["result"], (tuple, list))
    testcase.assertEqual(len(payload["result"]), expected_outputs)
    for value in payload["result"]:
        testcase.assertTrue(is_link(value), f"output is not a link: {value!r}")
    # Core iterates the subgraph as a mapping of node id -> node info.
    testcase.assertIsInstance(payload["expand"], dict)
    testcase.assertTrue(payload["expand"], "expanded subgraph is empty")
    for node_id, info in payload["expand"].items():
        testcase.assertIsInstance(node_id, str)
        testcase.assertTrue(node_id)
        testcase.assertTrue(info.get("class_type"), f"node {node_id} has no class_type")
        testcase.assertIn("inputs", info, f"node {node_id} has no inputs")


def shot_payload(**overrides):
    kwargs = dict(
        model=None,
        clip=None,
        video_vae=None,
        audio_vae=None,
        sampler=None,
        sigmas=None,
        sequence_prompt="[Chunk 1]\nBeat 1.",
        chunks=1,
        chunk_seconds=5.0,
        aspect="Portrait 9:16",
        quality="Draft 0.30 MP",
        base_seed=1,
        run_name="unit_test_run",
        mode="Full Run",
    )
    kwargs.update(overrides)
    return KanoH3Shot().expand(**kwargs)


class ExpansionContractTests(unittest.TestCase):
    def test_stack_expansion_contract(self):
        payload = KanoH3Stack().expand(
            preset="H3 int8 (24GB / RTX 3090)",
            sampler="res_multistep",
            scheduler="simple",
            steps=30,
            denoise=1.0,
            sage_attention=False,
        )
        assert_core_contract(self, payload, expected_outputs=6)
        classes = {info["class_type"] for info in payload["expand"].values()}
        self.assertIn("UNETLoader", classes)
        self.assertIn("CLIPLoader", classes)
        self.assertNotIn("MiniMaxH3MemoryEfficientSageAttentionPatch", classes)

    def test_stack_sage_attention_requires_supported_package(self):
        """Requesting sage attention without the required API must fail loudly."""
        if sage_attention_available():
            self.skipTest("installed sageattention supports the H3 patch")
        with self.assertRaisesRegex(RuntimeError, "sageattention"):
            KanoH3Stack().expand(
                preset="H3 int8 (24GB / RTX 3090)",
                sampler="res_multistep",
                scheduler="simple",
                steps=30,
                denoise=1.0,
                sage_attention=True,
            )

    def test_refcast_expansion_contract(self):
        payload = KanoH3RefCast().expand(reference_use="All chunks")
        assert_core_contract(self, payload, expected_outputs=1)
        info = next(iter(payload["expand"].values()))
        self.assertEqual(info["class_type"], "H3ContinuumReferenceImagesV39")

    def test_shot_expansion_contract(self):
        payload = shot_payload(chunks=2, chunk_seconds=15.0, quality="Balanced 0.60 MP")
        assert_core_contract(self, payload, expected_outputs=4)
        classes = {info["class_type"] for info in payload["expand"].values()}
        for expected in ("H3ContinuumSamplerV39", "H3DecodeCacheHelper",
                         "H3ContinuumAssembleSeamV35", "CreateVideo", "SaveVideo"):
            self.assertIn(expected, classes)

    def test_shot_create_video_bit_depth_is_an_integer(self):
        """Core's CreateVideo exposes bit_depth as an int combo, not a string."""
        payload = shot_payload()
        video = next(info for info in payload["expand"].values()
                     if info["class_type"] == "CreateVideo")
        self.assertEqual(video["inputs"]["bit_depth"], 8)
        self.assertIsInstance(video["inputs"]["bit_depth"], int)

    def test_shot_uses_manual_size_without_first_frame(self):
        payload = shot_payload()
        sampler = next(info for info in payload["expand"].values()
                       if info["class_type"] == "H3ContinuumSamplerV39")
        self.assertEqual(sampler["inputs"]["size_source"], "Manual")
        self.assertEqual(sampler["inputs"]["width"] % 32, 0)
        self.assertEqual(sampler["inputs"]["height"] % 32, 0)

    def test_expansion_helper_keeps_readable_alias(self):
        payload = expansion(
            {"a": {"class_type": "X", "inputs": {}}}, (["a", 0],), names=("out",)
        )
        self.assertEqual(payload["result"], (["a", 0],))
        self.assertEqual(payload["outputs"], {"out": ["a", 0]})
        self.assertFalse(payload["visible_to_parent"])

    def test_shot_is_an_output_node(self):
        """The shot node must be an output node or the branch is pruned.

        ComfyUI seeds the execution list with the output node ids and then
        walks backwards through their inputs. A subgraph only materialises when
        its parent node executes, so a non-output shot node with no consumer
        downstream is never scheduled: the prompt reports success in 0.06
        seconds and renders nothing.
        """
        self.assertTrue(KanoH3Shot.OUTPUT_NODE)

    def test_shot_expansion_contains_a_save_node(self):
        """The expanded subgraph has to contain the node that writes the file."""
        payload = shot_payload()
        classes = {info["class_type"] for info in payload["expand"].values()}
        self.assertIn("SaveVideo", classes)

    def test_node_function_returns_the_payload_dict(self):
        """The node function must return the payload dict itself.

        Core collects the function's return value into a list and then checks
        ``isinstance(r, dict)`` on the element. A one element tuple made that
        check fail: the subgraph was never recognised, the node reported one
        list output instead of its declared outputs, and the downstream link
        raised "list index out of range" in the executor.
        """
        for payload in (
            shot_payload(),
            KanoH3Stack().expand(
                preset="H3 int8 (24GB / RTX 3090)",
                sampler="res_multistep",
                scheduler="simple",
                steps=30,
                denoise=1.0,
                sage_attention=False,
            ),
            KanoH3RefCast().expand(reference_use="All chunks"),
        ):
            self.assertIsInstance(payload, dict)
            self.assertIn("expand", payload)
            self.assertIn("result", payload)

    def test_core_execution_engine_accepts_the_payload(self):
        """Feed the payload to Core's own return-value reader.

        ``execution.get_output_from_returns`` is the function that decides
        whether an expanding node hands its outputs back to the parent, so a
        mismatch there is a runtime failure, not a cosmetic issue. Core calls it
        with the list of node function return values, so the payload is passed
        as a single element of that list.
        """
        try:
            import comfy.utils  # noqa: F401
            from execution import get_output_from_returns
        except Exception:
            self.skipTest("ComfyUI core is not importable from this environment")
        payload = shot_payload()
        output, ui, has_subgraph = get_output_from_returns([payload], KanoH3Shot)
        self.assertTrue(has_subgraph)
        self.assertEqual(len(output), 1)
        new_graph, node_outputs = output[0]
        self.assertIsInstance(new_graph, dict)
        self.assertIsInstance(node_outputs, (tuple, list))
        self.assertEqual(len(node_outputs), len(KanoH3Shot.RETURN_TYPES))
        for value in node_outputs:
            self.assertTrue(is_link(value), f"not a link: {value!r}")

    def test_core_reads_the_declared_output_count(self):
        """The subgraph must resolve to exactly RETURN_TYPES outputs.

        ComfyUI merges the resolved subgraph values into the parent's output
        list and then indexes that list per outgoing link. If the merge
        produced fewer entries than the links expect, the executor raises
        "list index out of range".
        """
        try:
            import comfy.utils  # noqa: F401
            from execution import get_output_from_returns, merge_result_data
        except Exception:
            self.skipTest("ComfyUI core is not importable from this environment")
        payload = shot_payload()
        output, _, _ = get_output_from_returns([payload], KanoH3Shot)
        _, node_outputs = output[0]
        merged = merge_result_data([tuple(node_outputs)], KanoH3Shot)
        self.assertEqual(len(merged), len(KanoH3Shot.RETURN_TYPES))


if __name__ == "__main__":
    unittest.main()
