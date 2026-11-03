# ComfyUI Kano H3 Shot

Collapse the MiniMax H3 **Continuum** long-form video graph from ~17 nodes into 3.

| Node | Replaces | What you set |
|---|---|---|
| `KanoH3Stack` | UNETLoader + CLIPLoader + VAELoader x2 + KSamplerSelect + BasicScheduler + SageAttention patch | model preset, sampler, steps |
| `KanoH3RefCast` | ReferenceImagesV39 + 9 image loaders | up to 9 reference images (`@R1`..`@R9`) |
| `KanoH3Shot` | SamplerV39 + DecodeCacheHelper + AssembleSeamV35 + CreateVideo + SaveVideo | chunks, seconds, aspect, quality, seed, run name |

## Layout

```
KanoH3Stack ── model / clip / video_vae / audio_vae / sampler / sigmas ──> KanoH3Shot
KanoH3RefCast ── reference_images ──────────────────────────────────────> KanoH3Shot
KanoH3Shot ── images / audio / status / report
```

`KanoH3Shot` writes the finished MP4 to `output/video/<run_name>_*.mp4` itself, so
no output nodes are needed downstream.

## Model presets

`H3 int8 (24GB / RTX 3090)` is the default and expects these files:

```
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_int8_convrot.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
```

From `Comfy-Org/MiniMax-H3` on Hugging Face.

## Sequence prompt

`sequence_prompt` accepts all three Continuum formats, auto-detected:

- **Fixed** — one block describing the whole sequence
- **List** — chunks separated by a line containing only `---`
- **Timeline** — `[0-15s]` headers, with a global preamble above the first header

Scene description in English and spoken lines in the target language inside
quotation marks work together; H3 synthesises the speech natively.

## Run names and takes

`run_name` is the saved-run identity. Name it per scene (`scene1_chamber`) so
**Settings → H3Continuum → Project Manager** can list, resume, and branch it.
Raw AV chunks land in `output/h3_continuum/runs/<run_name>/`.

## Requirements

- ComfyUI with MiniMax H3 support
- `ComfyUI-H3-Continuum` (the nodes this package wraps)
- `ComfyUI-KJNodes` (only for the optional SageAttention switch)
- `rgthree-comfy` is **not** required

## ComfyUI compatibility

These nodes expand into a subgraph at execution time. ComfyUI Core 0.43 reads
the expanded node's outputs from the `result` key of the returned payload and
indexes that value **positionally**, in `RETURN_TYPES` order
(`execution.get_output_from_returns` -> `node_outputs[i]`). The shared
`common.expansion()` helper builds that payload, and `tests/test_expansion_contract.py`
pins the contract — including a check that feeds the payload to Core's own
return-value reader.

`CreateVideo.bit_depth` is passed as the integer `8`, because Core exposes it
as an int combo (`auto`, `8`, `10`), not as a string.

## SageAttention

`KanoH3Stack.sage_attention` is **off by default**. The KJNodes MiniMax H3 sage
patch imports `per_thread_int8_triton` and `get_cuda_arch_versions` from
`sageattention.core`; releases before 2.x do not provide them, and the patch
then raises at execution time. Turning the switch on with an unsupported
package now fails immediately with an actionable message instead of a late
runtime error.

## Tests

From this directory:

```
D:\Web\ComfyUI\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Set `PYTHONPATH` to the ComfyUI root to also run the check that hands the
expansion payload to Core's execution engine; without it that one test skips.
