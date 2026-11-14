# RECOMMENDATIONS — ranked by (payoff × confidence) ÷ effort

One page. **2026-11-13.** Scores are my judgement on a 1–5 scale: payoff = how much it moves the two consumers,
confidence = how sure I am the claimed effect is real on *our* machine, effort = engineer-days (1 = hours, 5 = a week).
`Score = P·C/E`.

---

## A. Do next — highest score first

| # | Action | Owner | P | C | E | **Score** | Why now |
|---|---|---|---:|---:|---:|---:|---|
| **1** | **Import the measured reference table + the 10 validation gates.** `PBR_REFERENCE` dict from `PBR_REFERENCE_VALUES.md` §8, plus the gate table: SDR, unit-normals, ORM channel audit, roughness variance, albedo plausibility, *albedo-is-lighting-free plane fit*. | pbr-texture-core | 5 | 5 | 1 | **25** | Every later decision (is this texture good?) becomes a boolean instead of an opinion. Pure numpy, half a day. |
| **2** | **Fix the sRGB/Non-Color + ORM packing contract** as a hard invariant: albedo sRGB, all data maps `Non-Color`, **R=AO / G=Rough / B=Metal / A=255**, Unity variant behind a flag. | pbr-texture-core | 5 | 5 | 1 | **25** | Silent, pervasive, unrecoverable. I verified in the wild that Poly Haven's `arm` ≡ our ORM, and that Unity is the one that reorders+inverts. |
| **3** | **Fix Blender 5.x baking now**: `scene.cycles.bake_type` (NOT `scene.render.bake.type`, which in 5.3 accepts only NORMALS/DISPLACEMENT/VECTOR_DISPLACEMENT and raises on `'COMBINED'`), and the "select **and** activate an image node in *every* material" rule. | blender-material-bridge | 5 | 5 | 1 | **25** | It is a one-line fix standing between you and a day of "why is my bake black". Reproduced the exact `ValueError` here. |
| **4** | **Verify texture embedding by parsing the GLB**, not by trusting `export_image_format`. Assert `images[].bufferView` (not a URI) and that one shared image backs both `occlusionTexture` and `metallicRoughnessTexture`. | blender-material-bridge | 5 | 5 | 1 | **25** | 12 lines; catches the most common shipping failure. Verified a real GLB doing exactly this. |
| **5** | **Ship `make_seamless()` in `offset_frequency` mode (fallback `offset_feather`) with SDR before/after published in the TextureSet.** Do not turn tiling into a diffusion project. | pbr-texture-core | 5 | 5 | 2 | **12.5** | The whole "is it seamless" question becomes one number with a pass threshold of 1.5. Deterministic, ms-scale. |
| **6** | **Deploy MOSS-Image at 512² first**, int8 DiT + w4a8 TE + small VAE (12.11 GiB on disk), `--reserve-vram 2`, and **measure** s/step + peak VRAM. 512² is the intended texture resolution (a 500 mm tile at 1024² is already 0.49 mm/texel), not a compromise. | moss-deploy | 5 | 4 | 2 | **10** | Closes the last guessing in the 8 GB question. Expect ~0.4–0.5 s/it `[inferred]`; per the prior art's measured O(n²) model, 1024² costs ~4.7× more per step than 512². |
| **7** | **Texture prompt grammar + the "is it lit?" gate** in `prompts.py`: flat/orthographic/shadowless skeleton, texture-specific negatives, and a plane-fit-to-log-luminance check that rejects lit samples before derivation. | pbr-texture-core | 4 | 4 | 2 | **8** | The one defect no derivation can undo. Cheap to enforce, and it is the difference between an albedo and a photo. |
| **8** | **Spectrum-MOSS caching** (custom node, checkpoint-agnostic, `max_consecutive_forecasts=1` for `er_sde`, SLG 0.000–1.000) — measure the real multiplier on gfx1200. | moss-deploy | 4 | 3 | 2 | **6** | Claimed "2–2.5×" at full quality is the best quality/effort ratio in the menu. Confidence drops because it is third-party, unaudited, and caching artefacts would land exactly in the micro-detail our normal maps amplify. |
| **9** | **4-step LoRA experiment** — `Kijai` rank-72 LoRA (312 MiB) + stock `KSampler(steps=4, cfg=1.0, euler, simple)`. Only 312 MiB to find out. | moss-deploy | 3 | 3 | 2 | **4.5** | The akhaliq graph proves stock nodes suffice, but **CFG 1 kills the negative prompt**, so all texture exclusions must move into the positive. A/B on the same seed before adopting. |
| **10** | **Colour+depth-control LoRA** for texture-from-geometry (`moss_image_color_depth_control` 298.6 MiB + MoGe-3), official graph verified to load `moss_image_sft_int8_convrot`. | moss-deploy → pbr-texture-core | 4 | 3 | 3 | **4** | The only MOSS feature that speaks to *our* problem (projective material placement, relighting consistency). Phase 2 because it stacks 5.04 + 0.30 GiB plus a depth estimator into 8 GB. |

---

## B. Do not do (and why) — put this in the plan so nobody proposes it later

| Item | Reason |
|---|---|
| **Switching production to `moss-infer-hip` / `lucebox` / `mossimage.cpp`** | `moss-infer-hip` is validated on gfx1200 and its streaming penalty is only 5–8 % — genuinely excellent **as a bit-exact bf16 quality reference**. But it **refuses LoRAs, GGUF, quantised files, img2img, inpaint, control adapters and non-euler samplers**, so it cannot run our texture pipeline (no speed LoRA, no depth control, no edit LoRA). It is a **measuring instrument, not a production path**. `lucebox` needs its own build and offers no LoRA either; `mossimage.cpp`'s CUDA engine is the real product, HIP is txt2img-only. |
| **Upgrading ComfyUI "to get MOSS support"** | Superseded: the tree was upgraded 0.36.0 → 0.45.0 and `moss_image` is in the `CLIPLoader` enum. **Instead** keep the regression check that the Qwen-Image-2.1 `int8_convrot` path still works, because the same upgrade moved `comfy-kitchen` 0.2.35 → 0.2.41, which sits on the int8 kernel path. |
| **8-step DMD LoRA right now** | Its sampler is **data-space DMD** (`x = z + Q·x0`, step codes) — not a `KSampler` loop — so it needs a custom node, and the adapter everybody links (`darask0/MOSS-Image_4-step_adapter_lora`) is **HTTP 401 gated**. Revisit only when a trusted node ships. |
| **Multi-view texturing stacks** (SyncMVD, TEXTure, Text2Tex, MatAtlas, MVPaint, Tiled Diffusion) | They solve cross-view consistency for a *mesh*. We have exact geometry and exact UVs and need one small material sample. All require many renders + per-object optimisation + several GB VRAM. Cite as prior art; do not build. |
| **Single-image intrinsic decomposition in the pipeline** (IntrinsicAnything, Material Palette, MaterialFusion, Intrinsic Image Diffusion) | Every serious method needs **multi-view** input or per-object optimisation, and their own papers say single-view decompositions fail to relight correctly. Adding a GPU/torch dependency to get *plausible* maps that our deterministic derivation already produces is negative value. Revisit only if we ever capture real photographs of real materials. |
| **Deriving metallic** | Measured reality: dielectrics read **exactly 0.000** and bare iron **0.989** in real CC0 assets. Any derived map with `0.1 < mean < 0.9` is a bug. Metallic is **authored** from the class table. |
| **16-bit height by default** | At 500 mm/1024² the 8-bit step is 3.9 µm ⇒ **0.46°** of normal tilt per level at 1 mm of relief; you only exceed a 1° floor above **~2.2 mm** of relief (`H_max ≈ 4.45·tile_mm/N`). If you need more relief, **tile smaller** rather than widen the format — glTF and most consumers will quantise back to 8-bit anyway. |
| **"tileable texture" LoRAs from SD1.5/SDXL/FLUX** | Keys and semantics don't transfer to a from-scratch 5B single-stream DiT with a Qwen3.5-9B encoder. No MOSS tiling LoRA exists as of today. |
| **Judging albedo/roughness under AgX or Filmic** | They desaturate and compress. Use `Standard` or `Khronos PBR Neutral` (verified present in 5.3). AgX is for beauty renders only. |
| **`--enable-dynamic-vram`, `--high-ram`, `HSA_OVERRIDE_GFX_VERSION`, `expandable_segments`** | Already proven dead/negative on this exact box by the prior art; `--enable-dynamic-vram` segfaults, `--reserve-vram 1` can take the desktop down. |

---

## C. The three numbers that should govern design (all measured or derived here)

1. **Streaming a model that doesn't fit costs 5–8 %**, not 2×. gfx1200, RX 9060 XT, bf16, 1024²: 4.24 s/step resident →
   4.43 at 17/42 blocks → 4.57 at 2/42, **bit-identical output**. ⇒ *8 GB is a throughput question, not a feasibility
   question.*
2. **Seam quality is measurable and thresholdable**: `SDR ≤ 1.5` pass. A photograph scores 3–20, mirror-blend ≈ 1.0,
   offset+repaired stochastic texture ≈ 1.2. Anyone arguing about seams without this number is arguing about taste.
3. **Texel density is `R·f` texels across the object** — the object's physical size cancels. So: *4K render at 85 %
   fill ⇒ ≥ 3264 texels ⇒ 4096 texture, whatever the part size*; and `px/mm = R·f/S`. 3D-print preview: 0.5–1 px/mm.

---

## D. What we still don't know (and the experiment that settles each)

| Unknown | Experiment | Cost |
|---|---|---|
| Real ComfyUI int8 s/step and peak VRAM at 512²/768²/1024² on this box | `moss-deploy` benchmark table, 3 seeds, `execution_cached` asserted empty | one session |
| Does Spectrum-MOSS survive contact with `er_sde` + SLG on ROCm, and what is the true multiplier? | A/B 25 steps with/without, same seed; report s/step **and** high-frequency energy of the derived normal | ~1 h |
| Does 4-step/CFG-1 destroy the flat-swatch quality we depend on? | Same seed, 50-step vs 4-step, then run the SDR + tilt-distribution + variance gates on both | ~1 h |
| Is the depth-control LoRA usable inside 8 GB? | MoGe depth on CPU, then 512² generation with DiT+LoRA resident; watch for `Not enough memory for command submission` | ~2 h |
| Does the `comfy-kitchen 0.2.41` bump break Qwen-Image-2.1 int8? | Lead already ordered this regression run — treat as **blocking** for anyone reusing the old pipeline | minutes |

---

## E. Where the evidence for every number above lives

This page is a synthesis; each claim carries its full citation in one of the five reference documents (all
`docs/research/`, compiled 2026-11-13):

| Claim in this page | Source document / section | Kind of evidence |
|---|---|---|
| Streaming penalty 5–8 %, 4.24/4.43/4.57 s/step, bit-identical streaming | `MOSS_IMAGE.md` §4.1 | **[M]** — `mossimage.cpp` `docs/hip.md`, validated on gfx1200 (RX 9060 XT) |
| File sizes, int8/w4a8 ladder, 12.11 GiB minimum set | `MOSS_IMAGE.md` §1.1 | **[V]** — HF API `?blobs=true`, 2026-11-13 |
| Sampling defaults (50/CFG 3/shift 5/layer 22 vs 25/2.5/er_sde) | `MOSS_IMAGE.md` §1, §2 | **[C]** + **[V]** (parsed the template JSON) |
| Speed-lever numbers (3.48 s / 3.33 s / 2.12 s / 2–2.5×) | `MOSS_IMAGE.md` §5 | **[M]** third-party 16 GB bench + repo claims — **not ours** |
| `scene.cycles.bake_type` trap, flat normal `(0.502,0.502,1.000)`, HIP device setup, GLB ORM round-trip | `BPY_HEADLESS.md` §2, §6, §7 | **[V]** — executed on Blender 5.3.0 here today |
| ORM = R/G/B, ARM ≡ ORM, Unity reorders+inverts | `PBR_REFERENCE_VALUES.md` §6 | **[SPEC]** glTF 2.0 + my 10-asset ARM measurement |
| Measured roughness ranges (concrete 0.62–0.79, leather 0.66 sd/mean 0.23, iron metallic 0.989…) | `PBR_REFERENCE_VALUES.md` §2 | **[V]** — my Poly Haven `arm` statistics |
| Gray card 0.18 linear; metals metalness = 1 | `PBR_REFERENCE_VALUES.md` §1, §4.1 | **[M]** — physicallybased.info v2 (CC0) |
| `H_max ≈ 4.45·tile_mm/N`; 8-bit = 0.46° tilt per level at 1 mm relief | `PBR_FROM_IMAGE.md` §2.2 | **[I]** — arithmetic stated in full, re-runnable |
| Texel density `R·f` (size cancels); px/mm table | `PBR_REFERENCE_VALUES.md` §7 | **[I]** — derivation shown, not folklore |
| Ill-posedness of single-image material estimation | `PBR_FROM_IMAGE.md` §1 | **[M]** — arXiv 2404.11593, 2409.15273, 2312.12274, 2508.19789; Material Palette CVPR 2024 |
| Tiling techniques, circular-padding refs, TexTile, MVD prior art | `TILEABLE_TEXTURES.md` §2, §6 | **[M]/[C]** — diffusers #556, sd-webui #911, arXiv/CVPR ids |
| SDR thresholds (≤1.5 pass, 3–20 visible, mirror ≈1.0) | `TILEABLE_TEXTURES.md` §3 | **[I]** — my formulation; mirror-blend and CC0 assets are the calibration fixtures |

Citation tags: **[V]** verified on this machine today · **[M]** published measurement with numbers · **[C]** primary
doc claim · **[SPEC]** specification · **[I]** my arithmetic/formulation, assumption stated · **[U]** unverified.

`MOSS_IMAGE.md` §4.2's per-step estimates for ComfyUI int8 remain **[I]** and must be replaced by `moss-deploy`'s
measured table before any of it appears in a plan or a promise. Both datasets used are **CC0**
(`physicallybased.info` v2, Poly Haven `arm` maps) so they can be vendored or re-measured at any time.
