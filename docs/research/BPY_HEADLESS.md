# Headless bpy cookbook — Blender 5.3.0 on this machine

Everything below is **verified by execution on this machine** (Blender 5.3.0, build 2026-11-10, snap
`/snap/blender/7998/5.3`) on **2026-11-13**, unless a line is tagged otherwise. Where 5.x broke something that worked
in 4.x, it is called out with a **⚠ 5.x** marker. This document exists to stop `blender-material-bridge` from
rediscovering the same six traps.

Tags: **[V]** executed and observed here · **[C]** from Blender/Cycles source shipped with this install · **[U]** unverified.

---

## 0. Invocation and environment

```bash
blender --version                 # → Blender 5.3.0, build 2026-11-10  [V]
blender -b --factory-startup --python worker.py -- --job job.json
```

- `--factory-startup` gives a clean state **but does not register the `cycles` add-on** **[V]**. `bpy.context.preferences.addons['cycles']` raises `KeyError` until you do:
  ```python
  import addon_utils
  addon_utils.enable("cycles", default_set=False, persistent=False)
  ```
- Under `-b` (background), **both Cycles and EEVEE render** **[V]**. EEVEE headless works on this box through the
  snap's EGL path — no X server needed (measured 64²@8spp = 0.23 s).
- A harmless line appears on every Cycles run: `WARNING CUEW initialization failed: Error opening the library` —
  that is the CUDA-probe, not an error. Ignore it.
- Exit code: `bpy.ops.*` returning `{'FINISHED'}` is **not** proof of success. Check the artefact (file exists,
  pixels non-uniform, GLB parses). See §9.

---

## 1. ⚠ 5.x breaking changes vs 4.x — the list that will bite our code

| # | Thing | 4.x | **5.3 (verified)** | Impact on us |
|---|---|---|---|---|
| 1 | **EEVEE engine id** | `'BLENDER_EEVEE_NEXT'` (4.2–4.5) | **`'BLENDER_EEVEE'`** **[V]** | Setting the 4.x string raises an enum error. Our presets must branch on `bpy.app.version`. |
| 2 | **Principled sockets** | `Emission`, `Specular`, `Transmission`, `Clearcoat`, `Subsurface`, `Sheen` | **`Emission Color`**, **`Specular IOR Level`**, `Transmission Weight`, `Coat Weight`, `Subsurface Weight`, `Sheen Weight` **[V]** | Any 4.x-era socket name raises `KeyError`. Full list in §3. |
| 3 | **New sockets in 5.x** | — | `Thin Wall`, `Thin Film Thickness`, `Thin Film IOR`, `Transmission Dispersion Scale`, `Transmission Dispersion Abbe Number`, `Diffuse Roughness` **[V]** | Harmless, but do not assume "unknown socket = typo". |
| 4 | **`scene.render.bake.type`** | full bake-type enum | **only `NORMALS`, `DISPLACEMENT`, `VECTOR_DISPLACEMENT`** **[V]**. Full enum moved to **`scene.cycles.bake_type`** **[V]**. | **This is the biggest trap in 5.x.** See §6. |
| 5 | `Material.use_nodes` | fine | works, emits `DeprecationWarning: expected to be removed in Blender 6.0` **[V]** | Keep using it; expect a 6.x change. |
| 6 | Auto-smooth | `use_auto_smooth` (≤4.0), modifier in 4.1 | `bpy.ops.object.shade_smooth_by_angle(...)` **[V]**, `bpy.ops.object.shade_auto_smooth` **[V]** | Use the operators. |
| 7 | glTF export | — | new: `export_image_quality`, `export_compressed_images_*`, `export_use_ktx_compression`, `export_ktx_zstd_level`, `export_unused_images`, `export_unused_textures`, `export_original_specular` **[V]** | Additive; old flags still work. |
| 8 | View transforms | AgX/Filmic/Standard | + **`Khronos PBR Neutral`** **[V]** | Use it for asset review (§5). |
| 9 | `image.pack()` on generated images | works | works, `packed_file is not None` → `True` **[V]** | Needed for GLB embedding. |

---

## 2. Cycles + HIP on gfx1200

```python
import bpy, addon_utils
addon_utils.enable("cycles", default_set=False, persistent=False)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'HIP'         # 'NONE' → CPU only
prefs.refresh_devices()                   # ⚠ REQUIRED. Before this, prefs.devices == []  [V]
for d in prefs.devices:
    d.use = (d.type == 'HIP')             # or (d.type == 'CPU')
sc.cycles.device = 'GPU'                  # ⚠ 'GPU', not 'HIP'. 'HIP' is not a valid value.
```

Verified device table on this box **[V]**:

```
('AMD Ryzen 5 5600X 6-Core Processor', 'CPU', False)
('AMD Radeon RX 9060 XT',               'HIP', True)
```

- `compute_device_type` accepts `NONE, HIP, CUDA, OPTIX, ONEAPI` **[V]**.
- **`prefs.devices` is empty until `refresh_devices()`** — the classic failure is: set `cycles.device='GPU'`, forget
  refresh, and silently render on CPU while the flag says GPU. **Verify by timing, not by reading the flag.**
- Denoiser: only **`OPENIMAGEDENOISE`** accepted here; `OPTIX` / `OPENIMAGEDENOISE_GPU` rejected **[V]**.
  `sc.cycles.use_denoising = True`, `sc.cycles.denoiser = 'OPENIMAGEDENOISE'`, `sc.cycles.denoising_use_gpu` exists.
- Measured HIP speed on this box (256², 64 spp, sphere + noise, no denoise) **[V]**: CPU **0.48–0.57 s**, HIP
  **0.17–0.30 s** → **≈2–3×**. That is the honest expectation for small tiles; it does not scale to "HIP is 10×".

---

## 3. Building the image-texture graph

Node inventory verified present with these exact socket names **[V]**:

| `nodes.new(...)` | inputs | outputs |
|---|---|---|
| `ShaderNodeTexImage` | `Vector` | `Color`, `Alpha` |
| `ShaderNodeNormalMap` | `Strength`, `Color` | `Normal` |
| `ShaderNodeMapping` | `Vector`, `Location`, `Rotation`, `Scale` | `Vector` |
| `ShaderNodeUVMap` | — | `UV` (set `.uv_map = mesh.uv_layers[0].name`) |
| `ShaderNodeSeparateColor` | `Color` | `Red`, `Green`, `Blue` |
| `ShaderNodeCombineColor` | `Red`, `Green`, `Blue` | `Color` |
| `ShaderNodeTexCoord` | — | `Generated`, `Normal`, `UV`, `Object`, `Camera`, `Window`, `Reflection` |
| `ShaderNodeBump` | `Strength`, `Distance`, `Filter Width`, `Height`, `Normal` | `Normal` |
| `ShaderNodeAmbientOcclusion` | `Color`, `Distance`, `Normal` | `Color`, `AO` |
| `ShaderNodeMixRGB` | `Factor`, `Color1`, `Color2` | `Color` |
| `ShaderNodeTexNoise` | `Vector`, `W`, `Scale`, `Detail`, `Roughness`, … | `Factor`, `Color` |

**Principled BSDF sockets, full list, 5.3.0 [V]** (in order):
`Base Color, Metallic, Roughness, IOR, Alpha, Thin Wall, Normal, Weight, Diffuse Roughness, Subsurface Weight,
Subsurface Radius, Subsurface Scale, Subsurface IOR, Subsurface Anisotropy, Specular IOR Level, Specular Tint,
Anisotropic, Anisotropic Rotation, Tangent, Transmission Weight, Transmission Dispersion Scale,
Transmission Dispersion Abbe Number, Coat Weight, Coat Roughness, Coat IOR, Coat Tint, Coat Normal, Sheen Weight,
Sheen Roughness, Sheen Tint, Emission Color, Emission Strength, Thin Film Thickness, Thin Film IOR`

4.x → 5.x name map (use this to migrate any old code):

```python
SOCKET_5X = {
    "Emission":       "Emission Color",
    "Specular":       "Specular IOR Level",
    "Transmission":   "Transmission Weight",
    "Clearcoat":      "Coat Weight",
    "Clearcoat Roughness": "Coat Roughness",
    "Subsurface":     "Subsurface Weight",
    "Sheen":          "Sheen Weight",
}
def sock(node, name):
    return node.inputs[SOCKET_5X.get(name, name)]      # raises KeyError if truly absent
```

### 3.1 A complete, verified textured-material builder

```python
def build_textured_material(mat, tex, *, uv_map=None, scale=(1, 1, 1),
                            normal_strength=1.0, tint=None):
    """tex: dict with 'albedo' (sRGB image), 'orm' (Non-Color, R=AO G=Rough B=Metal),
    optional 'normal' (Non-Color tangent-space), optional 'height'."""
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (600, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled');  bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    # --- UV + physical tiling -------------------------------------------------
    vec = None
    if uv_map:
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = uv_map; uvn.location = (-1100, -300)
        vec = uvn.outputs['UV']
    if scale != (1, 1, 1):
        mp = nt.nodes.new('ShaderNodeMapping'); mp.location = (-900, -300)
        mp.inputs['Scale'].default_value = scale
        if vec is not None:
            nt.links.new(vec, mp.inputs['Vector'])
        vec = mp.outputs['Vector']

    # --- albedo (sRGB) -------------------------------------------------------
    alb = nt.nodes.new('ShaderNodeTexImage'); alb.image = tex['albedo']
    alb.image.colorspace_settings.name = 'sRGB'          # explicit, never assume
    alb.extension = 'REPEAT'                             # 'MIRROR' for problem tiles
    alb.location = (-600, 300)
    if vec is not None: nt.links.new(vec, alb.inputs['Vector'])
    if tint is not None:
        mix = nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 1.0
        mix.inputs['Color2'].default_value = tint
        nt.links.new(alb.outputs['Color'], mix.inputs['Color1'])
        nt.links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        nt.links.new(alb.outputs['Color'], bsdf.inputs['Base Color'])

    # --- ORM: R=AO, G=Roughness, B=Metallic ---------------------------------
    if tex.get('orm'):
        orm = nt.nodes.new('ShaderNodeTexImage'); orm.image = tex['orm']
        orm.image.colorspace_settings.name = 'Non-Color'  # ⚠ MUST be Non-Color
        orm.extension = 'REPEAT'; orm.location = (-600, 0)
        if vec is not None: nt.links.new(vec, orm.inputs['Vector'])
        sep = nt.nodes.new('ShaderNodeSeparateColor'); sep.location = (-300, 0)
        nt.links.new(orm.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'],  bsdf.inputs['Metallic'])
        # AO is only consumed by the glTF exporter via the 'glTF Material Output'
        # node group (see §7.3); Cycles has no AO socket on Principled.
    elif tex.get('roughness'):
        r = nt.nodes.new('ShaderNodeTexImage'); r.image = tex['roughness']
        r.image.colorspace_settings.name = 'Non-Color'
        nt.links.new(r.outputs['Color'], bsdf.inputs['Roughness'])

    # --- normal --------------------------------------------------------------
    if tex.get('normal'):
        n = nt.nodes.new('ShaderNodeTexImage'); n.image = tex['normal']
        n.image.colorspace_settings.name = 'Non-Color'    # ⚠ MUST be Non-Color
        n.location = (-600, -300)
        if vec is not None: nt.links.new(vec, n.inputs['Vector'])
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -300)
        nm.inputs['Strength'].default_value = normal_strength
        nm.inputs['Color'] if False else None
        nt.links.new(n.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    elif tex.get('height'):
        h = nt.nodes.new('ShaderNodeTexImage'); h.image = tex['height']
        h.image.colorspace_settings.name = 'Non-Color'
        bump = nt.nodes.new('ShaderNodeBump'); bump.location = (-300, -300)
        bump.inputs['Strength'].default_value = 0.5
        nt.links.new(h.outputs['Color'], bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    return nt
```

**Colour-space rules (the #1 silent defect):**

| Map | Colour space | Why |
|---|---|---|
| albedo / base color | `sRGB` | perceptual encoding |
| **normal, roughness, metallic, AO, height, mask** | **`Non-Color`** | they are data; sRGB decode wrecks them |
| emissive | `sRGB` (usually) | it is colour |
| displacement | `Non-Color` | data |

Verified colour-space names include `sRGB`, `Non-Color`, `Linear Rec.709`, `AgX Base sRGB`, `Working Space` **[V]**.
Set it on the **image datablock**, not the node — there is no per-node colours`pace.

---

## 4. UV generation headless

### 4.1 `smart_project` — the only parameters that matter

Verified signature + defaults on 5.3 **[V]**:

```python
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit=1.1519173383712769,   # 66° IN RADIANS  ← the one knob that matters
    island_margin=0.02,               # default 0.0; raise to 0.01–0.03 when baking
    area_weight=0.0,                  # default 0.0
    correct_aspect=True,              # default True
    scale_to_bounds=False,            # default False
)
bpy.ops.object.mode_set(mode='OBJECT')
```

| Param | Verified default | What it actually does | Guidance |
|---|---|---|---|
| `angle_limit` | `1.1519` rad (**66°**) | angle above which a new chart is cut | **It is radians.** Passing `66` explodes into hundreds of islands. 60–66° is sane; lower = more islands. |
| `island_margin` | `0.0` | gap between islands in UV space | **For baking set 0.01–0.03**, otherwise the bake margin bleeds across islands |
| `margin_method` | `'SCALED'` | how margin is applied | leave |
| `rotate_method` | `'AXIS_ALIGNED_Y'` | island orientation | leave |
| `area_weight` | `0.0` | 0 = dimensionless (equal UV area), 1 = texel-density-homogeneous | set **1.0** when the mesh has mixed part sizes and you care about density |
| `scale_to_bounds` | `False` | normalises into 0–1 | `True` for a single-object bake |

**Why `smart_project` fails on CAD meshes:** it cuts at *every* angle above `angle_limit`, so a boolean-heavy part with
many 15–45° chamfers shatters into dozens-to-hundreds of islands. Result: unusable texel density, huge bake margins,
visible seams on flat faces, and `pack_islands` spending its budget on slivers. Measured on a plain 24-face cube it is
instant (0.02 s **[V]**), so speed is never the problem — **fragmentation is**.

**What to do instead, by geometry class:**

| Geometry | Strategy |
|---|---|
| Prismatic CAD part (boxy, planar faces) | `bpy.ops.uv.cube_project(cube_size=..., correct_aspect=False)` — box/triplanar; 6 islands max, uniform density, no cuts across faces. Best default for Atta parts. |
| Revolved part (lathe/cylinder) | `bpy.ops.uv.cylinder_project(...)` — one island, seam on the back |
| Organic / sculpted | `smart_project` then `bpy.ops.uv.pack_islands(...)` |
| Already-seamed mesh | `bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=...)` — best quality, needs seams (`bpy.ops.uv.mark_seam`) |
| Triplanar *without* UVs at all | do it in the shader: `ShaderNodeTexCoord.Object` → 3 × `ShaderNodeMapping` → 3 images → blend by `ShaderNodeSeparateXYZ` weights. No UVs, no bake, but **cannot be exported to glTF** (glTF has no triplanar) — render-only. |

Other UV ops verified present **[V]**: `align, arrange_islands, average_islands_scale, cube_project, cylinder_project,
export_layout, follow_active_quads, lightmap_pack, mark_seam, minimize_stretch, pack_islands,
project_from_view, sphere_project, smart_project, stitch, unwrap, weld`.
Mesh-side: `Mesh.calc_loop_triangles()`, `Mesh.loop_triangles`, `Mesh.uv_layers` **[V]**.

### 4.2 Computing texel density (do this, don't guess)

```python
def texel_density_px_per_m(obj, uv_layer_name=None, image_px=2048):
    """Median px per metre over the mesh's faces. Needs the object evaluated
    (modifiers applied) if you want the shipped geometry."""
    me = obj.data
    uvl = me.uv_layers[uv_layer_name] if uv_layer_name else me.uv_layers.active
    me.calc_loop_triangles()
    dens = []
    for tri in me.loop_triangles:
        # world-space area
        a3 = tri.area                                    # object space
        area3 = a3 * (obj.matrix_world.to_scale().x * obj.matrix_world.to_scale().y)
        # uv-space area
        uvs = [uvl.data[li].uv for li in tri.loops]
        area2 = abs((uvs[1].x-uvs[0].x)*(uvs[2].y-uvs[0].y)
                  - (uvs[2].x-uvs[0].x)*(uvs[1].y-uvs[0].y)) * 0.5
        if area3 > 1e-12:
            dens.append((area2 * image_px * image_px) / area3)   # px²/m²
    dens.sort()
    return (dens[len(dens)//2]) ** 0.5 if dens else 0.0          # px/m
```
Report `median` and `p10/p90` — a **spread > 3× between p10 and p90 means the UVs are non-uniform** and no texture will
look right across the part. This is the check that catches bad `smart_project` output automatically.

---

## 5. Colour management for material/product review

Verified enumerations **[V]**:

`scene.view_settings.view_transform` accepts:
`Standard`, `Filmic`, `Filmic Log`, `AgX`, **`Khronos PBR Neutral`**, `Raw`, `False Color`

`scene.view_settings.look` (with `view_transform='AgX'`) accepts:
`None`, `AgX - Punchy`, `AgX - Base Contrast`, `AgX - Medium Low Contrast`, `AgX - Medium High Contrast`,
`AgX - Very High Contrast`, `AgX - Very Low Contrast`

**Set `view_transform` before `look`** — the `look` enum is validated against the current transform, so assigning an
AgX look while `Standard` is active raises.

Recommended presets:

| Purpose | `view_transform` | `look` | Why |
|---|---|---|---|
| Judge albedo / roughness / PBR correctness | **`Standard`** | `None` | no tone compression, no hue skew — what a texture viewer shows |
| glTF / asset-store delivery check | **`Khronos PBR Neutral`** | `None` | the glTF-sample-viewer transform; preserves saturation and hue, rolls off only highlights. **This is the right "does it ship" check.** |
| Product hero render | `AgX` | `AgX - Punchy` or `- Medium High Contrast` | filmic, forgiving, pretty |
| Diagnose clipping | `False Color` | `None` | exposure/zebra |

**Do not judge albedo or roughness under AgX/Filmic.** They desaturate and compress; you will "fix" correct textures
until they are wrong, then they will look wrong everywhere else.

---

## 6. Cycles baking — the 5.x trap, then everything else

### 6.1 ⚠ The 5.x trap (verified, with the exact error)

In Blender **5.3**, `scene.render.bake.type` has **only three** valid values:

```python
b = bpy.context.scene.render.bake
b.type = 'COMBINED'
# ValueError: bpy_struct: item.attr = val: enum "COMBINED" not found in
#             ('NORMALS', 'DISPLACEMENT', 'VECTOR_DISPLACEMENT')
```
**[V]** — reproduced exactly. Those three belong to the multires `object.bake_image` path.

The **real** bake-type knob moved to the **Cycles scene property** **[V]**:

```python
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.bake_type = 'NORMAL'
# full enum [V]: COMBINED, AO, SHADOW, POSITION, NORMAL, UV, ROUGHNESS, EMIT,
#                ENVIRONMENT, DIFFUSE, GLOSSY, TRANSMISSION   (default 'COMBINED')
```

Corroborated in the Cycles UI source shipped with this install (`scripts/addons_core/cycles/ui.py`) **[C]**:

```python
layout.operator("object.bake", icon='RENDER_STILL').type = cscene.bake_type
layout.prop(cscene, "bake_type")
```

and the operator itself carries the authoritative enum **[V]**:

```python
bpy.ops.object.bake.get_rna_type()  # type=COMBINED default, plus:
# COMBINED, AO, SHADOW, POSITION, NORMAL, UV, ROUGHNESS, EMIT,
# ENVIRONMENT, DIFFUSE, GLOSSY, TRANSMISSION
```

**Rule: set `sc.cycles.bake_type` for UI parity, and always pass `type=` explicitly to the operator.**

### 6.2 Settings (all verified present)

| Property | Where | Valid values / default **[V]** |
|---|---|---|
| `margin` | `sc.render.bake` | `16` default; **use 16–32** |
| `margin_type` | `sc.render.bake` | `EXTEND` (default), `ADJACENT_FACES` |
| `target` | `sc.render.bake` | `IMAGE_TEXTURES` (default), `VERTEX_COLORS` |
| `normal_space` | `sc.render.bake` | `TANGENT` (default), `OBJECT` |
| `normal_r/g/b` | `sc.render.bake` | `POS_X/NEG_X/POS_Y/NEG_Y/POS_Z/NEG_Z` |
| `use_clear` | `sc.render.bake` | `False` |
| `use_selected_to_active` | `sc.render.bake` | `False` |
| `cage_extrusion`, `cage_object`, `use_cage` | `sc.render.bake` | `0.0`, `""`, `False` |
| `max_ray_distance` | `sc.render.bake` | `0.0` |
| `view_from` | `sc.render.bake` | `ABOVE_SURFACE` (default), `ACTIVE_CAMERA` |
| `save_mode` | `sc.render.bake` | `INTERNAL`, `EXTERNAL` |
| `uv_layer` | operator | `""` = active |
| `use_pass_color/direct/indirect/emit/diffuse/glossy/transmission` | `sc.render.bake` | bools |

### 6.3 The four recipes we need

```python
def bake(sc, img, *, bake_type, passes=None, margin=16, uv_layer=""):
    sc.render.engine = 'CYCLES'
    sc.cycles.bake_type = bake_type
    sc.render.bake.margin = margin
    sc.render.bake.margin_type = 'EXTEND'
    sc.render.bake.use_clear = True
    if passes:
        for k, v in passes.items():
            setattr(sc.render.bake, k, v)
    # target image node must be selected + active in EVERY material of the object
    r = bpy.ops.object.bake(type=bake_type, margin=margin, use_clear=True,
                            target='IMAGE_TEXTURES', uv_layer=uv_layer)
    return r
```

| Map | Call | Notes |
|---|---|---|
| **Albedo** | `bake_type='DIFFUSE'`, `use_pass_direct=False`, `use_pass_indirect=False`, `use_pass_color=True`, target image **sRGB** | This is the correct "bake albedo": pure base colour, zero lighting. **[V]** produced `0.89` on a `0.8` base colour, i.e. correct sRGB round-trip of the default Principled value. |
| **Normal (tangent)** | `bake_type='NORMAL'`, `normal_space='TANGENT'`, target image **Non-Color** | For DirectX/Unreal/Unity: `sc.render.bake.normal_g = 'NEG_Y'` before baking, or flip G after. **[V]** a flat cube face baked to exactly `(0.502, 0.502, 1.000)` = bytes (128,128,255). Use that as your unit test. |
| **Roughness** | `bake_type='ROUGHNESS'`, Non-Color | Bakes the *socket value*; if Roughness is driven by a texture you get the texture. Good for "bake my procedural roughness into a map for export". |
| **AO** | `bake_type='AO'`, Non-Color | Ray-traced AO, independent of the material. Set `view_from='ABOVE_SURFACE'` and a sane `sc.cycles.samples`. |

Measured on this box (CPU, 256², cube) **[V]**: DIFFUSE **0.29 s**, NORMAL **0.02 s**. Bake is cheap for simple
geometry; cost scales with texels × samples × geometry complexity, not with the model size.

### 6.4 Why bakes come out black — diagnosis order

1. **No selected+active image-texture node.** For **every material** on the object you must have an `Image Texture`
   node whose `.image` is the bake target, with `node.select = True` and `nt.nodes.active = node`. This is the cause
   ~70 % of the time. A helper:
   ```python
   def set_bake_target(obj, img):
       for slot in obj.material_slots:
           mat = slot.material
           if not (mat and mat.use_nodes): continue
           nt = mat.node_tree
           n = next((x for x in nt.nodes if x.type == 'TEX_IMAGE' and x.image == img), None)
           if n is None:
               n = nt.nodes.new('ShaderNodeTexImage'); n.image = img
           for x in nt.nodes: x.select = False
           n.select = True; nt.nodes.active = n      # ⚠ BOTH lines
   ```
2. **Colour space wrong.** Normal/AO/roughness into an `sRGB` image → washed-out garbage that reads as "too bright",
   not as an error. Set `Non-Color` **before** baking.
3. **Nothing was saved or packed.** Bake writes into the image datablock in RAM. Set `img.file_format='PNG'` and either
   `img.filepath_raw=...; img.save()` or `img.pack()`. Otherwise all you have is a `.blend`-less memory ghost.
4. **`margin` too small / `ADJACENT_FACES`.** Islands bleed or leave black gutters. Use `margin≥16`, `EXTEND`.
5. **`use_selected_to_active=True` without cage or `cage_extrusion`** → rays miss the low-poly cage → black.
6. **No UV layer**, or degenerate/zero-area UVs → garbage. Check `me.uv_layers` is non-empty *and* the UV island area
   is non-zero.
7. **Engine isn't Cycles.** `bpy.ops.object.bake` requires `sc.render.engine='CYCLES'` (EEVEE cannot bake).
8. **Emission-only material baked as DIFFUSE** → black, because DIFFUSE reads the diffuse closure. Use `EMIT`, or fix
   the material.

### 6.5 CPU vs HIP bake

- Bake honours the selected device, so **HIP bake works** once §2 is done. **No published, reliable speed-up number for
  bake on this GPU** — treat bake as MOQ-bound and measure per job; the tile sizes are small.
- Known behavioural difference **[U]**: HIP/OIDN paths can produce slightly different ray-hit noise at low sample counts,
  and OIDN on GPU vs CPU differs. For bakes that must be **bit-reproducible** (regression tests), pin
  `sc.cycles.device='CPU'` and a fixed seed + sample count.
- Prior art on this box: any GPU path that over-subscribes VRAM can take the **whole desktop down**. For long bakes keep
  `--reserve-vram`-equivalent headroom by rendering in small tiles, and never run bake at 4K on an 8 GB card with a
  browser open.

---

## 7. glTF export — flags, embedding, and verified round-trip

### 7.1 The call

```python
bpy.ops.export_scene.gltf(
    filepath="out.glb",
    export_format='GLB',
    export_image_format='AUTO',      # AUTO/JPEG/WEBP/NONE. ⚠ there is NO 'PNG'
    export_materials='EXPORT',       # EXPORT/PLACEHOLDER/VIEWPORT/NONE
    export_texcoords=True,
    export_normals=True,
    export_apply=False,              # True = apply modifiers into the mesh
    use_selection=True,
    export_keep_originals=False,     # True = ship original files, no re-encode
    export_unused_images=False,
    export_unused_textures=False,
    export_image_quality=90,         # 75 default
    export_tangents=False,           # True only if you need hand-written tangents
    export_original_specular=False,
)
```

`export_format` accepts `GLB` / `GLTF_SEPARATE` (the static enum reads empty because it is populated at runtime) **[V]**.
`export_image_format='AUTO'` is what you want: it keeps **PNG** for images that need alpha or are not JPEG, and there is
**no `'PNG'` option** — do not hunt for one. Full flag list is in `bpy.ops.export_scene.gltf.get_rna_type().properties` **[V]**;
5.x additions include `export_compressed_images_type ∈ {WEBP, KTX2}`, `export_use_ktx_compression`,
`export_ktx_zstd_level`, `export_texture_dir`, `export_image_quality`, `export_keep_originals`,
`export_unused_images`, `export_unused_textures`, `export_original_specular`.

### 7.2 Embedding: pack first, then verify by parsing

```python
for img in (albedo_img, orm_img, normal_img):
    img.pack()                       # works on generated images  [V]
    assert img.packed_file is not None
```

Verification (this is the part people skip, and it is 12 lines) **[V]**:

```python
import struct, json
def glb_audit(path):
    d = open(path, 'rb').read()
    magic, ver, total = struct.unpack_from('<III', d, 0)
    assert magic == 0x46546C67 and ver == 2, "not a GLB"
    clen, ctype = struct.unpack_from('<II', d, 12)
    assert ctype == 0x4E4F534A, "first chunk must be JSON"
    j = json.loads(d[20:20+clen])
    imgs = j.get('images', [])
    assert imgs, "no images — textures were not embedded"
    for im in imgs:
        assert 'bufferView' in im, f"image {im.get('name')} is an external URI, not embedded"
        assert im.get('mimeType', '').startswith('image/')
    for m in j.get('materials', []):
        pbr = m.get('pbrMetallicRoughness', {})
        if 'metallicRoughnessTexture' in pbr:
            t = pbr['metallicRoughnessTexture']['index']
            # G must come from a packed ORM, not a grey roughness image
            assert j['textures'][t]['source'] is not None
    return j
```

A real exported cube with a packed ORM produced **[V]**:

```json
"images":   [{"bufferView": 4, "mimeType": "image/png", "name": "orm"}],
"textures": [{"sampler": 0, "source": 0}],
"materials":[{"doubleSided": true, "name": "M",
              "occlusionTexture": {"index": 0},
              "pbrMetallicRoughness": {"baseColorFactor": [0.8,0.8,0.8,1],
                                       "metallicRoughnessTexture": {"index": 0}}}]
```

Note **one image, index 0, used for BOTH** `occlusionTexture` and `metallicRoughnessTexture` — exactly the glTF 2.0
shared-image allowance, and proof that the ORM pack survives. Also note **`doubleSided: true` is Blender's default** —
set `mat.use_backface_culling = True` if you want single-sided.

### 7.3 Getting occlusion into the GLB

Cycles' Principled has **no AO socket**, so an AO map is invisible to the exporter by default. Blender's glTF exporter
looks for a node group named **`glTF Material Output`** and reads its **`Occlusion`** input. Verified working **[V]**:

```python
ng = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
ng.interface.new_socket("Occlusion", in_out='INPUT', socket_type='NodeSocketFloat')
g = nt.nodes.new('ShaderNodeGroup'); g.node_tree = ng
nt.links.new(orm_sep.outputs['Red'], g.inputs['Occlusion'])   # R = AO
```
→ exporter emitted `occlusionTexture: {"index": 0}` **[V]**. Without this group you get baseColor +
metallicRoughness but **no occlusion**, and AO silently disappears from the asset.

### 7.4 OBJ / MTL sidecar

```python
bpy.ops.wm.obj_export(filepath="out.obj",
    export_selected_objects=True, export_uv=True, export_normals=True,
    export_materials=True, path_mode='COPY', apply_modifiers=True,
    export_triangulated_mesh=False, forward_axis='NEGATIVE_Z', up_axis='Y')
```
Verified defaults/flags **[V]**: `export_materials=True`, `path_mode='AUTO'` (`COPY` copies the PNGs next to the .obj —
use it), `export_uv`, `export_normals`, `apply_modifiers=True`, `export_triangulated_mesh=False`,
`forward_axis='NEGATIVE_Z'`, `up_axis='Y'`, `filter_glob='*.obj;*.mtl'`, plus `export_pbr_extensions`,
`export_smooth_groups`, `export_object_groups`. **MTL cannot express ORM packing or normal maps portably** — expect
albedo+specular only; if a consumer needs PBR, give them the GLB.

---

## 8. Render presets (verified knobs)

| Preset | Engine | Key settings | Use |
|---|---|---|---|
| `cycles_hip_final` | `CYCLES`, device GPU/HIP | `sc.cycles.samples=256`, `use_denoising=True`, `denoiser='OPENIMAGEDENOISE'`, `use_adaptive_sampling=True`, `adaptive_threshold≈0.01` | final textured renders |
| `cycles_hip_fast` | same | `samples=64`, adaptive, denoise | iteration |
| `cycles_cpu_repro` | `CYCLES`, device CPU | fixed `samples`, `use_denoising=False` (or CPU OIDN), fixed seed | **regression tests / bit-reproducibility** |
| `eevee_material` | **`'BLENDER_EEVEE'`** (⚠ not `_NEXT`) | `sc.eevee.taa_render_samples=64`, `use_raytracing=True`, `use_shadows=True` | fast material previews, works headless here (0.23 s for 64²) |
| `workbench_depth` | `BLENDER_WORKBENCH` | — | cheap **depth/geometry pass** for the depth-control LoRA (`MOSS_IMAGE.md` §6.4) |

EEVEE knobs verified present **[V]**: `taa_render_samples`, `taa_samples`, `use_raytracing`, `ray_tracing_method`,
`ray_tracing_options`, `use_shadows`, `shadow_ray_count`, `use_fast_gi`, `fast_gi_method`, `volumetric_*`,
`use_overscan`, `overscan_size`, `clamp_surface_direct/indirect`.

Get the **depth pass** for geometry-guided texturing:
```python
sc.view_layers[0].use_pass_z = True
sc.use_nodes = True
# ...then read the Z pass from the render result / Compositor File Output
```

---

## 9. Worker design notes (what actually keeps this robust)

1. **Never `bpy.ops.wm.read_factory_settings()` inside a running worker** unless you re-init everything; prefer one
   process per job. Blender leaks state between jobs (materials, images, orphan data) and library overrides are a
   classic source of "works alone, fails in a batch".
2. **Purge orphans between jobs**: `bpy.ops.outliner.orphans_purge(do_recursive=True)`.
3. **`{'FINISHED'}` ≠ success.** Gate on artefacts: file size > 0, GLB parses (§7.2), baked image has
   `max(pixels) > min(pixels)` (a constant image means the bake did nothing).
4. **Determinism**: set `sc.cycles.seed`, `sc.cycles.use_animated_seed = False`, pin device, pin samples. Denoising
   (especially OIDN-on-GPU) can break bit-equality — disable it in repro runs.
5. **Timeouts**: headless Blender can hang on a GPU fault. The prior art's VRAM watchdog pattern (`kill` the worker
   before the driver dies) is the right model; a hung HIP context does not recover.
6. **File paths**: keep `models/`, `/tmp` scratch and the venv on ext4; the NTFS data disk is fine for large read-only
   inputs but not for the MIOpen/Triton cache or anything needing exec bits.

---

## 10. Unverified / open items

- **[U]** `object.bake` on HIP: works in principle (device flag is honoured) but I did **not** run a GPU bake; no
  published gfx1200 bake benchmark exists. Measure before promising a bake preset.
- **[U]** `export_use_gltfpack*` and KTX2 flags exist in this build but need `gltfpack`/`toktx`; the Draco and
  MeshOptimizer libs **are** bundled (`libbf_intern_draco_bridge.so`, `libbf_intern_meshopt_bridge.so` were both
  reported available) **[V]**.
- **[U]** EEVEE headless depends on the snap's EGL; it worked here, but a machine without a GPU-renderable display
  server may differ.
- **[U]** `ShaderNodeMix` has **multi-type sockets with duplicated names** (`Factor` ×2, `A` ×4, `B` ×4, `Result` ×4) —
  index by position/type, not by name, or you will silently link the float socket. (`ShaderNodeMixRGB` is unambiguous;
  prefer it for colour-only work.)

---

## 11. Sources

- Blender 5.3 manual, colour management & image colour spaces: https://docs.blender.org/manual/en/5.3/render/color_management.html
- Cycles bake settings & `bpy.ops.object.bake`: https://docs.blender.org/manual/en/5.3/render/cycles/baking.html
- `smart_project` / UV operator reference: https://docs.blender.org/manual/en/5.3/modeling/meshes/uv/editing.html
- glTF 2.0 spec (image metadata, `occlusionTexture` R, `metallicRoughnessTexture` G/B, shared images): https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Blender glTF 2.0 exporter (flags, `glTF Material Output` node group, `export_image_format`): https://docs.blender.org/manual/en/5.3/addons/import_export/scene_gltf2.html
- Khronos PBR Neutral view transform: https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral
- **[C]** local source: `/snap/blender/7998/5.3/scripts/addons_core/cycles/ui.py` (bake panel uses `cscene.bake_type`)
- **[V]** all executed measurements: Blender 5.3.0 build 2026-11-10, this machine, 2026-11-13 — smoke bake, HIP vs CPU timing, GLB audit, glTF ORM round-trip, socket/enum dumps.
