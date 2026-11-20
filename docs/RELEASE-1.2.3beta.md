# atta-1.2.3beta

Atta 1.2.3beta is the next public beta of the 1.2 line. The Python package is
`1.2.3b1`; the DSH, ZCode, and Antigravity plugin packages are `1.2.3-beta.1`.
Plugin identity and the twelve `atta_*` tool names are unchanged, so this is an
in-place upgrade.

## What is new

### Geometry coverage milestone (v1.2.3-geometry)

- **46 parametric recipes** — 46 closed, single-component manifold recipes across
  11 of 12 shape categories (every targeted category ships at least 3 recipes),
  all passing `atta_verify --level standard` across all three presets
  (`safe`, `nominal`, `hero` — 139 cases in `tests/test_preset_matrix.py`) with
  parametric `expected_genus` checks.
- **56+ feature ops** — adds generic and grammar operations (`shape_grammar`,
  `lsystem`, `hull`, `minkowski`, `polyhedron`, `heightmap`, `deform`, `morph`,
  `projection_extrude`) alongside primitives, curves/surfaces, exact CAD,
  implicit fields (`gyroid`/`sdf`/`sdf_blend`/`field_tree`), text, and bio
  morphology.
- **Corpus coverage** — `692` total records (`680` accepted, `639` substantive,
  `41` reference-only, `304` families), with `12/12` corpus categories meeting
  their substantive targets (`docs/SHAPE_COVERAGE.md`).

### Text-to-CAD upgrades (v1.2.3beta P0 + P1)

All additions are clean-room MIT Python/NumPy/Manifold3D/OCCT/ezdxf
implementations with zero Non-Commercial / Tier-C code or data:

- **3D metrology & self-intersection (`atta.metrology`, `atta.verify`)** —
  adds `self_intersection` (`SIR`: triangle-triangle broadphase + Möller/SAT
  exact check), `dangling_edge_length_mm` (`DangEL`), and
  `flux_enclosure_error` (`FluxEE`) to `metrology.measure(...)` and `verify`
  receipts. Fixes stale `"use atta_solve"` hints in `verify.py` to
  `"use atta_variant"`.
- **3D mesh comparison (`atta.probe`, `atta.perception`, `atta.verify`)** —
  adds `probe.compare_meshes_3d` (`chamfer_mm`, `chamfer_a_to_b_mm`,
  `chamfer_b_to_a_mm`, `hausdorff_p95_mm`, `hausdorff_max_mm`,
  `volumetric_iou`), wires `metrics_3d` into `perception.diff(..., compare_3d=True)`,
  and adds `constraints.mesh_agreement` against a reference `target_model_id`.
- **2D engineering drawings (`atta.drawing`, `atta.engine.deliver`)** — adds
  `project_drawing`, `export_dxf` (layered `VISIBLE`, `HIDDEN`, `CENTER`,
  `ANNOTATION` via `ezdxf`), and `export_svg` (3-view + isometric sheet with
  title block and overall dimensions). Uses OCCT `HLRBRep_Algo` when a B-rep
  shape is available and falls back cleanly to pure-numpy crease/silhouette
  `mesh_hlr` with honest `backend` metadata. Exposed via
  `atta_deliver(formats=["dxf", "svg_drawing"])`.
- **Declarative 2D sketches & KCL-style face tags (`atta.curves`, `atta.features`, `atta.brep`)** —
  adds `curves.eval_sketch_segments` for typed 2D `segments` (`line`, `arc`,
  `tangent_arc`, `circle`, optional `tag`) on `extrude` / `revolve` (arcs
  compile to true circular edges in B-rep/STEP), plus optional face `tags` on
  primitives/`extrude`/`revolve` and topological edge selectors
  (`common_edge_of`, `on_face`, `from_feature`) in `brep.select_edges`.
- **Locate-then-infill graph patching & preflight (`atta.spec`, `atta.engine`, `atta.agent.tools`)** —
  adds `spec.patch_spec` (node-level edits for `recipe`, `graph` features/params,
  and `assembly` parts), `spec.preflight_spec`, `atta_spec(op="patch"|"preflight")`,
  `atta_model(patches=...)` for graphs and assemblies, and surfaces dropped
  unknown tool keys from `_clean_args` into `warnings` on the tool envelope.
- **Stability rule & L0–L3 prompt bench (`atta.design`, `scripts/run_bench.py`)** —
  adds `design.stability(mesh)` (center-of-mass XY projection vs `z_min` support
  convex hull) and `corpus/bench/generalization_v3_prompts.json` (`run_bench.py --prompts`)
  scoring `atta_conjure` and `atta_spec(propose)` across L0–L3 prompt
  abstraction tiers against executable geometric predicates.
- **Antigravity host integration (`atta.agent.install`, `plugins/antigravity/`)** —
  adds `plugins/antigravity/plugin.json` and `python3 -m atta.agent.install antigravity`
  (`install_antigravity`) to write/sync `plugin.json`, `mcp_config.json`,
  `assets/logo.svg`, and `skills/atta/` while preserving existing `_atta/`
  workspace state.

## Compatibility

- **12 `atta_*` geometry tools** (`atta_doctor`, `atta_catalog`, `atta_spec`,
  `atta_model`, `atta_variant`, `atta_describe`, `atta_verify`, `atta_preview`,
  `atta_deliver`, `atta_run`, `atta_conjure`, `atta_design`); look tooling stays
  in the separate `basid` package.
- **46 recipes, 56+ feature ops, 53 materials.**
- Public specs and parameters are millimetres and degrees; kernels and Blender
  scenes use metres.
- Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `trimesh`, `manifold3d`,
  `pillow` (plus `ezdxf` for DXF export and `build123d` as the optional
  `[brep]` exact-CAD / OCCT HLR extra).

## Upgrade

- Python: `pip install -e ".[all]"` (or `.[brep]` for exact STEP and OCCT HLR).
- DSH: install the repository root URL or `link:<checkout>/plugin`, then restart.
- ZCode: refresh the marketplace (`plugins/`), update `atta`, then restart.
- Antigravity: `python3 -m atta.agent.install antigravity --create`, then reload plugins.
