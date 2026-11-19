---
name: atta-modeling
description: Use Atta 1.2.3beta through its twelve MCP tools for parametric 3D modeling, exact CAD, sampled NURBS, bounded fields, multi-part assemblies, bio morphology, 2D DXF/SVG drawings, measured inspection, verification and GLB/OBJ/STL/STEP delivery. Trigger when the user asks to create, edit, solve, assemble, inspect, validate or export a 3D model, or mentions Atta, CAD parts, mates, DOF, STEP, DXF, or "make it printable". For camera, texture or look work use the separate basid program.
---

# Atta 1.2.3beta — modeling with receipts

This is the **ZCode routing stub**. The authoritative procedure lives in one place:

> **`plugin/skills/atta/SKILL.md`** — read that for the decision table, the
> iteration loop, the aesthetic/manufacturing rules and the error table.
> Reference detail: `plugin/skills/atta/references/`.

Do not maintain a second copy of the procedure here. This file only covers what
is specific to the ZCode/MCP host.

## The surface

Twelve geometry MCP tools, named `mcp__atta__<verb>`:

`atta_doctor` · `atta_catalog` · `atta_spec` · `atta_model` · `atta_variant`
`atta_describe` · `atta_verify` · `atta_preview` · `atta_deliver` · `atta_run`
`atta_conjure` · `atta_design`

Same verbs, arguments and envelopes as the DSH plugin — both are shells over the
one Python engine. `atta_conjure` invents a plausible build from a semantic label
when no real dimensions exist; `atta_design` checks whether the form reads and can
be manufactured (verify only judges whether it is a valid solid).

## The rule that matters

Never call a model done without an `atta_verify` receipt whose `verdict == "pass"`.
`degraded` is not `pass`. Facts come from measurement, not from looking at a render.

If a call returns `{ok:false}`, read `code`, `path` and `fix`, then correct exactly
that thing. `fix` is a patch, not an apology.

## Units

Every public spec / param / constraint / probe is **mm and degrees**. The kernel
and the Blender scene are metres. Never convert by hand — the tools do it at the
boundary. `+Z` is up.

## Fast loop (memorize this shape)

```
catalog -> spec(validate/propose) -> model(draft) -> describe(perceive=true)
   -> model(patches=...) -> verify -> [preview 1 image] -> deliver
```

`describe(perceive=true)` is the sub-second viewfinder (ASCII silhouettes,
sections, shape traits). Use it while the shape is still moving. Do **not** call
`atta_preview` until `verify` passes — it starts Blender and takes tens of seconds,
and a render has no diagnostic value for shape problems.

## Minimum viable correctness

- A recipe exists? Use the recipe, not a hand-written graph.
- `patches` must travel with `recipe=` or `spec=`; `patches` alone is a
  `bad_arguments` error.
- Check `genus` against the recipe's `expected_genus` — a plate with one through
  hole is genus 1, not 0.
- Give `verify` explicit `constraints` for the dimensions the user actually asked
  for. "It looks about right" is not a constraint.

## Quality

Before `atta_deliver`, run the 13-item checklist in
`plugin/skills/atta/references/quality.md` (full version: `docs/QUALITY.md`).
The short version: closed manifold · genus matches · pass verdict · wall ≥ process
minimum · every exposed hard edge beveled · reply carries model_id + verdict +
dimensions + paths.

## Kernel capabilities (46 recipes, 56+ ops, 53 materials)

Use `atta_catalog(feature=...)` before constructing graph operations:
- **Exact CAD & sketches**: typed 2D `segments` (`line`, `arc`, `tangent_arc`, `circle`) on `extrude`/`revolve`, KCL-style face `tags` on primitives/`extrude`/`revolve`, and topological edge selectors (`common_edge_of`, `on_face`, `from_feature`) for `fillet`/`chamfer` with matching OCCT/STEP output.
- **Surfaces, fields & grammars**: `nurbs_surface` for sampled rational patches, `field_tree` for bounded scalar-field composition, `shape_grammar`/`lsystem`/`hull`/`minkowski`/`deform`/`morph`/`projection_extrude` for procedural geometry, and the bio morphology ops (`capsid`, `diatom_frustule`, `pyramidal_neuron`, …).
- **Metrology, 3D diff & graph patching**: `metrology` reports `self_intersection` (SIR), `dangling_edge_length_mm` (DangEL), and `flux_enclosure_error` (FluxEE); `describe(compare_to=...)` adds 3D `chamfer_mm`, `hausdorff_p95_mm`, and `volumetric_iou`; `atta_spec(op="patch"|"preflight")` and `atta_model(patches=...)` support locate-then-infill editing on graphs and assemblies; `atta_design` adds `stability` (COM vs support hull).
- **Assemblies & look**: oriented ports and `attach` initialize placement, while mates/DOF and verification determine constraint satisfaction. Texture, camera and look work belongs to `basid` (`atta.look` or its own MCP tool). Source accuracy warnings survive booleans and assembly. Neither a sampled tolerance nor a closed mesh is an all-domain surface or manufacturing proof.

## Delivery

Mixed assembly STEP contains only supported exact graph parts and reports
mesh-only omissions with delivery status `partial`; GLB can contain every part.
Do not call that STEP a complete assembly. Real backend failures retain their
original error codes.

`atta_deliver(formats=["glb","obj"])`. Also supports `dxf` and `svg_drawing`
for 2D multi-view engineering drawings (OCCT HLR when B-rep applies, clean
mesh HLR fallback otherwise). STEP (`formats=["step"]`) is analytic B-rep
for the supported graph subset; unsupported ops fail with `brep_unsupported`
and a fix — there is never a fake tessellated STEP. The mesh verify receipt
still gates delivery.

## Style

Match the user's language. Report `model_id`, `verdict`, hashes and artifact paths.
Show the smallest useful artifact first (facts, ASCII, one image), then the detail.
Do not narrate the whole tool log.
