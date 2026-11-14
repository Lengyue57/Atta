# atta-1.2.2beta

Atta 1.2.2beta is the second public beta of the 1.2 line. The Python package is
`1.2.2b3`; the DSH and ZCode plugin packages are `1.2.2-beta.3`. Plugin identity
and tool names are unchanged, so this is an in-place upgrade.

## What is new

- **Sampled NURBS** — rational tensor-product patches with positive weights,
  non-uniform knots, analytic first derivatives/normals and bounded conforming
  refinement. `sampled_error` is reported honestly; there is no claimed global
  Hausdorff bound.
- **Exact CAD** — `fillet`/`chamfer` on a supported closed B-rep closure, selected
  geometrically, tessellated with shared OCCT topology, and exported to STEP from
  the same construction.
- **Bounded fields** — `field_tree` composes primitives, CSG, TPMS, noise, twist,
  shell and offset under hard node, sampling, work and triangle budgets, with
  fail-closed numeric handling.
- **Modular assemblies** — oriented ports with roll, deterministic `attach`
  placement, mate solving and DOF, collision/clearance gates, and exact-part STEP
  compounds that name mesh-only omissions instead of fabricating B-rep.
- **Quality provenance** — a missed surface tolerance or parent quality issue
  survives transforms, booleans and assemblies; assembly verification checks the
  final world mesh and per-part quality.
- **Bio morphology** — capsid, diatom, neuron, valve, lattice-shell and ring-stack
  graph ops. These are geometric constructions with the normal facts/verify
  receipts, not biological simulations.
- **Look pipeline moved out** — texture generation, camera planning and textured
  renders now live in the separate MIT-licensed `basid` package. Atta exports the
  GLB and can call it through the optional `atta.look` bridge; the two programs
  never import each other. Atta keeps scalar materials and can still resolve a
  basid-produced `atta.texture/v1` set for preview/deliver.

## Compatibility

- Twelve `atta_*` geometry tools; look tooling is `basid` (separate MCP server).
- 23 recipes, 47 feature ops, 53 materials.
- Public specs and parameters are millimetres and degrees; kernels and Blender
  scenes use metres.
- Python ≥ 3.10. Core dependencies are unchanged: numpy, scipy, trimesh,
  manifold3d and pillow. `build123d` remains the optional exact-CAD extra.

## Limits

No NURBS trim or cross-patch sewing, no global surface-error proof, no
process-wide resource hard limit, and no native CAD assembly hierarchy in STEP.
A passing test suite covers the documented contracts; it is not a manufacturing
or all-scale guarantee.

## Upgrade

- Python: `pip install -e ".[all]"` (or `.[brep]` for exact STEP).
- DSH: install the repository root URL or `link:<checkout>/plugin`, then restart.
- ZCode: refresh the marketplace, update `atta`, then restart or reconnect the host.
