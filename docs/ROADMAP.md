# Roadmap — from foundation to the "monster"

v1.2 is the first public release: a working Blender-first kernel with a small
agent surface and hard receipts. The phases below record what is done and what
is explicitly not, so a reader can tell capability from aspiration.

## Phase 1 — foundation (done)

| Dimension | State | Evidence |
|---|---|---|
| Geometry | strong | 46 closed recipes (all pass `verify` at `standard` across `safe`/`nominal`/`hero`); 56 feature ops (incl. `shape_grammar`, `lsystem`, `hull`, `minkowski`, `polyhedron`, `heightmap`, `deform`, `morph`, `projection_extrude`, composable `gyroid`/`sdf`/`field_tree`, `extrude(holes_xy)`); boolean via manifold3d |
| Verification | strong | three-state receipt; topology, wall, dihedral, curvature, constraints; coverage-matrix gate + full preset-matrix CI + parametric `expected_genus` |
| Engineering | medium→strong | bbox/volume/wall/clearance/interference; assembly interference/clearance; no tolerance stack yet |
| Interoperability | strong (subset) | analytic STEP via build123d/OCCT for box/cylinder/sphere/cone/torus/revolve/extrude/boolean/transform/mirror/pattern; mesh-only ops fail with a fix, never a fake STEP |
| Agent UX | strong | 10 verbs; MCP/DSH/HTTP; `fix` on every failure; solver and assembly reuse the same verbs |
| Spatial sense | strong | text/ASCII perception pack (views/sections/traits/trace/diff), ray grids, clearance, inside/nearest, assembly placements/DOF |
| Parameter solving | strong | LHS + compass + Nelder-Mead + differential evolution + LM/SLSQP polish; targets, sensitivity, closest-achievable, Pareto; assembly-level `part.param` |
| Shape range | broad | 46 recipes across 11 of the 12 categories (≥3 per targeted category); 12/12 corpus categories above target; v1 100%, v2 98.1%, blind v2_fresh 89.7% verify |
| Assembly | strong | parts, ports, 13 mate types, SE(3) solve, Jacobian-rank DOF, AABB broadphase, boolean interference, per-part export |
| Performance | strong for draft | numpy draft: all 46 recipes build in 6.5 s total; `surface.fillet_network` is the outlier at ~3.8 s (SDF-resampled `offset`), everything else ≤0.35 s; resident Blender; render cache |
| Visual | strong | per-part PBR materials, studio/catalog/dark/clay presets, focus macro, normal/depth/id/curvature/deviation passes, Zone/CIELAB/colourfulness/key-fill image analysis |

### Shape coverage (v1.2.3)

Twelve categories from [`coverage_matrix.json`](coverage_matrix.json). "corpus" is
**substantive** records — accepted (`A`/`B` licences) *and* carrying real parameters,
which is the implementation-grade bar used by the corpus reports;
reference-only accepted records are not counted toward a target. The family
column counts shipped recipes; the full
acceptance evidence (verify verdict + genus + param/constraint counts + before/after
bench tables) is in [`SHAPE_COVERAGE.md`](SHAPE_COVERAGE.md).

<!-- SHAPE-COVERAGE-TABLE:START -->

benches (generic Spec v3 feature graph — shipped recipes never called; op coverage 100.0 % across all three):
- `generalization_v1` (n=100): **faithful 100.0%**, verify 100.0%, build 100.0% (v1.2 was 80.0% / 94.0% / 100.0%)
- `generalization_v2` (n=160, frozen `432086f` pre-template): **faithful 98.1%**, verify 98.1%, build 100.0% (baseline was 83.8% / 74.4% / 96.2%)
- `generalization_v2_fresh` (n=116, frozen `8c6f906`, blind single-shot): **faithful 90.5%**, verify 89.7%, build 100.0%

| category | recipes shipped | corpus (substantive / target) | bench verify pass (v1 / v2 / v2_fresh) |
|---|---|---|---|
| primitive | 4 — `chamfer_box`, `rounded_cylinder`, `tube`, `wedge` | 70 / 30 | 8/8 · 13/13 · — |
| mechanical | 4 — `flange`, `gear.involute`, `gear`, `bracket` | 92 / 60 | 16/16 · 47/47 · — |
| product | 6 — `enclosure`, `bottle`, `bowl`, `mug`, `screw_cap`, `vase` | 41 / 40 | 11/11 · 17/17 · — |
| architecture | 5 — `arch_vault`, `dome_oculus`, `fluted_column`, `stair`, `tower.growth` | 50 / 40 | 11/11 · 7/7 · 29/29 |
| organic | 4 — `coral`, `phyllotaxis`, `seashell`, `tree` | 37 / 30 | 8/8 · 3/3 · 23/24 |
| pattern | 3 — `honeycomb_panel`, `knurl`, `louvre_panel` | 56 / 40 | 11/11 · 4/4 · 6/10 |
| assembly | 4 — `bolt_nut`, `clevis_pin`, `dovetail_joint`, `hinge_pin` | 85 / 30 | 8/8 · 9/9 · — |
| text | 4 — `braille_plate`, `engraved_plate`, `keychain_tag`, `stencil_sign` | 27 / 20 | 3/3 · 2/2 · 16/16 |
| lattice | 4 — `graded_gyroid`, `gyroid_slab`, `strut_cell`, `tpms_block` | 29 / 20 | 5/5 · 5/6 · 16/16 |
| surface | 4 — `coons_duct`, `dome`, `fillet_network`, `panel` | 105 / 30 | 10/10 · 42/44 · 9/10 |
| tolerance | 4 — `clearance_ladder`, `fit_pair_gauge`, `hinge_gauge`, `snap_fit` | 28 / 20 | 6/6 · 4/4 · 5/11 |
| material_finish | 0 (matrix: target_families 0) | 19 / 10 | 3/3 · 4/4 · — |

<!-- SHAPE-COVERAGE-TABLE:END -->

All **12/12** corpus categories now clear their substantive targets (`639`
substantive / `680` accepted / `692` total records across `304` families), and
all **11/11** targeted recipe categories ship at least 3 recipes passing
`safe`/`nominal`/`hero` in CI (`tests/test_preset_matrix.py`). `material_finish`
deliberately has no family (`target_families: 0`). The remaining misses on the
held-out benches (`3/160` on `v2`, `12/116` on blind `v2_fresh`) are documented
in [`SHAPE_COVERAGE.md`](SHAPE_COVERAGE.md) — chiefly exact through-hole counts,
multi-component tolerance gauges, and open-vs-closed sweep loops that cannot be
inferred from an op list alone.

### Perception layer (done)

`atta_describe(perceive=true)` / `python3 -m atta perceive` returns a text and
ASCII viewfinder in 0.1–0.3 s without Blender: low-resolution silhouettes,
Z-sections, shape traits, construction trace and silhouette diff. This removes
the render round-trip from the shape-iteration loop. `verify` remains the gate.

## Phase 2 — assembly + solver (done)

1. **Assembly layer (done).**
   `parts[]` with transforms, declared/auto ports, 13 mate types, an SE(3)
   least-squares solve, Jacobian-rank DOF (under/over/redundant/conflicts),
   AABB sweep-and-prune, exact boolean interference, sampled clearance, an
   assembly receipt, and a named-part GLB scene plus per-part OBJ/STL.
   `atta_variant(base=<assembly>)` also solves part parameters against
   composite targets (`part.param` variables, `dof`, `collisions`,
   `min_clearance_mm`, bbox/volume).

2. **Parameter solving (done).**
   `atta_variant` now accepts `targets`, `solve`, `free`, `fixed`, `initial`,
   `pareto`, `max_evaluations` and `tolerance`. Strategies: LHS, compass,
   Nelder-Mead, differential evolution, local least-squares, Pareto. Reports
   feasibility, per-target violation, closest achievable, sensitivity and
   "move this variable" recommendations with side effects. Every evaluation is
   a real draft build; constraints use verify semantics.

3. **Feature depth (partly done).**
   `extrude(holes_xy=...)` cuts real through-holes; `gyroid` and a general
   `sdf` op (gyroid / Schwarz P / Neovius / Mandelbulb / primitive fields) are
   composable implicit ops; `offset` and `text_engrave`/`text_emboss` are
   registered. Still missing: kernel fillet/chamfer, ribs/bosses, shell with
   openings, and exact reference genus on the remaining TPMS/fractal cases.

## Phase 3 — precision and scale

4. **Diagnostic preview (done).**
   Normal / depth / object-id / curvature / deviation passes, per-part
   materials, studio/catalog/dark/clay lighting, `focus=<part>` macro framing,
   and a mathematical image analysis layer (Zone system, CIELAB/OKLab,
   clipping, contrast, entropy, colourfulness, edge density, key/fill). The
   analysis warns; it never returns a geometry verdict.

5. **B-rep backend behind the same spec (done for a subset).**
   A graph whose ops are box/cylinder/sphere/cone/torus/revolve/extrude/
   boolean/transform/mirror/pattern routes to build123d/OCCT and delivers an
   analytic STEP. Mesh-only ops fail with `brep_unsupported` and a fix; the
   verdict still comes from the mesh verify receipt. The agent surface is
   unchanged.

6. **Performance.**
   Geometry cache by geometry hash, parallel variants in worker processes,
   optional GPU render, and a warm worker pool. Keep the fast loop under 100 ms
   for typical recipes.

## Phase 4 — product polish

7. **Materials and finish receipts.** Thumbnails, material previews, and a
   finish diff between draft and final.
8. **Project workspaces.** Per-project `ATTA_HOME`, model listing, cleanup, and
   an export bundle with spec + receipt + manifest.
9. **Host matrix.** Conformance tests that run the same 10-tool script against
   MCP, DSH plugin, HTTP and a generic OpenAI manifest.

## Non-goals

* Photoreal portfolio rendering.
* GUI/click automation.
* A separate "creative/aggressive" mode that fabricates measurements.
* Exposing every low-level op as a tool.
