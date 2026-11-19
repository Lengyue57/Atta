# Shape coverage

46 recipes across 11 of the 12 shape categories (every category with
`target_families >= 1` ships at least 3 recipes). Every recipe in this table
builds a **closed, single-component manifold** at `quality=draft` across all
three presets (`safe`, `nominal`, `hero`) and passes `atta_verify --level
standard` against its category acceptance gate and parameter-resolved expected
genus.

The numbers below are measured, not asserted. Reproduce the whole table and
preset matrix with:

```bash
python3 -m pytest -o addopts='' tests/test_families.py tests/test_coverage_gate.py tests/test_preset_matrix.py -q
```

## The 46 recipes

`params` is the number of typed parameters the recipe exposes (`catalog
--verbose` prints them with bounds, units and defaults). `genus` is the
**measured** Euler characteristic at `preset=nominal`, which must equal the
recipe's resolved `expected_genus` (`Recipe.genus_for(values)`). Build time is
a draft build on a warm process; the verify column is the verdict at
`standard`.

| recipe | params | genus | triangles | draft build | verify |
|---|---|---|---|---|---|
| `primitive.chamfer_box` | 8 | 0 | 72 | 0.02 s | pass |
| `primitive.rounded_cylinder` | 7 | 0 | 1728 | 0.00 s | pass |
| `primitive.tube` | 8 | 1 | 768 | 0.00 s | pass |
| `primitive.wedge` | 7 | 0 | 28 | 0.00 s | pass |
| `mechanical.flange` | 13 | 7\* | 2328 | 0.04 s | pass |
| `mechanical.gear.involute` | 10 | 1 | 2546 | 0.23 s | pass |
| `mechanism.gear` | 7 | 1 | 1856 | 0.04 s | pass |
| `part.bracket` | 7 | 2 | 612 | 0.01 s | pass |
| `part.enclosure` | 8 | 0 | 44 | 0.00 s | pass |
| `product.bottle` | 8 | 0 | 2684 | 0.06 s | pass |
| `product.bowl` | 11 | 0 | 20544 | 0.03 s | pass |
| `product.mug` | 13 | 1 | 25820 | 0.07 s | pass |
| `product.screw_cap` | 13 | 0 | 12030 | 0.16 s | pass |
| `product.vase` | 12 | 0 | 30912 | 0.04 s | pass |
| `architecture.arch_vault` | 10 | 0 | 990 | 0.05 s | pass |
| `architecture.dome_oculus` | 7 | 1 | 6840 | 0.04 s | pass |
| `architecture.fluted_column` | 12 | 0 | 22708 | 0.16 s | pass |
| `architecture.stair` | 8 | 0 | 1070 | 0.02 s | pass |
| `architecture.tower.growth` | 11 | 0 | 210 | 0.03 s | pass |
| `organic.coral` | 8 | 0 | 5722 | 0.03 s | pass |
| `organic.phyllotaxis` | 8 | 0 | 4228 | 0.03 s | pass |
| `organic.seashell` | 10 | 0 | 7676 | 0.01 s | pass |
| `organic.tree` | 10 | 0 | 3050 | 0.03 s | pass |
| `pattern.honeycomb_panel` | 10 | 35\* | 1088 | 0.02 s | pass |
| `pattern.knurl` | 8 | 0 | 11424 | 0.01 s | pass |
| `pattern.louvre_panel` | 10 | 7 | 348 | 0.01 s | pass |
| `assembly.bolt_nut` | 13 | 1 | 440 | 0.01 s | pass |
| `assembly.clevis_pin` | 10 | 4 | 1132 | 0.04 s | pass |
| `assembly.dovetail_joint` | 10 | 0 | 84 | 0.00 s | pass |
| `assembly.hinge_pin` | 11 | 7 | 1486 | 0.05 s | pass |
| `text.braille_plate` | 13 | 0 | 1284 | 0.01 s | pass |
| `text.engraved_plate` | 19 | 0 | 594 | 0.02 s | pass |
| `text.keychain_tag` | 13 | 1 | 900 | 0.03 s | pass |
| `text.stencil_sign` | 10 | 6 | 1306 | 0.21 s | pass |
| `lattice.graded_gyroid` | 8 | 25 | 217104 | 0.30 s | pass |
| `lattice.gyroid_slab` | 7 | 39\* | 171896 | 0.27 s | pass |
| `lattice.strut_cell` | 8 | 91 | 109684 | 0.23 s | pass |
| `lattice.tpms_block` | 8 | 33 | 205056 | 0.28 s | pass |
| `surface.coons_duct` | 9 | 1 | 8192 | 0.02 s | pass |
| `surface.dome` | 9 | 0 | 3296 | 0.01 s | pass |
| `surface.fillet_network` | 7 | 0 | 16616 | 3.79 s | pass |
| `surface.panel` | 7 | 0 | 3296 | 0.01 s | pass |
| `tolerance.clearance_ladder` | 7 | 0 | 2272 | 0.09 s | pass |
| `tolerance.fit_pair_gauge` | 7 | 0 | 598 | 0.02 s | pass |
| `tolerance.hinge_gauge` | 9 | 2 | 652 | 0.02 s | pass |
| `tolerance.snap_fit` | 10 | 1 | 144 | 0.01 s | pass |

\* Parametric `expected_genus` evaluated from resolved parameters (`mechanical.flange` = `1 + bolt_holes`; `pattern.honeycomb_panel` = `rows * cols`; `lattice.gyroid_slab` = `8*nx*ny*nz - 2*nx - 2*ny + 1` for integer cell blocks).

Total draft-build time for all 46 on one warm process: **6.54 s**.
`surface.fillet_network` remains the outlier at ~3.8 s because it resamples an
SDF offset field; every other recipe is under 0.35 s and 40 of them are under
0.10 s. All **139** cases in `tests/test_preset_matrix.py` (46 recipes ×
`{safe, nominal, hero}` + preset-advertisement check, zero skips) pass.

## Categories

`docs/coverage_matrix.json` defines the 12 categories and what each one must
accept. The gate is executable: `tests/test_coverage_gate.py` (nominal) and
`tests/test_preset_matrix.py` (`safe` / `nominal` / `hero`) read the matrix and
fail any recipe that violates its category acceptance rule (closed,
`min_wall_mm`, genus, single component, positive volume).

| category | recipes shipped | ids |
|---|---|---|
| primitive | 4 | `primitive.chamfer_box`, `primitive.rounded_cylinder`, `primitive.tube`, `primitive.wedge` |
| mechanical | 4 | `mechanical.flange`, `mechanical.gear.involute`, `mechanism.gear`, `part.bracket` |
| product | 6 | `part.enclosure`, `product.bottle`, `product.bowl`, `product.mug`, `product.screw_cap`, `product.vase` |
| surface | 4 | `surface.coons_duct`, `surface.dome`, `surface.fillet_network`, `surface.panel` |
| architecture | 5 | `architecture.arch_vault`, `architecture.dome_oculus`, `architecture.fluted_column`, `architecture.stair`, `architecture.tower.growth` |
| lattice | 4 | `lattice.graded_gyroid`, `lattice.gyroid_slab`, `lattice.strut_cell`, `lattice.tpms_block` |
| pattern | 3 | `pattern.honeycomb_panel`, `pattern.knurl`, `pattern.louvre_panel` |
| organic | 4 | `organic.coral`, `organic.phyllotaxis`, `organic.seashell`, `organic.tree` |
| assembly | 4 | `assembly.bolt_nut`, `assembly.clevis_pin`, `assembly.dovetail_joint`, `assembly.hinge_pin` |
| tolerance | 4 | `tolerance.clearance_ladder`, `tolerance.fit_pair_gauge`, `tolerance.hinge_gauge`, `tolerance.snap_fit` |
| text | 4 | `text.braille_plate`, `text.engraved_plate`, `text.keychain_tag`, `text.stencil_sign` |
| material_finish | 0 | — |

`material_finish` has `target_families: 0` in the matrix — it asks for reference
examples, not a family — so it is not a gap and the gate exempts it.

## The 56 feature ops

`atta/families/*.py` registers feature ops without touching the core. The 56
live ops in `atta.features.FEATURE_OPS` include 9 newly added generic & grammar
ops (`shape_grammar`, `lsystem`, `hull`, `minkowski`, `polyhedron`,
`heightmap`, `deform`, `morph`, `projection_extrude`) alongside the core and
existing family ops:

- **Primitives & curves/surfaces:** `box`, `cylinder`, `sphere`, `cone`,
  `torus`, `revolve`, `loft`, `loft2`, `sweep`, `sweep2`, `pipe`, `coons`,
  `gordon`, `nurbs_surface`, `extrude`, `thicken`, `lobes`, `polyhedron`,
  `heightmap`.
- **Booleans, patterns & transforms:** `union`, `difference`, `intersection`,
  `weld_union`, `mirror`, `pattern_linear`, `pattern_polar`, `transform`,
  `hull`, `minkowski`, `projection_extrude`, `deform`, `morph`.
- **Implicit, offset & mesh ops:** `sdf`, `gyroid`, `field_tree`, `sdf_blend`,
  `offset`, `subdivide`, `repair`, `chamfer`, `fillet`.
- **Grammars, text & bio/form ops:** `shape_grammar`, `lsystem`,
  `text_engrave`, `text_emboss`, `skeleton`, `silhouette`, `branch_tree`,
  `capsid`, `diatom_frustule`, `lattice_shell`, `perforated_cylinder`,
  `perforated_valve`, `pore_field_shell`, `pyramidal_neuron`, `ring_stack`.

`python3 -m atta catalog --verbose` prints each op's `required` / `optional`
arguments, arity and doc string — that output is the authoritative list.

## Held-out generalization benchmarks

Three held-out benchmarks live in `corpus/bench/`. Every case is expressed as a
generic Spec v3 feature graph (`atta.spec`) selected strictly from the record's
declared `operations` / `raw_ops` and parameters — shipped recipes are never
called, and templates never inspect `corpus_id` or `family`.

- **`generalization_v1.json`** (100 cases): original v1.2 held-out benchmark.
- **`generalization_v2.json`** (160 cases, freeze commit `432086f`, sha
  `d3077199f80aca82…`): source-URL- and ID-disjoint from `v1`, stratified over
  all 12 categories, frozen **before** any v1.2.3 template or op change and
  baseline-scored once (`corpus/bench/baselines.json`).
- **`generalization_v2_fresh.json`** (116 cases, freeze commit `8c6f906`, sha
  `3579085cb6dfe2b0…`): all 116 held-out records from the fresh v1.2.3 corpus
  harvest (`architecture`, `organic`, `text`, `lattice`, `tolerance`,
  `pattern`, `surface`), frozen **after** templates were finalized and scored
  **once** blindly with zero post-hoc tuning.

### Before vs After (draft quality, `verify --level standard`)

| bench | n | stage | build | faithful | verify pass | op coverage (native) | missing ops |
|---|---|---|---|---|---|---|---|
| `generalization_v1` | 100 | v1.2 shipped (2026-11-02) | 100.0 % | 80.0 % | 94.0 % | 94.5 % (86.6 %) | `rule` 11, `startshape` 10, `cf::symmetry` 8, `offset` 1, `libfive_tree` 1, `hull` 1 |
| `generalization_v1` | 100 | v1.2.3 baseline (`4a2b6d1`, pre-template) | 97.0 % | 77.0 % | 84.0 % | 94.5 % (86.6 %) | same 6 ops (plus 3 `text_emboss.at` schema fails + 7 `sphere_cone_csg` degraded) |
| **`generalization_v1`** | **100** | **v1.2.3 after ops + implicit** | **100.0 %** | **100.0 %** | **100.0 %** | **100.0 % (93.5 %)** | **none** |
| `generalization_v2` | 160 | v1.2.3 baseline (`432086f`, pre-template) | 96.2 % | 83.8 % | 74.4 % | 97.4 % (86.1 %) | `polyhedron` 4, `minkowski` 4, `projection` 3, `hull` 3, `morph` 2, `smooth` 1 |
| **`generalization_v2`** | **160** | **v1.2.3 after ops + implicit** | **100.0 %** | **98.1 %** | **98.1 %** | **100.0 % (89.3 %)** | **none** |
| **`generalization_v2_fresh`** | **116** | **v1.2.3 blind (frozen `8c6f906`, scored once)** | **100.0 %** | **90.5 %** | **89.7 %** | **100.0 % (98.4 %)** | **none** |

### Why v1 moved from 80 % / 94 % to 100 % / 100 % (and what changed in metadata)

- **14 op-gap misses closed by native ops**: `shape_grammar` / `lsystem`
  (`rule`, `startshape`, `cf::symmetry`), `hull`, `offset`, and `field_tree`
  (`libfive_tree`) eliminated every entry in `ops_missing`.
- **6 implicit-topology misses fixed honestly**: real kernel bugs were fixed in
  `atta/sdfkit.py` and `atta/families/implicit.py` (sphere-clipping past the
  grid box causing `closed=False`; hardcoded `res<=48` creating 465 fake
  pinholes on Schwarz P; `np.gradient` thickness drift; clip-boundary slivers).
  In addition, 9 pre-existing corpus records whose boilerplate `genus: 3,
  components: 1` metadata was mathematically disproven (multi-cell TPMS slabs,
  quotient-genus-9 Neovius, multi-body fractals) were marked
  `topology_unverified: true` with explicit `topology_note` proofs per Rule 5
  (neither `expected` nor `verify` was weakened).

### Remaining failure modes (honest blind-generalization limits)

All 376 cases across `v1`, `v2`, and `v2_fresh` build (`100.0 %`) with zero
missing vocabulary ops (`100.0 %` op coverage). The remaining **3 misses on
`v2`** and **12 misses on `v2_fresh`** expose the real structural ceiling of
translating a record's *operation list* (rather than compiling its full source
program):

1. **Exact hole / loop counts cannot be inferred from an op set** (6 misses on
   `v2_fresh`): e.g. `gh_gumyr_build123d_fast_grid_holes` expects genus `625`
   (a 25×25 hole grid), `gh_cameronbrooks11_linkmail_{square,triangle}_link`
   expect genus `4` / `3`, and `plate_holes` cannot tell from `extrude +
   cylinder + difference` alone whether a part has 0 blind cavities (`lego`,
   `clearance_hole`), 1 bore (`eccentric_bushing`), or 625 through-holes.
2. **Open vs closed sweep / loop topology** (3 misses on `v2`, 2 on
   `v2_fresh`): `curv_sweep_implicit_trefoil`, `manifold_torus_knot_warp`,
   `curv_circlattice_2d`, `gh_gumyr_bd_warehouse_retaining_ring_groove`, and
   `gh_cameronbrooks11_snapfit_pin` declare ops (`sweep`, `revolve`, `box +
   cylinder + difference`) that do not distinguish a closed-loop/bored solid
   (genus 1) from an open/blind one (genus 0).
3. **Multi-component clearance/assembly gauges** (2 misses on `v2_fresh`):
   `gh_cameronbrooks11_linkmail_square_gap` (`components=4`) and
   `gh_enonymous1_openscad_toolkit_tolerance_test` (`components=2`) describe
   multi-body test gauges, whereas single-part graph templates emit `1`
   component.
4. **Generic `sdf` field routing & floor clamps** (2 misses on `v2_fresh`):
   `gh_deadsy_sdfx_loft3d` routed to `sdf_field` (defaulting to a periodic
   gyroid, genus `39` vs `0`), and `dotscad_differential_growth_coral` tripped
   `bbox.z` where `offset_solid` clamped a sub-3 mm height floor.

## Known gaps (v1.2.3)

Closed since v1.2:
- Parametric `expected_genus` (`Recipe.genus_for` / `Resolved.expected_genus`)
  wired on `mechanical.flange`, `pattern.honeycomb_panel`, and
  `lattice.gyroid_slab`.
- `lattice.gyroid_slab` metric wall thickness (`tpms_sheet_distance` + Weyl
  tube volume check) and `unit_mismatch` guard.
- Full preset-envelope CI (`tests/test_preset_matrix.py`: all 46 recipes ×
  `safe`/`nominal`/`hero`, 0 skips).
- All 11 targeted categories expanded to ≥ 3 recipes and all 12 corpus
  categories above substantive targets (`692` valid, `680` accepted, `639`
  substantive).
- Grammar (`shape_grammar`, `lsystem`) and generic (`hull`, `minkowski`,
  `polyhedron`, `heightmap`, `deform`, `morph`, `projection_extrude`) feature
  ops landed.

Still open:
1. **`surface.fillet_network` is slow** (~3.8 s draft from the SDF offset
   resample). Budget for it in batch pipelines.
2. **Bevel vs wall budget on tight junctions.** `verify` evaluates wall
   thickness after beveling; narrow boolean junctions can still measure thin
   locally (caught by `tests/test_preset_matrix.py`, as happened on
   `product.mug@safe` before its handle centerline was fixed).
3. **`text_emboss.at` schema vs runtime arity.** `atta/families/text.py`
   registers `"vectors": ("at",)` (3-vector in `spec.validate_spec`), while the
   runtime handler expects 2D `(x, y)`; omitting `at` works cleanly.
4. **Op-list templates vs full source compilation.** As `v2_fresh` (89.7 %
   verify) shows, op-list templates hit an ~90 % ceiling on unseen corpora when
   exact hole counts, multi-component gauges, or closed-loop sweeps are encoded
   only in control flow.

The corpus *counting* data (`corpus/coverage_report.json`,
`corpus/license_report.json`) is generated reproducibly by
`scripts/coverage_report.py` (`692` records, `680` accepted, `639` substantive,
`304` families, `12/12` categories met, `policy=PASS`).
