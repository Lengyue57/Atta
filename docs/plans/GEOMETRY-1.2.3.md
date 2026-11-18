# v1.2.3beta — geometry coverage push (swarm plan)

Integration branch: `v1.2.3-geometry` (cut from `main@c209875`).
Goal: take the "geometry coverage" dimension from 8.2/10 to ≥9.5/10, closing
every known gap and **raising generalization honestly**.

## Baseline (measured 2026-11-18)

- 23 recipes, 47 feature ops, 12 tools. Coverage gate + families: 55 tests pass;
  kernel CI subset: 465 pass.
- Corpus (substantive = accepted A/B licence AND has params): short in
  architecture 20/40, organic 13/30, text 9/20, lattice 13/20, tolerance 17/20.
- `corpus/bench/generalization_v1.json`: build 100 %, **faithful 80 %**,
  verify 94 %. 14 misses = missing grammar ops (`rule`, `startshape`,
  `cf::symmetry`); 6 misses = implicit topology (gyroid sheet, Schwarz P,
  gyroid cube, porous torus, porous coral, Mandelbulb → genus / components).
- Other missing ops in the bench vocabulary: hull, minkowski, polyhedron,
  projection, smooth, morph, surface(heightmap), deform, subdiv, remesh,
  libfive_tree, offset.
- Known gaps (docs/SHAPE_COVERAGE.md): genus is a per-recipe constant;
  `lattice.gyroid_slab` has no unit guard (volume drifts 18k–26k mm³);
  boundary presets not in CI; categories with only 1 recipe.

## Targets (definition of done)

| metric | target |
|---|---|
| corpus substantive per short category | architecture ≥ 48, organic ≥ 36, text ≥ 25, lattice ≥ 25, tolerance ≥ 25 (all 12 categories ≥ target) |
| recipes | ≥ 45 total; every category with `target_families ≥ 1` has ≥ 3 recipes |
| recipe robustness | every recipe passes its category gate at **safe, nominal and hero** presets (new CI test) |
| parametric genus | `expected_genus` may be a function of params; flange / honeycomb / gyroid stay correct when counts change |
| bench v1 | faithful ≥ 97 %, verify ≥ 99 %, `ops_missing` empty for rule/startshape/cf::symmetry/hull/offset/libfive_tree |
| bench v2 (new, held-out) | ≥ 150 cases frozen **before** template work; faithful ≥ 90 %, verify ≥ 95 % |

## Ownership (one owner per file — never edit a file you do not own)

| member | model | owns |
|---|---|---|
| harvest-arch | Gemini 4 Hydrogen | `corpus/normalized/{architecture,organic,text}.jsonl` (append) |
| harvest-lat | Gemini 4 Hydrogen | `corpus/normalized/{lattice,tolerance,pattern,surface,primitive}.jsonl` (append), `scripts/coverage_report.py`, `corpus/coverage_report.json`, `corpus/license_report.json` |
| grammar | coordinator model | `atta/families/grammar_ops.py`, `tests/test_grammar_ops.py`, `docs/GRAMMAR.md` |
| genops | Gemini 4 Hydrogen | `atta/families/generic_ops.py`, `tests/test_generic_ops.py`, `docs/GENERIC_OPS.md` |
| implicit | coordinator model | `atta/sdfkit.py`, `atta/fieldtree.py`, `atta/families/{implicit,lattice,field_ops}.py`, `atta/registry.py`, genus-only edits in `atta/families/{mech,pattern}.py`, `tests/test_implicit_topology.py`, `tests/test_param_genus.py` |
| recipes-arch | Gemini 4 Hydrogen | new `atta/families/{architecture_ext,organic_ext,primitive_ext,pattern_ext}.py`, `tests/test_recipes_ext_a.py` |
| recipes-lat | Gemini 4 Hydrogen | new `atta/families/{lattice_ext,text_ext,tolerance_ext,assembly_ext}.py`, `tests/test_recipes_ext_b.py` |
| bench | coordinator model | `scripts/run_bench.py`, `corpus/bench/*`, `tests/test_preset_matrix.py`, `docs/SHAPE_COVERAGE.md`, `docs/ROADMAP.md` |
| verify-kernel | coordinator model | read-only reviewer of grammar / genops / implicit |
| verify-content | coordinator model | read-only reviewer of harvest-* / recipes-* / bench |

## Rules for every author

1. You work in your own `share` workspace. First `git checkout -b geo/<your-name> v1.2.3-geometry`.
   Commit only your own files on that branch. Never push to a remote.
2. Family modules are loaded with `ATTA_FAMILIES_STRICT=1` in tests: a module that
   fails to import breaks the whole suite. Run
   `python3 -m pytest -o addopts='' -q tests/test_families.py tests/test_coverage_gate.py`
   plus your own tests before posting.
3. New graph ops register through `atta/families/*` (see `families/nurbs_ops.py`,
   `families/field_ops.py` for the pattern) — never edit `atta/features.py` or `atta/engine.py`.
4. Public units are mm/deg; kernel meshes are metres. Every op/recipe must return a
   closed manifold (or fail with an `AttaError` carrying `code`, `path`, `fix`).
   Hard budgets (recursion depth, triangle counts) are mandatory — fail closed.
5. **Honesty**: never weaken a check, edit an expected value, or special-case a
   bench case id to make a number go up. If a reference record's declared
   topology is wrong, prove it (analytic argument or independent tool) and
   record it with `topology_unverified` plus a reason — do not silently change it.
6. Corpus records must validate against `docs/shape_corpus.schema.json`
   (`python3 scripts/validate_corpus.py corpus/normalized/`), carry a real source
   URL pinned to a commit/permalink, a verified licence (read the LICENSE file), and
   real parameters. Tier A = permissive (MIT/BSD/Apache/CC0/CC-BY/MPL/0BSD);
   tier B = share-alike/copyleft (CC-BY-SA, LGPL, GPL) and **must not embed code**
   (`code` omitted). NC/ND/unknown = tier C, not accepted. No duplicates of existing ids/URLs.
7. When done: post the branch name, head commit, test output and a one-paragraph
   summary to your verifier (`@verify-kernel` or `@verify-content`). Tag
   `@coordinator` only when your verifier has approved, or on a blocker.
