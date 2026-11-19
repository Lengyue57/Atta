# Shape grammars and L-systems (`shape_grammar`, `lsystem`)

`atta/families/grammar_ops.py` registers two zero-input feature-graph ops for
recursive, stochastic and symmetry-driven geometry:

- **`shape_grammar`** — declarative Context Free Art (CFDG v3) and Structure
  Synth (EisenScript) rules in JSON.
- **`lsystem`** — Prusinkiewicz & Lindenmayer (*The Algorithmic Beauty of
  Plants*, ABOP) string rewriting with stochastic and parametric rules, a 3D
  HLU turtle, tropism and optional leaves.

Both ops follow the kernel contract:

- **No `eval`.** `shape_grammar` is pure JSON data; `lsystem` evaluates
  parametric arithmetic and rule conditions through `atta.features._eval_expr`
  with an AST size/exponent guard.
- **Units.** Public lengths (`unit`, `step`, `width`, `min_width`, `leaf.size`)
  are in the spec's model units (`mm` by default); angles are degrees. Unit
  conversion happens only in `prepare`, and the returned `Mesh` is in metres.
- **Determinism.** Weighted rule selection and `{"rand": [lo, hi]}` draws use a
  dedicated `random.Random(seed)` stream. Identical `(spec, seed)` inputs
  produce identical `canonical_hash(mesh)` digests.
- **Closed manifold output.** Emitted primitives are unioned in one pass via
  `manifold3d.Manifold.batch_boolean(..., OpType.Add)` and checked with
  `atta.mesh.topology`. Open or failed booleans raise `AttaError`.
- **Budgets fail closed.** Recursion depth, rule expansions, symbol counts,
  instance counts, triangle counts and coordinate/scale bounds are enforced
  before and after boolean evaluation; exceeding any budget raises an
  `AttaError(code, path, fix)`.

Query the live schemas at any time with:

```bash
python3 -m atta catalog --verbose --json
```

---

## 1. `shape_grammar`

### Arguments

| Argument | Kind | Default | Ceiling | Description |
|---|---|---|---|---|
| `startshape` | string (required) | — | — | Entry rule name in `rules`. |
| `rules` | object (required) | — | — | Map `rule_name -> alternatives` (see below). |
| `unit` | length (`mm`) | `1.0` | — | Size of 1 grammar unit in model units (`> 0`). |
| `symmetry` | string \| object \| list | `null` | `max_instances` | Global symmetry group (`CF::Cyclic`, `CF::Dihedral`, `reflect`). |
| `seed` | int `>= 0` | `0` | `2^63 - 1` | Deterministic PRNG seed for weighted rules and `rand`. |
| `segments` | int `>= 3` | `24` | `128` | Circular tessellation for curved primitives. |
| `tile_thickness` | number `> 0` | `0.1` | `1e6` | Z thickness (in grammar units) of 2D CFDG tiles (`square`, `circle`, `triangle`). |
| `start_transform` | transform | `{}` | — | Initial transform applied before `startshape`. |
| `min_scale` | number `> 0` | `1e-3` | — | Prune any child whose minimum singular value in grammar units is `< min_scale`. |
| `max_depth` | int `>= 0` | `256` | `10,000` | Maximum call depth. |
| `depth_policy` | `"error" \| "prune"` | `"error"` | — | `"error"` fails closed when `depth > max_depth`; `"prune"` stops silently (EisenScript `maxdepth`). |
| `max_instances` | int `>= 1` | `4,096` | `50,000` | Maximum emitted primitives *after* symmetry expansion. |
| `max_expansions` | int `>= 1` | `200,000` | `1,000,000` | Maximum rule evaluations. |
| `max_triangles` | int `>= 1` | `1,000,000` | `5,000,000` | Pre-union and post-union triangle budget. |
| `max_components` | int `>= 1` | `null` | — | Optional upper bound on disconnected components (`1` requires a single solid). |

### Rules, alternatives and Structure Synth `max_depth`

Each entry in `rules` may be written in one of three forms:

1. **Shorthand body list** (single deterministic rule of weight `1.0`):
   ```json
   "STAR": [
     {"shape": "box", "transform": {"s": [4.0, 0.6, 0.6]}},
     {"call": "ARM", "transform": {"x": 2.0}}
   ]
   ```
2. **Single alternative object**:
   `{"weight": 1.0, "max_depth": 5, "then": "CAP", "body": [...]}`
3. **Weighted list of alternatives** (CFDG `rule 0.05 { ... }`):
   ```json
   "BRANCH": [
     {"weight": 1.0,  "body": [{"shape": "box"}, {"call": "BRANCH", "transform": {"y": 0.9, "s": 0.95}}]},
     {"weight": 0.08, "body": [{"call": "BRANCH", "transform": {"r": 35, "s": 0.85}},
                               {"call": "BRANCH", "transform": {"r": -35, "s": 0.85}}]},
     {"weight": 0.04, "body": []}
   ]
   ```

Per-alternative `max_depth` and optional `then` implement Structure Synth's
`rule R maxdepth N > T`: once rule `R` has been expanded `N` times along a call
chain, further calls to `R` either stop or transition to rule `T`.

### Body items and loops

Each item in `body` has:

- **Exactly one** of `"shape"` (a built-in unit primitive) or `"call"` (another
  rule name).
- Optional `"transform"`: an adjustment object or ordered list of adjustment
  objects.
- Optional `"loop"`: `{"count": N, "transform": {...}}` (CFDG `loop N [adj]
  target` / EisenScript `N * { adj } target`). Iteration `i` (`0 .. N-1`)
  applies `(loop.transform)^i` before `item.transform`.

### Unit primitives

All primitives are centred on the origin and span 1 grammar unit (`unit` mm)
before transforms:

- **3D solids:** `box` (`1 x 1 x 1`), `sphere` (diameter `1`), `cylinder`
  (diameter `1`, height `1` along `z`), `cone` (base diameter `1`, apex at
  `+z/2`, height `1`), `torus` (major radius `0.35`, minor radius `0.15`, outer
  diameter `1`).
- **2D CFDG tiles extruded to `tile_thickness` along `z`:** `square`
  (`1 x 1 x tile_thickness`), `circle` (diameter `1`, height `tile_thickness`),
  `triangle` (equilateral of side `1` about its centroid, height
  `tile_thickness`).

### Transforms: canonical `[...]` vs ordered `[[...]]`

- A **single dict** follows CFDG's canonical `[...]` order regardless of JSON
  key order:
  `translate (x, y, z or t) -> rotate (rz/r, ry, rx) -> scale (s) -> flip -> mirror (fx, fy, fz)`.
- A **list of dicts** `[{"r": 90}, {"x": 10}]` follows CFDG's ordered `[[...]]`
  adjustment list, multiplying steps left-to-right.
- Transform keys:
  - `x`, `y`, `z` or `t: [x, y]` / `[x, y, z]` — translation in grammar units.
  - `r` (alias of `rz`), `rx`, `ry`, `rz` — rotation in degrees.
  - `s` — positive uniform scale `s`, 2D `[sx, sy]` (keeps `sz = 1`), or 3D
    `[sx, sy, sz]`. Zero or negative scales are rejected (`bad_arg`); use
    `flip` or `fx`/`fy`/`fz` for reflections.
  - `flip` — reflection across the line through the origin at angle `deg` in the
    XY plane (`flip: 0` reflects across X, `flip: 90` reflects across Y).
  - `fx`, `fy`, `fz` — boolean mirror across the YZ, XZ or XY plane.
  - Any numeric transform value may be `{"rand": [lo, hi]}` for a seeded uniform
    draw per expansion.
  - CFDG colour/alpha keys (`hue`, `h`, `sat`, `saturation`, `b`, `brightness`,
    `a`, `alpha`, `color`) are accepted and recorded in
    `mesh.meta["shape_grammar"]["ignored_keys"]` so 2D CFDG ports do not need
    hand-stripping.

### Symmetry directives (`CF::Symmetry`)

`symmetry` accepts a CFDG string, a directive object, or a list of directives
(whose group product is deduplicated automatically):

- `"CF::Cyclic, 6"` or `{"type": "cyclic", "n": 6, "axis": "z", "center": [0, 0, 0]}`
  — $C_n$ rotational symmetry of order $n$.
- `"CF::Dihedral, 6"` or `{"type": "dihedral", "n": 6, "axis": "z", "center": [0, 0, 0], "mirror_angle": 90}`
  — $D_n$ dihedral group of order $2n$ ($n$ rotations composed with reflection
  across `mirror_angle`).
- `{"type": "reflect", "axes": ["x", "y"], "center": [0, 0, 0]}` — mirror group
  across one or more principal planes (order $2^k$).

> **CAD connectivity tip:** Exact face-to-face kissing (`x: 1.0` on unit boxes)
> can split into separate components under floating-point rounding. Overlap
> adjacent bricks slightly (e.g. `x: 0.95` or `s: 1.05`) and set
> `max_components: 1` when a single manifold body is required.

---

## 2. `lsystem`

### Arguments

| Argument | Kind | Default | Ceiling | Description |
|---|---|---|---|---|
| `axiom` | string (required) | — | — | Starting module string, e.g. `"A"` or `"A(40, 4)"`. |
| `rules` | object \| list (required) | — | — | Context-free productions (deterministic, stochastic or parametric). |
| `iterations` | int `>= 0` | — | `32` | Number of parallel rewriting steps. |
| `angle` | number (deg) | `25.0` | — | Default turn/pitch/roll angle $\delta$. |
| `step` | length (`mm`) | `10.0` | — | Default forward step length $d > 0$. |
| `width` | length (`mm`) | `1.0` | — | Initial tube diameter $w > 0$. |
| `step_decay` | number `> 0` | `0.9` | — | Multiplier for bare `"` commands. |
| `width_decay` | number `> 0` | `0.707` | — | Multiplier for bare `!` commands (ABOP Leonardo's rule $\approx 1/\sqrt{2}$). |
| `angle_decay` | number `> 0` | `1.0` | — | Multiplier for bare `;` commands. |
| `constants` | object | `{}` | — | Named numeric constants bound in parametric expressions. |
| `seed` | int `>= 0` | `0` | `2^63 - 1` | Seed for stochastic rule selection. |
| `tropism` | object | `null` | — | `{"vector": [tx, ty, tz], "e": 0.2}` bending after each drawn segment. |
| `leaf` | object | `null` | — | `{"symbol": "L", "shape": "sphere", "size": [wL, wU, lenH]}` in `mm`. |
| `segments` | int `>= 3` | `8` | `128` | Circular segments per tube cylinder and joint sphere. |
| `joints` | bool | `true` | — | Place watertight sphere joints of diameter `max(width)` at segment endpoints. |
| `draw_symbols` | string | `"F"` | — | Symbols that advance and emit a tube segment. |
| `move_symbols` | string | `"f"` | — | Symbols that advance without emitting geometry. |
| `min_width` | length (`mm`) | `1e-3` | — | Prune segments thinner than `min_width` (`pruned_thin`). |
| `max_symbols` | int `>= 1` | `200,000` | `1,000,000` | Maximum rewritten module count per iteration. |
| `max_segments` | int `>= 1` | `5,000` | `50,000` | Maximum emitted tube segments. |
| `max_depth` | int `>= 0` | `256` | `10,000` | Maximum `[` ... `]` bracket nesting depth. |
| `max_triangles` | int `>= 1` | `1,000,000` | `5,000,000` | Pre-union and post-union triangle budget. |
| `max_components` | int `>= 1` | `null` | — | Optional upper bound on disconnected components. |

### Productions: deterministic, stochastic and parametric

- **Deterministic map:** `{"F": "F[+F]F[-F]F"}`
- **Stochastic map:**
  ```json
  {"F": [
    {"succ": "F[+F]F[-F]F", "weight": 1.0},
    {"succ": "F[+F]F",      "weight": 1.0}
  ]}
  ```
- **Parametric & conditional list** (ABOP ch. 1.10):
  ```json
  [
    {"pred": "A(l,w)", "cond": "l >= 10", "succ": "!(w)F(l)[+A(l*r,w*0.7)][-A(l*r,w*0.7)]"}
  ]
  ```
  Arithmetic supports `+ - * / ** %`, comparisons `< <= > >= == !=`, boolean
  `and`/`or`/`not`, `pi`, and `sqrt`, `sin`, `cos`, `tan`, `min`, `max`, `abs`,
  `round`. Arbitrary function calls, attribute access and oversize ASTs are
  rejected with `bad_arg`.

### 3D turtle commands (ABOP ch. 1.5)

Initial turtle frame at `(0, 0, 0)`: heading $H = +Z$, left $L = +X$, up
$U = H \times L = +Y$. Every command accepts an optional numeric parameter
override `(x)`:

| Symbol | Action |
|---|---|
| `F` / `F(l)` | Move forward by `step` (or `l`) along $H$ and emit a cylinder of diameter `width`. |
| `f` / `f(l)` | Move forward by `step` (or `l`) without drawing. |
| `+` / `-` | Yaw left / right around $U$ by `angle` (or `(a)`). |
| `&` / `^` | Pitch down / up around $L$ by `angle` (or `(a)`). |
| `\` / `/` | Roll around $H$ by `angle` (or `(a)`). |
| `\|` | Turn $180^\circ$ around $U$. |
| `$` | Roll turtle so $L$ is horizontal ($L = \frac{Z \times H}{\|Z \times H\|}$). |
| `[` / `]` | Push / pop `(pos, H, L, U, width, step, angle)` on the branch stack. |
| `!` / `!(w)` | Multiply `width` by `width_decay`, or set `width = w`. |
| `"` / `"(k)` | Multiply `step` by `step_decay` (or `k`). |
| `;` / `;(k)` | Multiply `angle` by `angle_decay` (or `k`). |
| `L` / `L(k)` | Emit configured `leaf` primitive scaled by `k` along `(L, U, H)`. |

---

## 3. Examples

### CFDG dihedral rosette (`shape_grammar`)

```json
{
  "id": "rosette",
  "op": "shape_grammar",
  "args": {
    "startshape": "R",
    "unit": 3.0,
    "symmetry": "CF::Dihedral, 13",
    "max_components": 1,
    "rules": {
      "R": [
        {"shape": "cylinder", "transform": {"s": [2.0, 2.0, 0.5]}},
        {"call": "ARM", "transform": {"x": 0.8}}
      ],
      "ARM": [{
        "weight": 1.0,
        "max_depth": 5,
        "body": [
          {"shape": "box", "transform": {"s": [1.1, 0.22, 0.4]}},
          {"call": "ARM", "transform": {"x": 0.85, "r": 1.5, "s": 0.92}}
        ]
      }]
    }
  }
}
```

### ABOP 3D tree with leaves (`lsystem`)

```json
{
  "id": "tree",
  "op": "lsystem",
  "args": {
    "axiom": "A",
    "iterations": 4,
    "angle": 22.5,
    "step": 12.0,
    "width": 3.0,
    "width_decay": 0.7,
    "rules": {"A": "!F[&FL!A]/////[&FL!A]///////[&FL!A]"},
    "leaf": {"symbol": "L", "shape": "sphere", "size": [4.0, 1.5, 6.0]},
    "max_components": 1
  }
}
```

---

## 4. Error codes

| Code | Meaning |
|---|---|
| `bad_arg` | Invalid schema, non-finite number, non-positive scale/step/width, unknown rule/primitive, or disallowed expression. |
| `grammar_budget_ceiling` | Requested budget exceeds the hard kernel ceiling (`CEILING`). |
| `grammar_depth_budget` | `shape_grammar` call chain exceeded `max_depth` with `depth_policy="error"`. |
| `grammar_expansion_budget` | Rule evaluations exceeded `max_expansions`. |
| `grammar_instance_budget` | Emitted primitives (including symmetry copies) exceeded `max_instances`. |
| `grammar_triangle_budget` | Input or boolean output triangle count exceeded `max_triangles`. |
| `grammar_overflow` | Coordinates or scales diverged past finite physical bounds (`1e6`). |
| `grammar_empty` | Grammar/L-system produced zero non-pruned primitives. |
| `grammar_components` | Disconnected component count exceeded `max_components`. |
| `grammar_boolean_failed` | `manifold3d` reported a boolean status error or empty result. |
| `grammar_not_closed` | Union output failed manifold closure verification. |
| `lsystem_symbol_budget` | Rewritten L-system string exceeded `max_symbols`. |
| `lsystem_segment_budget` | Emitted tube segments exceeded `max_segments`. |
| `lsystem_depth_budget` | `[` ... `]` bracket nesting exceeded `max_depth`. |
| `lsystem_unbalanced` | Unmatched `]` in axiom or production successor. |
| `lsystem_width` | Segment diameter reached `<= 0`. |

## 5. Verification

```bash
python3 -m pytest -o addopts='' -q tests/test_grammar_ops.py tests/test_families.py tests/test_coverage_gate.py
```
