# Rational tensor-product NURBS surfaces

`nurbs_surface` is an **open** parametric graph surface. Chain it through
`thicken` for a boundary-closed mesh solid, then ordinary booleans/assembly.
It is not a STEP/BREP surface exporter, trimmed patch system, or intersection solver.

Implementation: [kernel](../atta/nurbs.py), [automatic family registration](../atta/families/nurbs_ops.py),
[regressions](../tests/test_nurbs.py). The implementation uses NumPy; tests use
SciPy's public `BSpline` evaluator as an independent numerical oracle. There is no
copied third-party NURBS source or new dependency.

## Graph contract

| Argument | Default | Meaning |
|---|---:|---|
| `control_points` | required | Rectangular `[nu][nv][3]`, at least 2 controls per axis, in model length units |
| `weights` | all 1 | Rectangular `[nu][nv]`, finite and strictly positive |
| `degree_u`, `degree_v` | `min(3, axis_count-1)` | Whole number 1..8, smaller than axis control count |
| `knots_u`, `knots_v` | clamped uniform | Full, nondecreasing dimensionless knot vectors, length `axis_count+degree+1` |
| `tolerance` | 0.05 | Positive **sampled** triangle-correspondence tolerance, in model length units |
| `u_samples`, `v_samples` | 3 | Initial points **per nonzero active knot span**, 2..1025 (not total axis points) |
| `max_vertices` | 100000 | Per-output mesh vertex budget; hard maximum 250000 |
| `max_faces` | 200000 | Per-output triangle budget; hard maximum 500000 |
| `max_evaluations` | 1000000 | Cumulative positions evaluated over all refinement levels; hard maximum 2000000 |
| `max_refinements` | 8 | Global bisections after initial grid; 0..16 |
| `fail_on_tolerance` | true | Raise `nurbs_tolerance_unmet` if the sampled target was not reached |
| `flip` | false | Reverse every triangle's winding |

Model units default to **mm**. `control_points` and `tolerance` are converted
**once in the family prepare boundary**, using the model's unit scale. Knots,
weights, counts and degrees are never length-scaled. The direct Python kernel
uses **metres** and its default tolerance is `5e-5 m`. In a graph using `units=m`,
the default `0.05` means `0.05 m`: supply the tolerance explicitly when changing
model units. A graph contains no `thickness` argument on `nurbs_surface`; use a
separate `thicken` feature.

### Exact rational quarter-cylinder, then a solid

The weight `sqrt(1/2)` makes the quadratic quarter-circle exact analytically.
The returned mesh still approximates that analytic surface with triangles.

```json
{
  "schema": "atta.spec/v3",
  "name": "quarter-cylinder-shell",
  "model": {
    "kind": "graph",
    "units": "mm",
    "features": [
      {
        "id": "skin",
        "op": "nurbs_surface",
        "args": {
          "control_points": [
            [[30, 0, 0], [30, 0, 50]],
            [[30, 30, 0], [30, 30, 50]],
            [[0, 30, 0], [0, 30, 50]]
          ],
          "weights": [[1, 1], [0.7071067811865476, 0.7071067811865476], [1, 1]],
          "degree_u": 2,
          "degree_v": 1,
          "knots_u": [0, 0, 0, 1, 1, 1],
          "knots_v": [0, 0, 1, 1],
          "tolerance": 0.01
        }
      },
      {"id": "shell", "op": "thicken", "inputs": ["skin"], "args": {"thickness": 2}}
    ],
    "output": "shell"
  }
}
```

The centered shell has an expected volume near
`(pi/2) * 30 * 50 * 2 = 4712.389 mm³`. This is a mesh shell, not an exact analytic
offset: `thicken` uses mesh vertex normals. Boundary closure and measured volume
are tested, but neither this formula nor topology proves absence of self-intersection.

## Python evaluation and derivatives

```python
import numpy as np
from atta.nurbs import NURBSSurface, tessellate
from atta.surfaces import thicken

s = NURBSSurface(
    [[[0, 0, 0], [0, .02, 0]], [[.03, 0, 0], [.03, .02, 0]]],
    degree_u=1, degree_v=1,
)
p = s.evaluate(.25, .75)             # [.0075, .015, 0] metres
su, sv = s.derivatives(.25, .75)      # [.03, 0, 0], [0, .02, 0]
n = s.normal(.25, .75)               # [0, 0, 1]
# Paired broadcasting; explicit [:,None] / [None,:] requests an outer grid.
positions = s.evaluate(np.linspace(0, 1, 5)[:, None], np.linspace(0, 1, 7)[None, :])
patch = tessellate(s, tolerance=1e-5)
solid = thicken(patch, .002)
```

The parameter domain on each axis is `[knots[degree], knots[control_count]]`.
It need not be `[0,1]`. Evaluation **does not extrapolate**. First partials are
with respect to the **original knot parameter units**, computed analytically
from B-spline basis derivatives and the rational quotient rule. `normal` is the
unit cross product `Su × Sv`, not a finite difference or mesh normal. At a
repeated interior knot, evaluation/derivatives choose the right span; at the
final active endpoint they use the left span. C0 knots may have different left
and right normals. Singular or numerically near-parallel tangents raise
`nurbs_singular` rather than supplying a fabricated normal.

## What the refinement report does — and does not — prove

Each active nonzero knot span is present in the initial grid. Each refinement
bisects **all** intervals in both directions; neighboring cells share exact
vertex indices. No local quadtree, T-junction, spatial tolerance weld, or patch
seam repair is involved. Interior knot lines are included, and every rectangle
is split along the same 00→11 diagonal. The guarantee is **conforming/crack-free
connectivity within one patch**, not stitching independent patches together.

Nine parameter probes per cell measure correspondence to its output triangles:
four boundary midpoints, diagonal midpoint, two triangle barycenters, and two
additional diagonal points. `sampled_error` is the largest Euclidean distance
at these probes between the analytic surface and barycentric triangle point.
Output float64 world-coordinate rounding is included in those triangle points;
`max_vertex_roundoff` separately reports the largest observed vertex rounding.

`mesh.meta["nurbs"]` includes:

- `method`, `error_metric`, `exact: false`, `length_unit: "m"`;
- `sampled_error`, `tolerance`, `max_vertex_roundoff` in metres;
- **`surface_error_bound: null`** — no fabricated all-domain error guarantee;
- `target_met`, `stop_reason`, `grid_shape`, `refinements`, cumulative `evaluations`;
- applied vertex/face/evaluation/work budgets and scratch chunk size;
- `crack_free_within_patch: true`, `cross_patch_sewing: false`,
  `self_intersection_checked: false`.

These finite samples can miss narrow or oscillatory deviations. A satisfied
sampled target is **not** a Hausdorff bound, an all-domain maximum, geometric
regularity proof, or manufacturing guarantee. Analytic normals at requested
points do not prove regularity over the entire patch either. Uniform global
refinement intentionally spends samples in flat regions too, favoring simple,
bounded, conforming connectivity over minimal face count.

Default behavior fails on an unmet sampled target. Set `fail_on_tolerance=false`
only when an explicitly degraded approximation is acceptable: the mesh then
carries `meta.issues` with `nurbs_tolerance_unmet` and `target_met: false`.
`thicken` and rigid mesh transforms preserve these metadata. Boolean operations
carry a flat, deduplicated `geometry_provenance.sources` record plus source
`issues`; they do not copy the source NURBS description into the result's top-level
attributes or derive an error bound for the new surface. A missed source target
remains `surface_tolerance` in verification (`degraded` at basic/standard,
`fail` at strict), including inside assemblies. Generic inherited issues are
reported as `geometry_issue.*`. Provenance is limited to 128 source records and
128 issues, with a combined 256 KiB serialized budget; exceeding this budget
raises `geometry_provenance_budget` instead of discarding a warning.
Even degraded mode cannot override an over-budget initial grid or malformed input.

## Allocation, numerical and shape limits

- At most **65536 control points**, at most **1024 controls per axis**. Numeric
  nested shapes/types and item counts are checked before NumPy dense allocation;
  ndarray views with huge declared sizes are rejected before copying.
- A maximum **64000000 basis products** bounds evaluation work independently of
  vertex count. Degree support is local; no dense `samples × all controls` basis
  matrix or `u_samples × v_samples × all controls` tensor is created.
- Evaluator and error-probe temporaries are chunked to **1024 paired samples**.
  Initial/future grids, triangle arrays, query broadcasting and cumulative work
  are budgeted before allocation. User budget values cannot exceed hard caps.
  Budget counters describe positions evaluated, not bytes or a wall-clock SLA.
- Input containers themselves already exist outside this kernel. Python/NumPy
  overhead and downstream `thicken`/boolean allocations are not billed to the
  patch's budgets; `thicken` approximately doubles the vertex count and adds
  boundary triangles. This is not a process-RSS limiter.
- Positive weights only; `max(weight)/min(weight) <= 1e12`. Multiplying all
  weights by a common representable factor leaves the surface unchanged.
- Endpoint multiplicity at most `degree+1`; active interior multiplicity at
  most `degree` (C0 or smoother). Disconnected/discontinuous spline pieces are
  rejected. Neighboring active knot values must be separated by at least
  `64 * machine epsilon` relative to the active domain.
- Computation subtracts a local origin and normalizes spatial extent to avoid
  catastrophic world-coordinate cancellation in rational derivatives and
  fixed absolute normal thresholds. Tests include **1e6 m translations** and
  **1e-12 m patches**, including vertical-patch shelling after the scale-aware
  core `thicken` fix. Float64 cannot recover input detail already rounded away;
  collapsed output triangles fail instead of silently welding.
- Only continuous, untrimmed rectangular-domain patches. No periodic index
  wrap, trims, pole collapsing, automatic multipatch sewing, curvature/second
  partial API, analytic offset, or NURBS STEP export. A full cylinder with a
  duplicated seam is not automatically a closed periodic surface.
- Detected triangle collapse fails clearly, but folded/self-intersecting
  patches or excessive shell thickness may still be invalid; run ordinary
  downstream verification. Topological closure is not a self-intersection proof.

## Tests

```bash
python3 -m pytest tests/test_nurbs.py -q
```

Coverage includes an analytic rational cylinder, weight rescaling, SciPy tensor
oracle (nonuniform/nonclamped/repeated knots), derivative finite differences,
parameter-domain chain rule, mm/cm/m boundary conversion, crack-free shared
indices, deterministic hashes, global refinement, pre-allocation budgets,
malformed inputs, default failure/explicit degraded metadata, tiny and
translated geometry, solid shell volume and a through-hole boolean.
