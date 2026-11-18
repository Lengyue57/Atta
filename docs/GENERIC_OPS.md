# Generic Feature-Graph Ops (`atta/families/generic_ops.py`)

This document describes the seven generic feature-graph ops added to widen
Atta's Spec v3 DAG vocabulary for unseen shapes, together with the mapping of
pre-existing family ops that cover the remaining `MISSING_OPS` entries in
`scripts/run_bench.py`.

All graph arguments use **millimetres** (or model units configured via
`units=...`) and **degrees**; all kernel `Mesh` inputs and outputs use
**metres**. Every op returns a closed, outward-oriented manifold mesh or raises
`AttaError(code, message, path=..., fix=...)`, enforces hard vertex/triangle/sample
budgets (failing closed), and executes deterministically.

---

## New Ops (`atta/families/generic_ops.py`)

### 1. `hull`
Convex hull of one or more input meshes (`min_inputs=1`, receives `meshes=[...]`).

- **Required args:** none
- **Optional args:**
  - `max_vertices` (`int >= 4`, default `250000`, cap `1000000`)
  - `max_triangles` (`int >= 4`, default `500000`, cap `1500000`)
- **Algorithm:** Deduplicates all input vertices deterministically and runs
  `scipy.spatial.ConvexHull`, extracts used vertices/simplices, orients faces
  outward relative to the interior centroid, and verifies watertight manifold
  topology (`meta["exact"] = True`).
- **Errors:** `budget_exceeded`, `hull_degenerate`, `open_input`/`bad_arg`.

### 2. `minkowski`
Minkowski sum $A \oplus B$ across two or more closed manifold inputs (`min_inputs=2`).

- **Required args:** none
- **Optional args:**
  - `method` (`"auto" | "exact" | "sdf"`, default `"auto"`)
  - `resolution` (`int` in `8..128`, default `48`) — used by SDF fallback
  - `max_vertices` (`int >= 4`, default `500000`, cap `1000000`) — pairwise vertex cloud budget
  - `max_triangles` (`int >= 4`, default `500000`, cap `1500000`)
  - `max_samples` (`int >= 343`, default `500000`, cap `1200000`)
  - `max_face_product` (`int >= 1`, default `12000`, cap `200000`) — threshold for exact non-convex manifold sum
- **Algorithm & Exactness:**
  - **Convex $\oplus$ Convex** (when `method != "sdf"`): exact convex hull of
    pairwise hull-vertex sums $\{a + b : a \in V(A), b \in V(B)\}$. Sets
    `meta["exact"] = True`, `meta["minkowski"]["exact"] = True`,
    `method = "convex_hull_pairwise_sum"`.
  - **General (non-convex)** when `len(faces_A) * len(faces_B) <= max_face_product`
    and `method != "sdf"`: exact `manifold3d.Manifold.minkowski_sum` (with
    tiny simplification only if float32 quantization creates zero-area slivers).
    Sets `meta["exact"] = True`, `method = "manifold_minkowski_sum"`.
  - **Documented SDF Approximation** (`method == "sdf"`, or `method == "auto"`
    when `face_product > max_face_product` or `pairwise_points > max_vertices`):
    voxel occupancy convolution (`scipy.signal.fftconvolve`) + signed Euclidean
    distance transform (`distance_transform_edt`) + bounded marching tetrahedra
    (`sdfkit.extract`). Explicitly marks `meta["exact"] = False` and
    `meta["minkowski"]["exact"] = False`.
- **Errors:** `budget_exceeded` (e.g. `method="exact"` above budget),
  `open_input`, `minkowski_empty`, `minkowski_degenerate`.

### 3. `polyhedron`
OpenSCAD-style solid from explicit vertices and face index loops (`arity=0`).

- **Required args:**
  - `points` (`mat3`: `[[x, y, z], ...]`, `n >= 4`, mm)
  - `faces` (`list[list[int]]`, `>= 4` loops of `>= 3` vertex indices; n-gons are
    projected to their best-fit plane and ear-clip triangulated)
- **Optional args:**
  - `convexity` (`int >= 1`, default `1`)
  - `fix_winding` (`bool`, default `True`) — flood-fills consistent edge winding
    and orients each connected shell outward
  - `weld_tol` (`length >= 0`, default `0.0`, mm)
  - `max_vertices` (`int >= 4`, default `250000`)
  - `max_triangles` (`int >= 4`, default `500000`)
- **Validation & Errors:** Fails closed with concrete `fix` on open shells
  (`polyhedron_open`), non-manifold edges (`polyhedron_non_manifold`),
  degenerate/zero-area faces (`polyhedron_degenerate`), non-triangulable n-gons
  (`polyhedron_bad_face`), non-orientable surfaces (`polyhedron_non_orientable`),
  or bad/inward winding when `fix_winding=False` (`polyhedron_bad_winding`,
  `polyhedron_inside_out`).

### 4. `heightmap`
OpenSCAD `surface()` equivalent: turns a 2D height grid into a closed solid
with a flat base at `z = -base` (`arity=0`).

- **Required args:**
  - `heights` (`2D array[rows][cols]`, `rows >= 2, cols >= 2`, mm)
- **Optional args:**
  - `cell_size` (`number > 0` or `[dx, dy]`, default `1.0` mm; mutually exclusive with `size`)
  - `size` (`[Lx, Ly]` positive mm; derives `dx = Lx / (cols - 1)`, `dy = Ly / (rows - 1)`)
  - `base` (`length >= 0`, default `1.0` mm) — flat bottom plane placed at `z = -base`
  - `center` (`bool`, default `True`) — centers X/Y around `(0, 0)`
  - `invert` (`bool`, default `False`) — negates heights before placing over `z = -base`
  - `max_vertices` (`int >= 4`, default `250000`)
  - `max_triangles` (`int >= 4`, default `500000`)
- **Geometry:** Each quad cell is symmetrically triangulated into 4 triangles
  around its bilinear midpoint $(x_c, y_c, \tfrac{1}{4}(h_{00}+h_{10}+h_{11}+h_{01}))$,
  giving exact bilinear prism volume $\sum dx\,dy\,(\bar{h}_{ij} + \text{base})$.
- **Errors:** `heightmap_degenerate` (if `min(z_top) <= -base`), `budget_exceeded`, `bad_arg`.

### 5. `deform`
Smooth axis or free-form lattice deformation with bounded auto-subdivision (`arity=1`).

- **Required args:**
  - `mode` (`"twist" | "bend" | "taper" | "lattice"`)
- **Optional args:**
  - `axis` (`"x" | "y" | "z"`, default `"z"`)
  - `bend_axis` (`"x" | "y" | "z"`, perpendicular to `axis`; default first orthogonal axis)
  - `angle_deg` (`number`, default `0.0`) — total angle for `twist` and `bend`
  - `start_scale` (`number > 0`, default `1.0`), `end_scale` (`number > 0`, default `1.0`) — for `taper`
  - `lattice` (`{"shape": [nx, ny, nz], "offsets": ...}` with `2 <= ni <= 8` and
    offsets in mm of shape `(nx*ny*nz, 3)` or `(nx, ny, nz, 3)`) — tensor-product
    Bernstein FFD for `mode="lattice"`
  - `max_edge` (`length > 0`, optional mm; default `bbox_diag / 28`) — input edges
    longer than `max_edge` are iteratively split before deformation
  - `max_vertices` (`int >= 4`, default `250000`)
  - `max_triangles` (`int >= 4`, default `500000`)
- **Errors:** `budget_exceeded` (checked *before* allocating each subdivision round),
  `deform_self_intersection` (when `bend` radius folds inside the part thickness),
  `deform_degenerate`, `open_input`.

### 6. `morph`
Smoothly interpolates two closed manifold solids via linear SDF blend at
$t \in [0, 1]$ (`min_inputs=2`).

- **Required args:** none
- **Optional args:**
  - `t` (`number` in `[0, 1]`, default `0.5`)
  - `resolution` (`int` in `8..128`, default `40`)
  - `align_centers` (`bool`, default `False`) — translates input B's bbox center onto A's before blending
  - `max_samples` (`int >= 343`, default `500000`, cap `1200000`)
  - `max_triangles` (`int >= 4`, default `500000`, cap `1500000`)
- **Algorithm:** Exact copies at $t = 0$ and $t = 1$. For $t \in (0, 1)$, computes
  signed distances $\phi_A, \phi_B$ on a padded `sdfkit.grid`, blends
  $\phi = (1 - t)\phi_A + t\phi_B$, and extracts a closed manifold mesh with
  `sdfkit.extract` (`marching_tets`).
- **Errors:** `morph_empty` (when disjoint inputs leave $\min \phi \ge 0$),
  `budget_exceeded` / `sdf_too_dense`, `open_input`, `bad_arg`.

### 7. `projection_extrude`
OpenSCAD `projection(cut=true/false)` followed by linear extrusion (`arity=1`).

- **Required args:**
  - `thickness` (`length > 0`, mm)
- **Optional args:**
  - `cut` (`bool`, default `False`) — `False` computes orthographic shadow projection; `True` slices at `cut_offset`
  - `cut_offset` (`length`, default `0.0`, mm) — slice plane coordinate along `axis`
  - `axis` (`"x" | "y" | "z"`, default `"z"`)
  - `center` (`bool`, default `False`) — centers extrusion in `[-thickness/2, +thickness/2]`
  - `max_triangles` (`int >= 4`, default `500000`)
  - `max_input_triangles` (`int >= 4`, default `500000`)
- **Algorithm:** Uses `manifold3d.Manifold.project()` / `.slice(cut_offset)`
  and `CrossSection.extrude(thickness)`, preserving outer loops and inner holes
  (e.g. genus-1 extrusion of a projected torus).
- **Errors:** `projection_empty`, `budget_exceeded`, `open_input`, `bad_arg`.

---

## Mapping of Existing Family Ops (No Duplication)

Several entries in `scripts/run_bench.py`'s `MISSING_OPS` were already shipped in
existing family modules and were intentionally **not duplicated** here:

| Corpus / Bench Op | Existing Atta Op | Module | Notes |
|---|---|---|---|
| `subdiv` | `subdivide` (`arity=1`) | `atta/families/kernel_ops.py` | Catmull-Clark, Loop, and Simple subdivision (`levels=1..5`, `crease_angle_deg`). |
| `smooth` | `repair` (`arity=1`, `smooth>0`) / `subdivide` / `sdf_blend` | `atta/families/kernel_ops.py` | `repair` runs Taubin smoothing (`lam=0.5, mu=-0.53`), hole fill, and consistent orientation. |
| `remesh` | `repair` (`target_faces`, `max_error_mm`) | `atta/families/kernel_ops.py` | Tolerance-bounded Manifold simplification; `atta.repair.remesh_isotropic` is also available in Python. |
| `offset` | `offset` (`arity=1`) | `atta/families/offset_ops.py` | `mode="round"` (SDF surface nets), `"mitre"` (exact L1 hull/halfspace), `"shell"` (hollow). |
| `libfive_tree` | `field_tree` (`arity=0`) / `sdf_blend` (`arity=0`) | `atta/families/field_ops.py`, `kernel_ops.py` | Nested scalar field DAG (`tree`, `bounds`, `spacing`, hard budgets). |
| `skeleton`, `silhouette`, `weld_union` | `skeleton`, `silhouette`, `weld_union` | `atta/families/form_ops.py`, `bio.py` | Form-first builders and batch weld union. |
