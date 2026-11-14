# Oriented module ports and deterministic attachment

An assembly part can initialize its pose by attaching one named local port to a named port on another part. This works with recipe, graph, inline spec and model_id parts. Attachment is a rigid placement operation before mate solving. It does not add constraints, remove degrees of freedom, authorize interference, or prove manufacturability.

Public lengths are millimetres and angles are degrees. Kernel poses and mesh coordinates remain metres. Placement matrices and attachment receipt matrices use millimetres in their translation column, so they can be copied into a part's `transform` without conversion.

## Port frame

```json
{"name":"dock", "origin_mm":[0,0,10], "axis":[0,0,1], "x_axis":[1,0,0]}
```

`axis` is the frame Z direction. `x_axis` supplies the roll reference; `tangent` is an alternative spelling, and supplying both is an error. The X vector is normalized and projected onto the plane perpendicular to Z. Frame Y is `Z × X`, giving a right-handed orthonormal frame. The entire local basis is transformed into world coordinates, preserving roll when the part rotates.

Finite non-zero direction vectors are required. A zero or nearly parallel X/Z pair (normalized transverse component ≤ 1e-8), invalid origin, or non-finite radius fails validation and direct assembly building. When an oriented port has `normal` and omits `axis`, its Z inherits `normal`. When both are supplied, normal must point in the same direction as axis (unit dot ≥ 1 − 1e-8). Use attachment `facing` to reverse the source interface. Distinct oriented axis/normal directions are rejected as ambiguous.

Old ports without X retain the previous deterministic basis and independent axis/normal semantics. In particular, a legacy normal-only port still has the historical +Z axis. Existing mate types and auto bbox ports remain available; declared ports override auto ports with the same name. Bbox ports are geometric defaults, not measured interface locations.

## Attach contract

```json
{"id":"cap", "graph":{"features":[
  {"id":"body","op":"box","args":{"size":[20,12,4],"center":[0,0,-2]}}
]}, "ports":[
  {"name":"in", "origin_mm":[0,0,0], "axis":[0,0,1], "x_axis":[1,0,0]}
], "attach":{
  "target":"base", "target_port":"out", "source_port":"in",
  "gap_mm":0.5, "roll_deg":30, "facing":"opposite"
}}
```

- `target`, `target_port` and `source_port` are required explicit names. Target parts may occur later in the list.
- `gap_mm` defaults to 0. Its signed offset always follows the target port's world +Z, regardless of facing.
- `roll_deg` defaults to 0. It is a right-handed rotation around target +Z.
- `facing` defaults to `opposite`. `same` aligns both Z axes. `opposite` applies a proper 180° rotation about target X, preserving X and reversing Y/Z before applying roll. This is never a reflection.
- `attach` and `transform` are mutually exclusive. Bake the resulting placement into `transform` if a later workflow needs an explicit pose.
- Unknown fields, absent parts/ports, self references and reference cycles are explicit errors. The contract intentionally has no nested expression/offset DSL.

Let `Ft` be the target world frame rotation, `Fs` the source local frame rotation, `os` the local source origin and `ot` the target world origin. Then:

```text
C = Rz(roll_deg)                 # facing=same
C = Rz(roll_deg) @ Rx(180°)       # facing=opposite
R = Ft @ C @ Fs.T
t = ot + gap_mm/1000 * Ft[:,2] - R @ os
```

All parts build locally first. Attachments are evaluated in a topological target order, with lexical tie-breaking among independent parts. Assemblies using attachment therefore have placements independent of part-list order, including the implicit root anchor when no part is fixed. Assemblies without attachment retain their historical list/anchor behavior. For explicit solver intent, mark the intended root `fixed:true`.

## Runnable recipe/graph example

```python
from atta import assembly
from atta.spec import validate_spec

spec = {
    "name": "two_modules", "material": "steel",
    "model": {"kind": "assembly", "parts": [
        {"id": "cap", "graph": {"features": [
            {"id": "body", "op": "box", "args": {
                "size": [20, 12, 4], "center": [0, 0, -2]}}
        ]}, "ports": [{"name": "in", "origin_mm": [0, 0, 0],
                        "axis": [0, 0, 1], "x_axis": [1, 0, 0]}],
         "attach": {"target": "base", "target_port": "out", "source_port": "in",
                    "gap_mm": 0.5, "roll_deg": 30, "facing": "opposite"}},
        {"id": "base", "fixed": True, "recipe": "primitive.chamfer_box",
         "params": {"length": 40, "width": 30, "height": 10},
         "ports": [{"name": "out", "origin_mm": [0, 0, 20],
                    "axis": [0, 0, 1], "x_axis": [1, 0, 0]}]}
    ], "mates": [
        {"id": "hold", "type": "lock", "a": "base", "b": "cap",
         "port_a": "out", "port_b": "in"}
    ]}
}
validated = validate_spec(spec)
assert validated["ok"], validated["errors"]
built = assembly.build(validated["spec"])
receipt = assembly.verify(built, constraints={"fully_constrained": True})
assert receipt["verdict"] == "pass", receipt["issues"]
print(built["assembly"]["attachments"])
```

A lock mate captures the relative pose after attachment and constrains all six relative DOFs. Other mates may intentionally move an attached part; attachment is only an initial condition. Collision and clearance gates use the final world meshes. Negative gap or unsuitable interface locations can produce a perfectly computed attachment and a failing interference check. Use the existing clearance/interference constraints and exact receipt verdict; do not call successful snapping manufacture-ready.

NURBS graph modules use the same contract. For example, build `nurbs_surface` with `control_points=[[[0,0,0],[0,20,0]],[[30,0,0],[30,20,0]]]`, feed its output into `thicken` with `thickness=2`, and declare/attach ports on that solid. Thickness and interface origin remain mm; the new surface op needs no special assembly integration.

## Receipt and verification

`assembly.attachments` is exposed by model/describe and contains `method`, `initialization_only:true`, `order`, and `items`. Each item names source/target parts and ports, gap/roll/facing, the source's initial matrix, and initial world port frames. Frame origins are mm and axes are unit directions. After solving, `final_position_error_mm` and `final_rotation_error_deg` diagnose how far the final result moved from the attachment relation. Those diagnostics do not add a gate or override mate, DOF, solid, collision or clearance checks.

Assembly verification also evaluates declared whole-assembly mesh constraints (including `bbox` and `volume_mm3`) on the final combined world mesh. These dimensions are not imposed independently on each local part. Approximation failures and inherited source issues on each part remain visible: an unmet NURBS sampled target yields `degraded` at basic/standard and `fail` at strict. Delivery blocks a failed receipt; degraded delivery requires the existing explicit `allow_degraded=true` option.

The regression suite covers arbitrary world rotations, opposing normals, roll, proper rotations, forward references and permutation invariance, mm matrix round trips, invalid frames/aliases, missing refs and cycles, initialization changed by mates, lock DOF, actual overlap rejection, legacy ports/mates, and a NURBS/thicken graph part through engine model/describe/verify. `tests/test_assembly_acceptance.py` additionally checks whole-assembly size gates, source-quality gates, and STEP geometry after solving.

## STEP compound delivery

`engine.deliver(formats=["step", "glb"])` rebuilds supported exact graph parts using the same BREP construction and applies their **final solved world poses** to a millimetre STEP compound. GLB includes all tessellated parts. STEP is not a native CAD assembly hierarchy or an encoding of the mate system.

Graph parts whose reachable closure contains mesh-only operations, and recipe parts without an implemented exact assembly mapping, are explicitly listed in STEP `omitted` and delivery `errors`. A mixed export therefore has `status="partial"` even when geometric verification passes. No mesh-to-BREP approximation is silently substituted. A missing CAD dependency, invalid BREP, selection error or other actual build failure retains its original `code/path/fix/details` and fails the STEP format rather than being renamed an unsupported-part omission. Other requested formats may still succeed.

The eight facing/unit combinations in the low-level regression compare reimported STEP vertices with final world mesh vertices after arbitrary rotations, roll, parameter substitution and mate movement. The public Spec remains millimetre-based; low-level graph builders additionally accept cm/m/in. See the runnable [mixed-kernel example](../examples/kernel_showcase/README.md) for the exact/mesh coverage boundary and manifest checks.
