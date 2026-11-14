#!/usr/bin/env python3
"""Compare kernel changes with a git baseline in bounded, fresh processes.

No checkout, plugin configuration or external assets are modified. Baseline
modules are read with git show; their unchanged dependencies use this checkout.
Peak RSS includes imports and verification. Timings exclude input construction
and final fact/hash calculation; they are single-run observations, not CI gates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ("field_sphere", "field_shell", "field_elongated", "field_dense",
         "smooth", "decimate", "decimate_million", "measurement")


def baseline_module(path, ref):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
    # Family modules also register recipes at import time. The benchmark only
    # needs their pure kernel helper; loading decorators would collide with the
    # current registry and would make a baseline measurement non-reproducible.
    if path == "atta/families/lattice.py" and "@recipe(" in source:
        source = source.split("@recipe(", 1)[0]
    package = ".".join(Path(path).parts[:-1])
    name = package + ".__kernel_benchmark_baseline"
    module = types.ModuleType(name)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
    return module


def worker(case, implementation, ref):
    import numpy as np
    from atta.mesh import Mesh
    from atta.primitives import box, sphere

    # BLAS thread limits are set before interpreter startup by the parent.
    memory_limit = 4 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    legacy = implementation == "baseline"
    details = {}
    if case.startswith("field_"):
        if legacy:
            extract = baseline_module("atta/families/lattice.py", ref).marching_tets
        else:
            from atta.families.lattice import marching_tets as extract
        if case == "field_elongated":
            x, y, z = np.meshgrid(np.linspace(-1.1, 1.1, 160),
                                  np.linspace(-.12, .12, 26),
                                  np.linspace(-.12, .12, 26), indexing="ij")
            field = (x / 1.0) ** 2 + (y / .08) ** 2 + (z / .08) ** 2 - 1
            spacing = (2.2 / 159, .24 / 25, .24 / 25)
            origin = (-1.1, -.12, -.12)
        else:
            n = 32 if case == "field_dense" else 64
            axis = np.linspace(-1.2, 1.2, n)
            x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
            radius = np.sqrt(x*x + y*y + z*z)
            field = (np.sin(20*x) * np.sin(20*y) * np.sin(20*z)
                     if case == "field_dense" else
                     (np.abs(radius - .8) - .08 if case == "field_shell" else radius - 1))
            if case == "field_dense":
                field[[0, -1], :, :] = 1
                field[:, [0, -1], :] = 1
                field[:, :, [0, -1]] = 1
            spacing, origin = 2.4 / (n - 1), (-1.2,) * 3
        start = time.perf_counter()
        result = extract(field, origin=origin, spacing=spacing)
        elapsed = time.perf_counter() - start
        details["samples"] = int(field.size)
    elif case in ("smooth", "decimate", "decimate_million"):
        if legacy:
            repair = baseline_module("atta/repair.py", ref)
        else:
            from atta import repair
        if case == "smooth":
            source = sphere(.05, segments=128, rings=64)
            start = time.perf_counter()
            result = repair.laplacian_smooth(source, iterations=10)
        else:
            segments = 1024 if case == "decimate_million" else 128
            source = sphere(.05, segments=segments, rings=segments // 2)
            start = time.perf_counter()
            kwargs = {} if legacy else {"max_error": .0005}
            result = repair.quadric_decimate(source, 10000 if segments > 128 else 5000, **kwargs)
        elapsed = time.perf_counter() - start
        details.update(input_faces=len(source.faces), input_volume_m3=source.volume())
        if "decimation" in result.meta:
            details["decimation"] = result.meta["decimation"]
    elif case == "measurement":
        measure = baseline_module("atta/mesh.py", ref) if legacy else None
        result = box([.01, .02, .03], center=(1e6, -1e6, 1e6))
        start = time.perf_counter()
        details["measured_volume_m3"] = measure.volume(result) if legacy else result.volume()
        details["centroid_m"] = (measure.centroid(result) if legacy else result.centroid()).tolist()
        elapsed = time.perf_counter() - start
        details["expected_volume_m3"] = 6e-6
    else:
        raise ValueError(case)
    topology = result.topology()
    return {
        "status": "ok", "seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "faces": len(result.faces), "verts": len(result.verts),
        "mesh_hash": result.hash(), "topology": topology,
        "volume_m3": result.volume(), **details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="d371e44")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/benchmarks/kernel.json")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--implementation", choices=("baseline", "current"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.implementation, args.baseline)))
        return
    import importlib.metadata
    report = {
        "schema": "atta.kernel-benchmark/v1", "baseline": args.baseline,
        "current_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_changes": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "manifold3d")},
        "protocol": {
            "repetitions": 1, "threads": 1, "address_space_limit_gib": 4,
            "rss_includes_imports_and_fact_calculation": True,
            "baseline_modules_share_unchanged_dependencies": True,
            "face_target_best_effort": True,
            "note": "Decimation methods differ: baseline has no error budget; current max_error=0.5mm. Timeouts are censored, not measured speedups.",
        }, "cases": {},
    }
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for case in args.cases:
        pair = {}
        for implementation in ("baseline", "current"):
            budget = min(args.timeout, 10) if implementation == "baseline" and case.startswith("decimate") else args.timeout
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                       "--implementation", implementation, "--baseline", args.baseline]
            try:
                completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                           capture_output=True, timeout=budget)
                if completed.returncode:
                    pair[implementation] = {"status": "error", "returncode": completed.returncode,
                                            "error": completed.stderr[-4000:]}
                else:
                    pair[implementation] = json.loads(completed.stdout)
            except subprocess.TimeoutExpired:
                pair[implementation] = {"status": "timeout", "budget_seconds": budget}
            print(case, implementation, pair[implementation]["status"],
                  pair[implementation].get("seconds", ""), flush=True)
        if all(pair[k]["status"] == "ok" for k in ("baseline", "current")):
            pair["same_hash"] = pair["baseline"]["mesh_hash"] == pair["current"]["mesh_hash"]
        report["cases"][case] = pair
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
