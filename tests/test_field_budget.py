# -*- coding: utf-8 -*-
"""Bounded field extraction: legacy geometry, seams, units and hard budgets."""
import math

import numpy as np
import pytest

from atta import form, sdf, sdfkit
from atta.errors import AttaError
from atta.families import kernel_ops
from atta.families.lattice import marching_tets as lattice_extract
from atta.features import evaluate


def _sphere(n=32):
    axis = np.linspace(-0.03, 0.03, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.sqrt(x*x + y*y + z*z) - 0.019, [axis[0]] * 3, axis[1] - axis[0]


@pytest.mark.parametrize("chunk_cells", [257, 16384, 100000])
def test_chunks_preserve_legacy_sphere_hash_and_closed_seams(chunk_cells):
    # Captured from the pre-optimization extractor, not from this implementation.
    phi, origin, spacing = _sphere()
    mesh = sdf.marching_tets(phi, origin, spacing, chunk_cells=chunk_cells)
    assert mesh.hash() == "cbdcef1f7e472f966ad11c00f2735d8dd20dd4c3c6275d0148d885aacc9af94c"
    topo = mesh.topology()
    assert topo["closed"] and topo["genus"] == 0
    assert topo["boundary_edges"] == topo["non_manifold_edges"] == 0
    assert mesh.meta["extraction"]["active_cells"] == 1802
    assert mesh.meta["extraction"]["chunk_cells"] == chunk_cells


def test_old_entry_points_use_the_same_extractor():
    args = _sphere(20)
    hashes = [fn(*args).hash() for fn in (lattice_extract, sdfkit.extract, form._isosurface)]
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("shape", ["torus", "shell", "exact_zero", "anisotropic", "dense"])
def test_topology_and_hash_are_chunk_independent(shape):
    n = 20
    spacing = np.array([0.002, 0.0015, 0.0025])
    origin = -spacing * (n-1) / 2
    x, y, z = np.meshgrid(*(origin[i] + np.arange(n)*spacing[i] for i in range(3)), indexing="ij")
    radius = np.sqrt(x*x + y*y + z*z)
    if shape == "torus":
        phi = np.sqrt((np.sqrt(x*x+y*y)-0.008)**2+z*z)-0.0035
    elif shape == "shell":
        phi = np.abs(radius-0.010)-0.0025
    elif shape == "exact_zero":
        phi = np.maximum.reduce([np.abs(x)-abs(x[4, 0, 0]),
                                 np.abs(y)-abs(y[0, 4, 0]),
                                 np.abs(z)-abs(z[0, 0, 4])])
    elif shape == "dense":
        phi = np.where(np.indices((n, n, n)).sum(axis=0) % 2 == 0, 1.0, -1.0)
        phi[[0, -1], :, :] = phi[:, [0, -1], :] = phi[:, :, [0, -1]] = 1
    else:
        phi = radius-0.011
    a = sdf.marching_tets(phi, origin, spacing, chunk_cells=257)
    b = sdf.marching_tets(phi, origin, spacing, chunk_cells=20000)
    assert a.hash() == b.hash()
    assert a.topology() == b.topology()
    assert a.topology()["closed"]
    tris = a.verts[a.faces]
    assert np.all(np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=1) > 0)


def test_tetrahedron_work_never_exceeds_cell_chunk(monkeypatch):
    original = sdf._cell_triangles
    seen = []
    def bounded(f8, cell, *args):
        seen.append(len(cell))
        assert len(cell) <= 128
        yield from original(f8, cell, *args)
    monkeypatch.setattr(sdf, "_cell_triangles", bounded)
    sdf.marching_tets(*_sphere(20), chunk_cells=128)
    assert len(seen) > 1


def test_output_budget_stops_dense_surface_before_retaining_it():
    phi, origin, spacing = _sphere(20)
    with pytest.raises(AttaError) as exc:
        sdf.marching_tets(phi, origin, spacing, chunk_cells=128, max_triangles=500)
    assert exc.value.code == "sdf_too_dense"
    assert exc.value.details["max_triangles"] == 500


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_field_is_rejected(value):
    phi, origin, spacing = _sphere(12)
    phi[0, 0, 0] = value
    with pytest.raises(AttaError) as exc:
        sdf.marching_tets(phi, origin, spacing)
    assert exc.value.code == "bad_arg"


@pytest.mark.parametrize("size,spacing", [([1.070, .197, .032], .032/48),
                                            ([.2, .2, .2], .001),
                                            ([1., .01, .002], .002/160)])
def test_spacing_fits_actual_padded_rounded_shape(size, spacing):
    half = np.asarray(size)/2
    fitted = kernel_ops._fit_spacing(half, spacing)
    assert math.prod(sdf.grid_shape(-half, half, fitted)) <= 1_200_000
    assert fitted >= spacing
    assert kernel_ops._fit_spacing(half, fitted) == fitted


def test_form_grid_coarsens_more_than_once_without_allocating_over_budget(monkeypatch):
    monkeypatch.setattr(form, "_MAX_SAMPLES", 10000)
    coords, origin, spacing = form._field_grid([0, 0, 0], [.2, .2, .2], .001)
    assert coords[0].size <= 10000
    assert spacing > .001
    assert origin == pytest.approx([-2.5*spacing]*3)


def test_direct_grid_budget_checked_before_meshgrid(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget grid was allocated")
    monkeypatch.setattr(np, "meshgrid", forbidden)
    with pytest.raises(AttaError) as exc:
        sdfkit.grid([0, 0, 0], [1, 1, 1], .001)
    assert exc.value.code == "sdf_too_dense"


def test_silhouette_height_fallback_is_metres():
    mesh, info = form.silhouette_solid({"front": {"outline": [[0, 0], [.02, 0], [.02, .02], [0, .02]]}},
                                      height=.01, resolution=16)
    assert mesh.topology()["closed"]
    assert mesh.bbox()["size"] == pytest.approx([.02, .01, .02], abs=.0003)
    assert info["samples"] <= form._MAX_SAMPLES


def test_repair_graph_exposes_native_simplification_and_scales_tolerance_once():
    mesh = evaluate([
        {"id": "ball", "op": "sphere", "args": {"radius": 10, "segments": 32, "rings": 16}},
        {"id": "out", "op": "repair", "inputs": ["ball"],
         "args": {"fill": False, "target_faces": 300, "max_error_mm": .1}},
    ])
    report = mesh.meta["decimation"]
    assert report["max_error"] == pytest.approx(.0001)
    assert report["method"] == "manifold_tolerance"
    assert mesh.topology()["closed"]
    assert mesh.bbox()["size"][0] == pytest.approx(.02, abs=.0003)


@pytest.mark.parametrize("args", [{"target_faces": 3}, {"target_faces": 4.5},
                                   {"target_faces": 10, "max_error_mm": -1},
                                   {"max_error_mm": .1}])
def test_repair_rejects_invalid_simplification_arguments(args):
    with pytest.raises(AttaError):
        kernel_ops._repair_prepare(args, .001, "features[0]")


def test_sdf_blend_automatic_simplification_records_target_and_spacing_tolerance(monkeypatch):
    monkeypatch.setattr(kernel_ops, "_FIELD_FACE_TARGET", 500)
    mesh = evaluate([{"id": "out", "op": "sdf_blend", "args": {
        "size": [30, 30, 30], "resolution": 20,
        "primitives": [{"kind": "sphere", "radius_mm": 10}],
    }}])
    automatic = mesh.meta["field_decimation"]
    report = mesh.meta["decimation"]
    assert automatic["automatic"] and automatic["target_faces"] == 500
    assert report["max_error"] == pytest.approx(mesh.meta["spacing_mm"] * .001 * .25)
    assert report["surface_error_bound"] <= report["max_error"]
    assert automatic["target_reached"] == report["target_reached"]
    assert mesh.topology()["closed"] and mesh.topology()["genus"] == 0


def test_sdf_blend_records_effective_sampling_in_millimetres():
    mesh = evaluate([{"id": "out", "op": "sdf_blend", "args": {
        "size": [30, 30, 30], "resolution": 12, "blend_mm": 1.25,
        "primitives": [{"kind": "sphere", "radius_mm": 10}],
    }}])
    assert mesh.topology()["closed"]
    assert mesh.meta["spacing_mm"] == pytest.approx(2.5)
    assert mesh.meta["blend_mm"] == pytest.approx(1.25)
    assert mesh.meta["sampling"]["samples"] <= 1_200_000
    assert not mesh.meta["sampling"]["coarsened"]
    assert mesh.bbox()["size"] == pytest.approx([.02, .02, .02], abs=.0004)
