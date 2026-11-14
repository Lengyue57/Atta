# -*- coding: utf-8 -*-
"""Independent analytic, numerical, graph and bounded-allocation NURBS tests."""
import math

import numpy as np
import pytest
from scipy.interpolate import BSpline

from atta import families
from atta.boolean import difference
from atta.errors import AttaError
from atta.features import FEATURE_OPS, evaluate
from atta.mesh import topology, volume
from atta.nurbs import NURBSSurface, tessellate
from atta.primitives import box
from atta.surfaces import thicken


def plane(scale=1.0, origin=(0, 0, 0)):
    return np.array([[[0, 0, 0], [0, 2, 0]], [[3, 0, 0], [3, 2, 0]]], dtype=float) * scale + origin


def cylinder(radius=0.03, height=0.05, **kwargs):
    cp = np.array([[[radius, 0, 0], [radius, 0, height]],
                   [[radius, radius, 0], [radius, radius, height]],
                   [[0, radius, 0], [0, radius, height]]])
    return NURBSSurface(cp, degree_u=2, degree_v=1,
                       weights=[[1, 1], [math.sqrt(0.5)] * 2, [1, 1]], **kwargs)


def assert_actionable(call, code=None):
    with pytest.raises(AttaError) as info:
        call()
    assert info.value.path and info.value.fix
    if code:
        assert info.value.code == code
    return info.value


def test_analytic_quarter_cylinder_rational_weights():
    s = cylinder()
    u = np.linspace(0, 1, 101)
    p = s.evaluate(u, 0.37)
    assert np.linalg.norm(p[:, :2], axis=1) == pytest.approx(np.full(101, 0.03), abs=1e-16)
    assert p[:, 2] == pytest.approx(np.full(101, 0.05 * 0.37), abs=1e-16)
    assert s.evaluate(0.5, 0.5) == pytest.approx([0.03 / math.sqrt(2)] * 2 + [0.025])
    normals = s.normal(u, 0.37)
    assert normals[:, :2] == pytest.approx(p[:, :2] / 0.03, abs=1e-14)
    assert normals[:, 2] == pytest.approx(np.zeros(101), abs=1e-14)
    polynomial = NURBSSurface(s.control_points, degree_u=2, degree_v=1)
    assert np.linalg.norm(polynomial.evaluate(0.5, 0.5)[:2]) > 0.03


def test_weight_global_rescaling_is_invariant():
    s = cylinder()
    for factor in (1e-100, 1e100):
        other = NURBSSurface(s.control_points, degree_u=2, degree_v=1, weights=s.weights * factor)
        assert other.evaluate(0.37, 0.49) == pytest.approx(s.evaluate(0.37, 0.49), abs=1e-16)
        assert other.derivatives(0.37, 0.49) == pytest.approx(np.asarray(s.derivatives(0.37, 0.49)), abs=1e-15)


def test_derivatives_against_central_difference_and_knot_domain_scale():
    s = cylinder(knots_u=[2, 2, 2, 5, 5, 5], knots_v=[-4, -4, 6, 6])
    for u, v in ((2.2, -3.0), (3.4, 2.5), (4.9, 5.8)):
        h = 1e-5
        du, dv = s.derivatives(u, v)
        assert du == pytest.approx((s.evaluate(u + h, v) - s.evaluate(u - h, v)) / (2 * h), abs=2e-12)
        assert dv == pytest.approx((s.evaluate(u, v + h) - s.evaluate(u, v - h)) / (2 * h), abs=2e-12)
    baseline = cylinder()
    du, dv = s.derivatives(3.5, 1)
    a, b = baseline.derivatives(0.5, 0.5)
    assert du == pytest.approx(a / 3)
    assert dv == pytest.approx(b / 10)
    # Endpoint derivatives use the interior one-sided limit.
    du, _ = s.derivatives(5, 6)
    assert du == pytest.approx((s.evaluate(5, 6) - s.evaluate(5 - 1e-6, 6)) / 1e-6, abs=2e-8)


def test_nonuniform_knots_against_independent_scipy_tensor_oracle():
    rng = np.random.default_rng(712)
    cp = rng.normal(size=(5, 4, 3)) * 0.02
    w = rng.uniform(0.2, 2, size=(5, 4))
    ku = [2, 2, 2, 2.2, 3.7, 5, 5, 5]
    kv = [-3, -3, -3, -0.7, 1, 1, 1]
    s = NURBSSurface(cp, weights=w, degree_u=2, degree_v=2, knots_u=ku, knots_v=kv)
    a = BSpline(ku, np.eye(5), 2)
    b = BSpline(kv, np.eye(4), 2)
    for u, v in ((2, -3), (2.2, -0.7), (3.7, 0), (4, -2), (5, 1)):
        coeff = a(u)[:, None] * b(v)[None, :] * w
        p = np.einsum('ij,ijk->k', coeff, cp) / coeff.sum()
        assert s.evaluate(u, v) == pytest.approx(p, abs=5e-17)
        derivs = []
        for c in (a(u, 1)[:, None] * b(v)[None, :] * w,
                  a(u)[:, None] * b(v, 1)[None, :] * w):
            derivs.append((np.einsum('ij,ijk->k', c, cp) - p * c.sum()) / coeff.sum())
        actual = np.asarray(s.derivatives(u, v))
        assert actual == pytest.approx(np.array(derivs), abs=1e-15)


def test_unclamped_knots_and_repeated_final_active_knot():
    cp = np.zeros((5, 2, 3))
    cp[:, :, 0] = np.arange(5)[:, None]
    cp[:, 1, 1] = 2
    ku = [-1, 0, 1, 2, 3, 3, 4, 5]
    s = NURBSSurface(cp, degree_u=2, degree_v=1, knots_u=ku)
    oracle = BSpline(ku, cp, 2)
    for u in (1, 1.5, 2, 2.9):
        assert s.evaluate(u, 0.25) == pytest.approx(oracle(u)[0] * 0.75 + oracle(u)[1] * 0.25)
    assert s.evaluate(3, 0.25) == pytest.approx(s.evaluate(3 - 1e-10, 0.25), abs=2e-9)
    du, _ = s.derivatives(3, 0.25)
    assert du == pytest.approx((s.evaluate(3, 0.25) - s.evaluate(3 - 1e-7, 0.25)) / 1e-7, abs=2e-7)


def test_c0_knot_right_derivative_and_conforming_span_boundaries():
    cp = np.zeros((5, 2, 3))
    cp[:, :, 0] = np.arange(5)[:, None]
    cp[:, 1, 1] = 1
    cp[3:, :, 2] = np.array([1, 2])[:, None]
    s = NURBSSurface(cp, degree_u=2, degree_v=1, knots_u=[0, 0, 0, .3, .3, 1, 1, 1])
    du, _ = s.derivatives(.3, .4)
    assert du == pytest.approx((s.evaluate(.3 + 1e-7, .4) - s.evaluate(.3, .4)) / 1e-7, abs=1e-7)
    m = tessellate(s, tolerance=.5, u_samples=2, v_samples=2)
    info = topology(m)
    nx, ny = m.meta['nurbs']['grid_shape']
    assert info['boundary_edges'] == 2 * (nx - 1) + 2 * (ny - 1)
    assert info['non_manifold_edges'] == info['inconsistent_winding_edges'] == 0
    assert nx >= 3  # Interior knot boundary is always in the tensor grid.


def test_broadcast_and_empty_evaluation():
    s = NURBSSurface(plane())
    p = s.evaluate(np.array([0, .5, 1])[:, None], np.array([0, .5, 1, .75])[None, :])
    assert p.shape == (3, 4, 3)
    assert p[1, 3] == pytest.approx([1.5, 1.5, 0])
    assert s.evaluate([], []).shape == (0, 3)
    assert_actionable(lambda: s.evaluate(np.zeros(3), np.zeros(4)))
    assert_actionable(lambda: s.evaluate(np.zeros((10, 1)), np.zeros((1, 10)), max_evaluations=99), 'nurbs_budget')


@pytest.mark.parametrize('scale,origin', [(1e-12, (0, 0, 0)), (1e-7, (0, 0, 0)),
                                         (1e-2, (1e6, -1e6, 2e6))])
def test_large_coordinates_and_microscopic_evaluation(scale, origin):
    s = NURBSSurface(plane(scale, origin))
    assert s.evaluate(.4, .7) == pytest.approx(np.array(origin) + np.array([1.2, 1.4, 0]) * scale,
                                             abs=max(scale * 1e-13, 5e-10 if origin[0] else 0))
    du, dv = s.derivatives(.4, .7)
    assert du == pytest.approx([3 * scale, 0, 0], rel=2e-8, abs=scale * 1e-12)
    assert dv == pytest.approx([0, 2 * scale, 0], rel=2e-8, abs=scale * 1e-12)
    assert s.normal(.4, .7) == pytest.approx([0, 0, 1], abs=1e-14)
    m = tessellate(s, tolerance=scale * 1e-6)
    assert len(m.verts) == 9
    assert m.meta['nurbs']['target_met']


def test_large_translation_error_includes_representable_output_roundoff():
    cp = plane(.01, (1e6, -1e6, 0))
    s = NURBSSurface(cp)
    m = tessellate(s, tolerance=1e-16, u_samples=4, v_samples=4,
                   max_refinements=0, fail_on_tolerance=False)
    assert not m.meta['nurbs']['target_met']
    assert m.meta['nurbs']['max_vertex_roundoff'] > 0
    assert m.meta['nurbs']['sampled_error'] > 1e-16


def test_adaptive_error_refines_and_never_claims_global_bound():
    s = cylinder()
    coarse = tessellate(s, tolerance=.01)
    fine = tessellate(s, tolerance=1e-5)
    c, f = coarse.meta['nurbs'], fine.meta['nurbs']
    assert len(fine.faces) > len(coarse.faces)
    assert f['sampled_error'] < c['sampled_error']
    assert f['sampled_error'] <= f['tolerance'] and f['target_met']
    assert f['surface_error_bound'] is None and not f['exact']
    nx, ny = f['grid_shape']
    topo = topology(fine)
    assert not topo['closed'] and topo['n_components'] == 1
    assert topo['boundary_edges'] == 2 * (nx + ny - 2)
    assert topo['non_manifold_edges'] == topo['inconsistent_winding_edges'] == 0
    assert fine.hash() == tessellate(s, tolerance=1e-5).hash()


@pytest.mark.parametrize('budget,expected', [({'max_vertices':9}, 'max_vertices'),
                                             ({'max_faces':8}, 'max_faces'),
                                             ({'max_evaluations':45}, 'max_evaluations'),
                                             ({'max_refinements':0}, 'max_refinements')])
def test_budget_stops_default_raise_and_explicit_degradation(budget, expected):
    s = cylinder()
    exc = assert_actionable(lambda: tessellate(s, tolerance=1e-10, **budget), 'nurbs_tolerance_unmet')
    assert exc.details['stop_reason'] == expected
    m = tessellate(s, tolerance=1e-10, fail_on_tolerance=False, **budget)
    assert not m.meta['nurbs']['target_met']
    assert m.meta['nurbs']['stop_reason'] == expected
    assert m.meta['issues'][0]['code'] == 'nurbs_tolerance_unmet'
    assert m.meta['nurbs']['evaluations'] <= m.meta['nurbs']['max_evaluations']


def test_initial_grid_budget_fails_before_evaluation(monkeypatch):
    s = cylinder()
    def forbidden(*a, **kw):
        pytest.fail('evaluation must not run before grid budget validation')
    monkeypatch.setattr(s, '_evaluate_normalized', forbidden)
    for options in ({'u_samples':1025, 'v_samples':1025}, {'max_vertices':4},
                    {'max_evaluations':1}, {'max_faces':2}):
        assert_actionable(lambda: tessellate(s, **options), 'nurbs_budget')


def test_control_input_budget_precedes_dense_conversion():
    huge = np.broadcast_to(np.zeros((1, 1, 3)), (1000000, 1000000, 3))
    assert_actionable(lambda: NURBSSurface(huge), 'nurbs_budget')
    assert_actionable(lambda: NURBSSurface(np.zeros((1025, 2, 3))), 'nurbs_budget')


@pytest.mark.parametrize('options', [
    {'degree_u':True}, {'degree_v':1.2}, {'degree_u':2}, {'degree_u':10},
    {'weights':[[1, -1], [1, 1]]}, {'weights':[[1, 0], [1, 1]]},
    {'weights':[[1, float('nan')], [1, 1]]}, {'weights':[[1e-13, 1], [1, 1]]},
    {'weights':[[1, 1]]}, {'knots_u':[0, 0, 1]}, {'knots_u':[0, 1, 0, 1]},
    {'knots_u':[0, 0, 0, 0]}, {'knots_u':[0, 0, 1, float('inf')]},
])
def test_invalid_surface_arguments_are_actionable(options):
    assert_actionable(lambda: NURBSSurface(plane(), **options))


@pytest.mark.parametrize('cp', [None, [], [[[0,0,0]]], [[[0,0],[1,2]]],
                               [[[0,0,0]],[[0,0,0],[1,1,1]]],
                               [[['0',0,0],[0,1,0]],[[1,0,0],[1,1,0]]],
                               np.zeros((2,2,3)), np.full((2,2,3), np.nan)])
def test_invalid_control_nets_are_actionable(cp):
    assert_actionable(lambda: NURBSSurface(cp))


def test_discontinuous_and_unresolved_knot_spans_are_rejected():
    cp = np.zeros((4,2,3)); cp[:, :, 0] = np.arange(4)[:,None]; cp[:,1,1]=1
    assert_actionable(lambda: NURBSSurface(cp, degree_u=1, knots_u=[0,0,.5,.5,1,1]))
    assert_actionable(lambda: NURBSSurface(cp, degree_u=1, knots_u=[0,0,1e-16,.5,1,1]))


@pytest.mark.parametrize('options', [{'tolerance':0}, {'tolerance':float('nan')}, {'tolerance':True},
                                    {'max_vertices':1e30}, {'max_faces':True}, {'max_evaluations':0},
                                    {'max_refinements':17}, {'u_samples':1}, {'v_samples':2.5},
                                    {'fail_on_tolerance':'false'}, {'flip':1}])
def test_bad_tessellation_options_are_actionable(options):
    assert_actionable(lambda: tessellate(NURBSSurface(plane()), **options))


@pytest.mark.parametrize('u,v', [(float('nan'),0), (0,float('inf')), (-1,0), (0,1.01), (True,0), ('0',0)])
def test_bad_evaluation_parameters(u, v):
    assert_actionable(lambda: NURBSSurface(plane()).evaluate(u,v))


def test_degenerate_normal_and_mesh_fail_without_fallback():
    cp = plane(); cp[:, :, 1] = 0
    s = NURBSSurface(cp)
    assert_actionable(lambda: s.normal(.5,.5), 'nurbs_singular')
    assert_actionable(lambda: tessellate(s), 'nurbs_singular')


def test_graph_registration_units_and_closed_solid_boolean():
    families.load(strict=True)
    assert 'nurbs_surface' in FEATURE_OPS
    mm = plane(.01) * 1000
    for units, points, thickness in [('mm',mm,2), ('cm',mm/10,.2), ('m',mm/1000,.002)]:
        graph = [{'id':'s','op':'nurbs_surface','args':{'control_points':points.tolist()}},
                 {'id':'b','op':'thicken','inputs':['s'],'args':{'thickness':thickness}}]
        m = evaluate(graph, units=units)
        assert m.bbox()['size'] == pytest.approx([.03,.02,.002])
        assert volume(m) == pytest.approx(1200e-9)
        assert topology(m)['closed'] and topology(m)['genus'] == 0
        assert m.meta['nurbs']['tolerance'] == pytest.approx(.05 * {'mm':.001,'cm':.01,'m':1}[units])
    cutter = box([.006,.006,.01], center=(.015,.01,0))
    cut = difference(m, cutter)
    assert topology(cut)['closed'] and topology(cut)['genus'] == 1
    assert volume(cut) == pytest.approx((1200-72)*1e-9, rel=1e-5)


def test_graph_rational_arguments_and_single_unit_conversion():
    families.load(strict=True)
    s=cylinder()
    mesh=evaluate([{'id':'skin','op':'nurbs_surface','args':{
        'control_points':(s.control_points*1000).tolist(), 'weights':s.weights.tolist(),
        'degree_u':2,'degree_v':1,'knots_u':[2,2,2,5,5,5],'knots_v':[0,0,1,1],
        'tolerance':.05}}])
    assert np.linalg.norm(mesh.verts[:,:2],axis=1) == pytest.approx(np.full(len(mesh.verts),.03),abs=1e-15)
    assert mesh.meta['nurbs']['tolerance'] == 5e-5
    assert mesh.meta['domain_u'] == [2,5]
    assert_actionable(lambda: evaluate([{'id':'s','op':'nurbs_surface','args':{'control_points':[[None]]}}]))


@pytest.mark.parametrize('scale,origin', [(1e-7,(0,0,0)),(1e-12,(0,0,0)),(.01,(1e6,1e6,1e6))])
def test_nurbs_thicken_vertical_small_scale_and_large_translation(scale, origin):
    cp=plane(scale)
    cp=cp[:,:,[2,0,1]] + origin  # yz patch, analytic normal +X
    s=NURBSSurface(cp)
    mesh=tessellate(s,tolerance=scale*1e-6)
    solid=thicken(mesh,scale/10)
    assert topology(solid)['closed']
    assert solid.bbox()['size'] == pytest.approx([scale/10,3*scale,2*scale],rel=1e-6)
    assert volume(solid) == pytest.approx(.6*scale**3,rel=1e-6)
    assert solid.meta['nurbs']['target_met']


def test_quarter_cylinder_shell_volume_and_flip():
    s=cylinder()
    patch=tessellate(s,tolerance=1e-5)
    shell=thicken(patch,.002)
    assert topology(shell)['closed']
    assert volume(shell) == pytest.approx(math.pi/2*.03*.05*.002,rel=.002)
    flipped=tessellate(s,tolerance=1e-5,flip=True)
    assert flipped.verts == pytest.approx(patch.verts)
    assert np.array_equal(flipped.faces,patch.faces[:,::-1])
    assert volume(thicken(flipped,.002)) == pytest.approx(volume(shell),rel=1e-12)



@pytest.mark.parametrize('degree', [1, 3, 8])
def test_supported_degree_extremes_against_scipy(degree):
    rng = np.random.default_rng(529 + degree)
    nu, nv = degree + 4, degree + 3
    cp = rng.normal(size=(nu, nv, 3))
    weights = rng.uniform(.1, 2, size=(nu, nv))
    ku = np.r_[np.zeros(degree + 1), np.sort(rng.uniform(.05, .95, nu - degree - 1)), np.ones(degree + 1)]
    kv = np.r_[np.zeros(degree + 1), np.sort(rng.uniform(.05, .95, nv - degree - 1)), np.ones(degree + 1)]
    s = NURBSSurface(cp, weights=weights, degree_u=degree, degree_v=degree, knots_u=ku, knots_v=kv)
    a, b = BSpline(ku, np.eye(nu), degree), BSpline(kv, np.eye(nv), degree)
    for u, v in [(0, 0), (1, 1), (.37, .61)]:
        coeff = a(u)[:, None] * b(v)[None, :] * weights
        pos = np.einsum('ij,ijk->k', coeff, cp) / coeff.sum()
        assert s.evaluate(u, v) == pytest.approx(pos, abs=2e-14)
        for dc, got in zip((a(u, 1)[:, None] * b(v)[None, :] * weights,
                            a(u)[:, None] * b(v, 1)[None, :] * weights), s.derivatives(u, v)):
            expected = (np.einsum('ij,ijk->k', dc, cp) - pos * dc.sum()) / coeff.sum()
            assert got == pytest.approx(expected, abs=5e-12)


def test_documented_spec_validates_and_builds_closed_shell():
    import json
    from pathlib import Path
    from atta.spec import validate_spec
    text = (Path(__file__).parent.parent / 'docs' / 'NURBS.md').read_text()
    doc = json.loads(text.split('```json\n', 1)[1].split('\n```', 1)[0])
    families.load(strict=True)
    validated = validate_spec(doc)
    assert validated['ok'], validated['errors']
    graph = validated['spec']['model']
    out = evaluate(graph['features'], units=graph['units'], output=graph['output'])
    assert topology(out)['closed']
    assert volume(out) == pytest.approx(math.pi / 2 * .03 * .05 * .002, rel=.002)


def test_degraded_metadata_survives_thicken():
    patch=tessellate(cylinder(),tolerance=1e-12,max_refinements=0,fail_on_tolerance=False)
    shell=thicken(patch,.001)
    assert not shell.meta['nurbs']['target_met']
    assert shell.meta['issues'][0]['code']=='nurbs_tolerance_unmet'
