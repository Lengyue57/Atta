# -*- coding: utf-8 -*-
"""shape_grammar / lsystem graph ops: closure, topology, determinism,
symmetry counts, budgets that fail closed, and canonical examples."""
from __future__ import annotations

import os

import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import families  # noqa: E402

families.load()

from atta import features, verify  # noqa: E402
from atta.errors import AttaError  # noqa: E402
from atta.families import grammar_ops as G  # noqa: E402
from atta.mesh import canonical_hash  # noqa: E402
from atta.spec import validate_spec  # noqa: E402


def grammar(**args):
    return features.evaluate([{"id": "g", "op": "shape_grammar", "args": args}])


def lsys(**args):
    return features.evaluate([{"id": "t", "op": "lsystem", "args": args}])


def raises(code, fn, *a, **kw):
    with pytest.raises(AttaError) as ei:
        fn(*a, **kw)
    err = ei.value
    assert err.code == code, (err.code, err.message)
    assert err.path and err.fix, err.to_dict() if hasattr(err, "to_dict") else err
    return err


def solid_ok(mesh, components=None, genus=None):
    topo = mesh.topology()
    assert topo["closed"], topo
    assert topo["degenerate_faces"] == 0, topo
    assert mesh.volume() > 0
    if components is not None:
        assert topo["n_components"] == components, topo
    if genus is not None:
        assert topo["genus"] == genus, topo
    return topo


# A star: one bar per symmetry copy, all crossing the centre -> 1 component, genus 0.
BAR = {"B": [{"shape": "box", "transform": {"x": 2.0, "s": [4.0, 0.6, 0.6]}}]}


# ------------------------------------------------------------------ registry

def test_ops_are_registered_with_catalog_schema():
    for op, required in (("shape_grammar", ("startshape", "rules")),
                         ("lsystem", ("axiom", "rules", "iterations"))):
        spec = features.FEATURE_OPS[op]
        assert spec["required"] == required
        assert spec["arity"] == 0 and spec["doc"]
        kinds = features.arg_kinds(op)
        assert set(kinds) <= set(spec["required"]) | set(spec["optional"])
        assert op in features.extra_ops()


def test_spec_validation_accepts_graph_and_rejects_unknown_arg():
    spec = {"schema": "atta.spec/v3", "name": "g", "model": {
        "kind": "graph", "units": "mm", "output": "g", "features": [
            {"id": "g", "op": "shape_grammar",
             "args": {"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "unit": 5}}]}}
    assert validate_spec(spec)["ok"]
    spec["model"]["features"][0]["args"]["depth_policy"] = "sometimes"
    assert not validate_spec(spec)["ok"]


# ------------------------------------------------------------------ shape_grammar basics

def test_single_box_is_mm_in_metres_out():
    m = grammar(startshape="S", rules={"S": [{"shape": "box"}]}, unit=10)
    solid_ok(m, components=1, genus=0)
    assert m.bbox()["size"] == pytest.approx([0.01, 0.01, 0.01], abs=1e-9)
    assert m.volume() == pytest.approx(1e-6, rel=1e-9)
    cm = features.evaluate([{"id": "g", "op": "shape_grammar", "args": {
        "startshape": "S", "rules": {"S": [{"shape": "box"}]}, "unit": 1}}], units="cm")
    assert cm.bbox()["size"][0] == pytest.approx(0.01)


def test_every_primitive_builds_closed():
    for name in G.PRIMITIVES:
        m = grammar(startshape="S", rules={"S": [{"shape": name}]}, unit=10, segments=16)
        topo = solid_ok(m, components=1)
        assert topo["genus"] == (1 if name == "torus" else 0), name


def test_transform_order_dict_is_canonical_list_is_ordered():
    # dict: translate then rotate (CFDG [ ]) -> box stays at x=10
    a = grammar(startshape="S", unit=1, rules={"S": [{"shape": "box", "transform": {"r": 90, "x": 10}}]})
    assert a.centroid()[0] == pytest.approx(0.010, abs=1e-9)
    # list: rotate first, then translate along the rotated x (now +y)
    b = grammar(startshape="S", unit=1, rules={"S": [{"shape": "box",
                                                       "transform": [{"r": 90}, {"x": 10}]}]})
    assert b.centroid()[1] == pytest.approx(0.010, abs=1e-9)


def test_loop_and_flip_and_colour_keys():
    m = grammar(startshape="S", unit=1, segments=12, rules={"S": [{
        "shape": "box", "loop": {"count": 12, "transform": {"r": 30}},
        "transform": {"x": 3, "s": [6, 1, 1], "hue": 120, "sat": 0.5}}]})
    solid_ok(m, components=1, genus=0)
    meta = m.meta["shape_grammar"]
    assert meta["instances"] == 12 and meta["ignored_keys"] == ["hue", "sat"]
    # Ordered list [{"flip": 90}, {"x": 5}] flips across the y axis first -> -0.005 m
    f = grammar(startshape="S", unit=1, rules={"S": [{
        "shape": "box", "transform": [{"flip": 90}, {"x": 5}]}]})
    assert f.centroid()[0] == pytest.approx(-0.005, abs=1e-9)
    # Canonical dict {"x": 5, "flip": 90} (CFDG [ ]) translates to +5 then flips child frame -> +5 - 2 = +0.003 m
    c = grammar(startshape="S", unit=1, rules={
        "S": [{"call": "C", "transform": {"x": 5, "flip": 90}}],
        "C": [{"shape": "box", "transform": {"x": 2}}]})
    assert c.centroid()[0] == pytest.approx(0.003, abs=1e-9)


def test_cfdg_like_dihedral_ornament_and_quadtree():
    """Ports of CF::Dihedral ornament (D6 + odd D13) and 2D quadtree facade."""
    m = grammar(startshape="SHAPE1", unit=4, seed=7, segments=12, symmetry="CF::Dihedral, 6",
                depth_policy="prune", max_depth=40, max_components=1, rules={
                    "SHAPE1": [{"call": "SHAPE3", "transform": {"x": 0.5}}],
                    "SHAPE3": [
                        {"weight": 1.0, "body": [{"shape": "square", "transform": {"s": [1.2, 0.4]}},
                                                 {"call": "SHAPE3", "transform": {"x": 0.9, "r": 4, "s": 0.97}}]},
                        {"weight": 0.1, "body": [{"call": "SHAPE3", "transform": {"flip": 1}}]},
                        {"weight": 0.05, "body": []}]})
    topo = solid_ok(m, components=1)
    assert verify.verify(m)["verdict"] == "pass"
    meta = m.meta["shape_grammar"]
    assert meta["symmetry_order"] == 12
    assert meta["instances"] % 12 == 0 and meta["instances"] >= 24
    assert isinstance(topo["genus"], int) and topo["genus"] >= 0

    # Odd-order dihedral rosette with central hub -> genus 0, 1 component
    d13 = grammar(startshape="R", unit=3, symmetry="CF::Dihedral, 13", max_components=1, rules={
        "R": [{"shape": "cylinder", "transform": {"s": [2.0, 2.0, 0.5]}},
              {"call": "ARM", "transform": {"x": 0.8}}],
        "ARM": [{"weight": 1.0, "max_depth": 5,
                 "body": [{"shape": "box", "transform": {"s": [1.1, 0.22, 0.4]}},
                          {"call": "ARM", "transform": {"x": 0.85, "r": 1.5, "s": 0.92}}]}]})
    solid_ok(d13, components=1, genus=0)
    assert verify.verify(d13, expected_genus=0)["verdict"] == "pass"
    assert d13.meta["shape_grammar"]["symmetry_order"] == 26

    # 2D quadtree with s: [0.5, 0.5] terminates on min_scale via minimum singular value
    quad = grammar(startshape="Q", unit=20, min_scale=0.2, max_components=1, rules={
        "Q": [{"shape": "square", "loop": {"count": 4, "transform": {"r": 90}},
               "transform": {"x": 0.25, "s": [0.6, 0.35]}},
              {"call": "Q", "transform": {"x": 0.38, "y": 0.38, "s": [0.5, 0.5], "flip": 0}},
              {"call": "Q", "transform": {"x": -0.38, "y": 0.38, "s": [0.5, 0.5], "flip": 90}},
              {"call": "Q", "transform": {"x": 0.38, "y": -0.38, "s": [0.5, 0.5], "flip": 90}},
              {"call": "Q", "transform": {"x": -0.38, "y": -0.38, "s": [0.5, 0.5], "flip": 0}}]})
    solid_ok(quad, components=1)
    assert quad.meta["shape_grammar"]["pruned_scale"] > 0

    # Cyclic ring of overlapping chords around an open centre -> exact genus 1
    ring = grammar(startshape="S", unit=5, symmetry={"type": "cyclic", "n": 8}, rules={
        "S": [{"shape": "box", "transform": {"y": 2.0, "s": [2.2, 0.7, 0.7]}}]})
    solid_ok(ring, components=1, genus=1)


@pytest.mark.parametrize("sym,order", [
    ({"type": "cyclic", "n": 5}, 5),
    ({"type": "dihedral", "n": 4}, 8),
    ("CF::Cyclic, 3", 3),
    ({"type": "reflect", "axes": ["x", "y"]}, 4),
    # C2 = (reflect x)(reflect y): the product group is deduplicated to order 4
    ([{"type": "cyclic", "n": 2}, {"type": "reflect", "axes": ["x", "y"]}], 4),
])
def test_symmetry_group_counts(sym, order):
    m = grammar(startshape="B", unit=1, rules=BAR, symmetry=sym, max_components=1)
    meta = m.meta["shape_grammar"]
    assert meta["symmetry_order"] == order
    # a bar on the x axis is invariant under reflection across x, so dedupe may drop copies
    assert meta["instances"] + meta["duplicates"] == order
    solid_ok(m, components=1, genus=0)


def test_cyclic_star_is_genus_zero_and_rotationally_symmetric():
    m = grammar(startshape="B", unit=1, rules=BAR, symmetry={"type": "cyclic", "n": 6})
    solid_ok(m, components=1, genus=0)
    assert m.meta["shape_grammar"]["instances"] == 6
    bb = m.bbox()["size"]
    assert bb[0] == pytest.approx(0.008, rel=1e-3)  # 2 * (2 + 4/2) mm


def test_symmetry_center_and_axis():
    m = grammar(startshape="S", unit=1, rules={"S": [{"shape": "sphere", "transform": {"x": 5}}]},
                symmetry={"type": "reflect", "axis": "x", "center": [10, 0, 0]})
    assert m.meta["shape_grammar"]["instances"] == 2
    assert m.centroid()[0] == pytest.approx(0.010, abs=1e-6)
    y = grammar(startshape="S", unit=1, rules={"S": [{"shape": "sphere", "transform": {"z": 5}}]},
                symmetry={"type": "cyclic", "n": 4, "axis": "y"})
    assert y.meta["shape_grammar"]["instances"] == 4
    assert y.bbox()["size"][1] == pytest.approx(0.001, rel=1e-2)  # all copies in the xz plane


# ------------------------------------------------------------------ determinism

STOCHASTIC = {"startshape": "T", "unit": 2, "segments": 8, "depth_policy": "prune", "max_depth": 60,
              "rules": {"T": [{"weight": 1, "body": [{"shape": "box", "transform": {"s": [0.4, 1.2, 0.4]}},
                                                     {"call": "T", "transform": {"y": 1, "s": 0.98}}]},
                              {"weight": 0.15, "body": [{"call": "T", "transform": {"r": {"rand": [-40, 40]}}}]},
                              {"weight": 0.05, "body": []}]}}


def test_same_spec_and_seed_same_geometry_hash():
    a = grammar(seed=11, **STOCHASTIC)
    b = grammar(seed=11, **STOCHASTIC)
    assert canonical_hash(a) == canonical_hash(b)
    assert a.meta["shape_grammar"]["instances"] == b.meta["shape_grammar"]["instances"]
    hashes = {canonical_hash(grammar(seed=s, **STOCHASTIC)) for s in range(1, 6)}
    assert len(hashes) > 1, "seed must drive the weighted choice"


def test_rand_values_stay_in_range():
    g = G.compile_shape_grammar({"startshape": "S", "seed": 4, "rules": {"S": [{
        "shape": "box", "loop": {"count": 50}, "transform": {"x": {"rand": [2, 3]}}}]}})
    inst, _ = G.expand_shape_grammar(g)
    xs = [m[0, 3] for _, m in inst]
    assert len(xs) == 50 and min(xs) >= 2 and max(xs) <= 3 and len(set(xs)) == 50


# ------------------------------------------------------------------ termination + budgets

def test_min_scale_terminates_self_similar_recursion():
    m = grammar(startshape="R", unit=10, min_scale=0.05, rules={"R": [
        {"shape": "box"}, {"call": "R", "transform": {"x": 0.7, "s": 0.5}}]})
    solid_ok(m, components=1, genus=0)
    meta = m.meta["shape_grammar"]
    assert meta["instances"] == 5  # 1, .5, .25, .125, .0625 ; .03125 < 0.05 pruned
    assert meta["pruned_scale"] == 1


def test_unbounded_recursion_fails_closed_by_default():
    rules = {"R": [{"shape": "box"}, {"call": "R", "transform": {"y": 0.9}}]}
    err = raises("grammar_depth_budget", grammar, startshape="R", rules=rules, max_depth=20)
    assert err.path.endswith("args.max_depth") and "prune" in err.fix
    m = grammar(startshape="R", rules=rules, max_depth=20, depth_policy="prune")
    assert m.meta["shape_grammar"]["instances"] == 21
    solid_ok(m, components=1, genus=0)


def test_expansion_instance_triangle_and_component_budgets():
    binary = {"R": [{"call": "R", "transform": {"x": 1}}, {"call": "R", "transform": {"y": 1}}]}
    raises("grammar_expansion_budget", grammar, startshape="R", rules=binary,
           max_depth=10000, max_expansions=5000)
    raises("grammar_instance_budget", grammar, startshape="S", max_instances=10,
           rules={"S": [{"shape": "box", "loop": {"count": 11, "transform": {"x": 0.5}}}]})
    raises("grammar_instance_budget", grammar, startshape="B", rules=BAR, max_instances=4,
           symmetry={"type": "cyclic", "n": 5})
    raises("grammar_triangle_budget", grammar, startshape="S", max_triangles=1000, segments=64,
           rules={"S": [{"shape": "sphere"}]})
    raises("grammar_components", grammar, startshape="S", max_components=1,
           rules={"S": [{"shape": "box"}, {"shape": "box", "transform": {"x": 5}}]})
    raises("grammar_budget_ceiling", grammar, startshape="S", rules={"S": [{"shape": "box"}]},
           max_instances=10 ** 9)
    raises("grammar_empty", grammar, startshape="S", rules={"S": [{"weight": 1, "body": []}]})


@pytest.mark.parametrize("args,code,where", [
    ({"startshape": "X", "rules": {"S": [{"shape": "box"}]}}, "bad_arg", "startshape"),
    ({"startshape": "S", "rules": {"S": [{"call": "NOPE"}]}}, "bad_arg", "call"),
    ({"startshape": "S", "rules": {"S": [{"shape": "teapot"}]}}, "bad_arg", "shape"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"skewz": 1}}]}}, "bad_arg", "skewz"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"x": float("nan")}}]}}, "bad_arg", "x"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"y": float("inf")}}]}}, "bad_arg", "y"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"s": 0}}]}}, "bad_arg", "s"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"s": [1, 0, 1]}}]}}, "bad_arg", "s[1]"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"s": -2}}]}}, "bad_arg", "s"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box", "transform": {"s": {"rand": [0, 1]}}}]}}, "bad_arg", "rand"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "min_scale": 0}, "bad_arg", "min_scale"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "tile_thickness": 0}, "bad_arg", "tile_thickness"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "symmetry": "CF::Dihedral, 4, 2, 1"},
     "bad_arg", "symmetry"),
    ({"startshape": "S", "rules": {"S": [{"weight": 0, "body": [{"shape": "box"}]}]}}, "bad_arg", "S"),
    ({"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "unit": 0}, "bad_arg", "unit"),
    ({"startshape": "R", "depth_policy": "prune", "max_depth": 40,
      "rules": {"R": [{"shape": "box"}, {"call": "R", "transform": {"s": 10.0}}]}}, "grammar_overflow", "rules.R"),
])
def test_bad_grammar_input_errors_carry_path_and_fix(args, code, where):
    err = raises(code, grammar, **args)
    assert where in err.path and err.path.startswith("features[0]"), err.path


def test_structure_synth_max_depth_successor():
    # rule R maxdepth 3 > T : three boxes, then the successor caps with a sphere
    m = grammar(startshape="R", unit=5, rules={
        "R": [{"weight": 1, "max_depth": 3, "then": "T",
               "body": [{"shape": "box"}, {"call": "R", "transform": {"x": 0.8}}]}],
        "T": [{"shape": "sphere", "transform": {"s": 1.4}}]})
    solid_ok(m, components=1, genus=0)
    meta = m.meta["shape_grammar"]
    assert meta["instances"] == 4 and meta["rule_counts"] == {"R": 4, "T": 1}


def test_params_resolve_inside_rules():
    rows = [{"id": "g", "op": "shape_grammar", "args": {
        "startshape": "S", "unit": 1, "rules": {"S": [{"shape": "box", "loop": {
            "count": {"$param": "n"}, "transform": {"r": {"$expr": "360 / n"}}},
            "transform": {"x": 2, "s": [4, 1, 1]}}]}}}]
    m = features.evaluate(rows, params={"n": 8})
    assert m.meta["shape_grammar"]["instances"] == 8
    solid_ok(m, components=1, genus=0)


# ------------------------------------------------------------------ lsystem

def test_algae_rewriting_is_fibonacci():
    g = G.compile_lsystem({"axiom": "A", "rules": {"A": "AB", "B": "A"}, "iterations": 7})
    out = "".join(s for s, _ in G.expand_lsystem(g))
    assert len(out) == 34 and out.startswith("ABAABABAABAAB")


def test_koch_curve_chain_is_one_genus_zero_tube():
    m = lsys(axiom="F", rules={"F": "F+F--F+F"}, iterations=2, angle=60, step=10, width=2)
    solid_ok(m, components=1, genus=0)
    assert m.meta["lsystem"]["segments"] == 16
    # a Koch curve of generation 2 spans step * 9 between its end points
    assert max(m.bbox()["size"]) == pytest.approx(0.090 + 0.002, rel=0.02)


def test_abop_tree_is_closed_single_component():
    """ABOP fig. 1.25-style 3D tree with leaves (Prusinkiewicz & Lindenmayer 1990)."""
    m = lsys(axiom="A", iterations=4, angle=22.5, step=12, width=3, width_decay=0.7,
             rules={"A": "!F[&FL!A]/////[&FL!A]///////[&FL!A]"},
             leaf={"symbol": "L", "shape": "sphere", "size": [4, 1.5, 6]}, max_components=1)
    topo = solid_ok(m, components=1)
    meta = m.meta["lsystem"]
    # every rewritten A emits 3 leaves and 4 segments (!F + 3 bracketed F);
    # 1 + 3 + 9 + 27 = 40 A's are rewritten in 4 iterations, the last 81 draw nothing.
    assert meta["leaves"] == 120
    assert meta["segments"] == 160
    # Root A starts with !F, so the base joint has diameter 3 * 0.7 = 2.1 mm -> z_min = -1.05 mm.
    assert m.bbox()["min"][2] == pytest.approx(-0.00105, abs=1e-5)
    assert verify.verify(m)["verdict"] == "pass"
    assert isinstance(topo["genus"], int)


def test_parametric_conditional_rule_terminates():
    m = lsys(axiom="A(40,4)", iterations=10, angle=30, constants={"r": 0.6},
             rules=[{"pred": "A(l,w)", "cond": "l >= 10", "succ": "!(w)F(l)[+A(l*r,w*0.7)][-A(l*r,w*0.7)]"}])
    meta = m.meta["lsystem"]
    # l: 40, 24, 14.4 draw (>=10); 8.64 stops -> 1 + 2 + 4 segments
    assert meta["segments"] == 7
    solid_ok(m, components=1, genus=0)
    assert m.bbox()["size"][2] > 0.040


def test_stochastic_lsystem_is_seeded():
    args = dict(axiom="F", iterations=4, angle=25, step=5, width=1,
                rules={"F": [{"succ": "F[+F]F[-F]F", "weight": 1}, {"succ": "F[+F]F", "weight": 1},
                             {"succ": "F[-F]F", "weight": 1}]})
    a, b = lsys(seed=3, **args), lsys(seed=3, **args)
    assert canonical_hash(a) == canonical_hash(b)
    strings = set()
    for s in range(6):
        g = G.compile_lsystem(dict(args, seed=s))
        strings.add("".join(m for m, _ in G.expand_lsystem(g)))
    assert len(strings) > 1


def test_tropism_bends_growth_downward():
    base = dict(axiom="FFFFFF", rules={"X": "X"}, iterations=1, step=10, width=2)
    straight = lsys(**base)
    bent = lsys(tropism={"vector": [1, 0, 0], "e": 0.3}, **base)
    assert bent.bbox()["size"][2] < straight.bbox()["size"][2]
    assert bent.bbox()["size"][0] > straight.bbox()["size"][0]


@pytest.mark.parametrize("args,code", [
    (dict(axiom="F", rules={"F": "FF"}, iterations=20, max_symbols=10000), "lsystem_symbol_budget"),
    (dict(axiom="F", rules={"F": "F[+F]F"}, iterations=6, max_segments=100), "lsystem_segment_budget"),
    (dict(axiom="A", rules={"A": "[F A"}, iterations=6, max_depth=4), "lsystem_depth_budget"),
    (dict(axiom="F]", rules={"X": "X"}, iterations=1), "lsystem_unbalanced"),
    (dict(axiom="F", rules={"F": "F"}, iterations=33), "grammar_budget_ceiling"),
    (dict(axiom="F", rules={"F": "F+F"}, iterations=6, angle=10, max_triangles=500), "grammar_triangle_budget"),
    (dict(axiom="F", rules={"F": "F"}, iterations=1, width=-1), "bad_arg"),
    (dict(axiom="!(0)F", rules={"X": "X"}, iterations=1), "lsystem_width"),
    (dict(axiom='"(0)F', rules={"X": "X"}, iterations=1), "bad_arg"),
    (dict(axiom="FL(0)", rules={"X": "X"}, iterations=1,
          leaf={"symbol": "L", "shape": "sphere", "size": [3, 3, 3]}), "bad_arg"),
    (dict(axiom="F(__import__('os'))", rules={"F": "F"}, iterations=1), "bad_arg"),
    (dict(axiom="F(round(10)**round(10000000))", rules={"F": "F"}, iterations=1), "bad_arg"),
    (dict(axiom="A(1)", rules=[{"pred": "A(x)", "succ": "F('a'*x)"}], iterations=1), "bad_arg"),
    (dict(axiom="F(", rules={"F": "F"}, iterations=1), "bad_arg"),
    (dict(axiom="FfF", rules={"X": "X"}, iterations=1, max_components=1), "grammar_components"),
    (dict(axiom="A", rules={"A": '"FA'}, iterations=25, step=100, step_decay=10.0), "grammar_overflow"),
])
def test_lsystem_budgets_and_bad_input_fail_closed(args, code):
    err = raises(code, lsys, **args)
    assert err.path.startswith("features[0]"), err.path


def test_lsystem_leaf_count_and_move_symbols():
    m = lsys(axiom="FL+FL+FL", rules={"X": "X"}, iterations=1, angle=90, step=10, width=2,
             leaf={"symbol": "L", "shape": "box", "size": [3, 1, 3]})
    assert m.meta["lsystem"]["leaves"] == 3
    solid_ok(m, components=1)
    gap = lsys(axiom="FfF", rules={"X": "X"}, iterations=1, step=10, width=2)
    assert gap.meta["lsystem"]["segments"] == 2 and gap.meta["lsystem"]["components"] == 2


def test_direct_builders_validate_units():
    raises("bad_arg", G.build_shape_grammar, {"startshape": "S", "rules": {"S": [{"shape": "box"}]}}, units="km")
    raises("bad_arg", G.build_lsystem, {"axiom": "F", "rules": {"F": "F"}, "iterations": 1}, units="km")
    m = G.build_shape_grammar({"startshape": "S", "rules": {"S": [{"shape": "box"}]}, "unit": 5}, units="mm")
    solid_ok(m, components=1, genus=0)
