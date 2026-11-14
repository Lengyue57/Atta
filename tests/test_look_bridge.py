# -*- coding: utf-8 -*-
"""The optional Atta→basid bridge: lazy, structured and non-duplicating."""
from __future__ import annotations

import sys
import types

import pytest

from atta import look
from atta.errors import AttaError


def test_available_is_false_without_basid(monkeypatch):
    monkeypatch.setitem(sys.modules, "basid", None)
    assert look.available() is False
    with pytest.raises(AttaError) as error:
        look.doctor()
    assert error.value.code == "look_unavailable"
    assert error.value.fix


def test_calls_delegate_to_the_public_basid_api(monkeypatch):
    seen = {}

    def shoot(parts=None, shots=None, *, mesh=None, texture=None, **kwargs):
        seen.update(mesh=mesh, texture=texture, kwargs=kwargs)
        return {"ok": True, "gate": {"pass": True}}

    def plan(subject=None, shots=None, *, mesh=None, **kwargs):
        seen.update(plan_mesh=mesh, plan_kwargs=kwargs)
        return {"ok": True, "n_shots": 3}

    fake = types.SimpleNamespace(__version__="0.2.0", shoot=shoot, plan=plan)
    monkeypatch.setitem(sys.modules, "basid", fake)
    assert look.available() is True
    assert look.plan(mesh="m.glb", kit="product_hero")["n_shots"] == 3
    seen.clear()
    result = look.shoot("m.glb", texture="tex_1", kit="product_hero", res=256)
    assert result["gate"]["pass"] is True
    assert seen == {"mesh": "m.glb", "texture": "tex_1",
                    "kwargs": {"kit": "product_hero", "res": 256}}


def test_foreign_errors_become_atta_errors(monkeypatch):
    def failing(**_kwargs):
        raise RuntimeError("no GPU")

    fake = types.SimpleNamespace(__version__="0.2.0", doctor=failing)
    monkeypatch.setitem(sys.modules, "basid", fake)
    with pytest.raises(AttaError) as error:
        look.doctor()
    assert error.value.code == "look_error"
    assert "no GPU" in str(error.value)


def test_incompatible_basid_is_named(monkeypatch):
    fake = types.SimpleNamespace(__version__="0.0.1")
    monkeypatch.setitem(sys.modules, "basid", fake)
    with pytest.raises(AttaError) as error:
        look.plan(mesh="m.glb")
    assert error.value.code == "look_incompatible"
    assert "plan()" in error.value.message
