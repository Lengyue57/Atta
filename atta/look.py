# -*- coding: utf-8 -*-
"""Optional bridge to **basid**, the separate look pipeline.

Atta owns geometry: it builds, measures, verifies and exports a mesh/GLB. Basid
owns what that geometry *looks like*: verified PBR texture sets, planned and
measured cameras, textured renders and textured GLB export. The two programs
never import each other; this module imports basid lazily, only when a caller
asks for a look operation, and turns a missing/incompatible basid into an
:class:`AttaError` with a ``fix``.

    from atta import look
    look.available()                       # False when basid is absent
    look.shoot("model.glb", kit="product_hero", texture="tex_ab12cd34")
    look.deliver("model.glb", "tex_ab12cd34", out="out/model.glb")

The argument names and receipts are basid's public API. No texture or camera
implementation lives in Atta after 1.2.2beta; this is deliberately a thin
transport, not a second copy of basid.
"""

from __future__ import annotations

import importlib
from typing import Any

from .errors import AttaError

__all__ = [
    "available", "doctor", "generate", "ingest", "kit", "plan", "shoot",
    "deliver", "verify_set",
]

_FIX = ("install basid (`pip install -e /path/to/basid`) or point PYTHONPATH "
        "at its checkout; Atta does not vendor it")


def _module():
    try:
        return importlib.import_module("basid")
    except Exception as exc:  # noqa: BLE001 - any import failure means unavailable
        raise AttaError(
            "look_unavailable",
            "the basid look pipeline is not importable: %s" % exc,
            path="basid",
            fix=_FIX,
        ) from exc


def _call(name: str, /, *args, **kwargs) -> Any:
    module = _module()
    func = getattr(module, name, None)
    if not callable(func):
        raise AttaError(
            "look_incompatible",
            "basid %s does not expose %s()" % (
                getattr(module, "__version__", "unknown"), name),
            path="basid.%s" % name,
            fix="install a basid release with the public API (0.2 or newer)",
        )
    try:
        return func(*args, **kwargs)
    except AttaError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize a foreign error boundary
        code = getattr(exc, "code", None) or "look_error"
        raise AttaError(
            code, str(exc), path=getattr(exc, "path", "basid.%s" % name),
            fix=getattr(exc, "fix", "inspect the basid receipt and retry"),
            details=getattr(exc, "details", None),
        ) from exc


def available() -> bool:
    """True when ``import basid`` succeeds. Never raises."""
    try:
        _module()
        return True
    except AttaError:
        return False


def doctor(**kwargs) -> dict:
    """Environment report from basid (Blender, texture library, backend)."""
    return _call("doctor", **kwargs)


def generate(prompt: str, **kwargs) -> dict:
    """Generate a texture image through basid's optional local backend."""
    return _call("generate", prompt, **kwargs)


def ingest(image: str, **kwargs) -> dict:
    """Turn an existing image into a verified, seamless PBR texture set."""
    return _call("ingest", image, **kwargs)


def kit(id: str | None = None, **kwargs) -> dict:
    """Camera kit index, or one kit expanded into shots."""
    return _call("kit", id, **kwargs)


def plan(mesh: str | None = None, shots=None, **kwargs) -> dict:
    """Resolve shot intent into numeric cameras without rendering."""
    return _call("plan", None, shots, mesh=mesh, **kwargs)


def shoot(mesh: str, texture: str | None = None, shots=None, **kwargs) -> dict:
    """Render and measure planned cameras against a mesh file."""
    return _call("shoot", None, shots, mesh=mesh, texture=texture, **kwargs)


def deliver(mesh: str, texture: str | None = None, out: str | None = None,
            **kwargs) -> dict:
    """Export a textured GLB/OBJ through basid and return its receipt."""
    return _call("deliver", mesh, texture, out=out, **kwargs)


def verify_set(value: Any, level: str = "standard") -> dict:
    """Verify a texture-set id or a set dict with basid's measurement gate."""
    return _call("verify_set", value, level=level)
