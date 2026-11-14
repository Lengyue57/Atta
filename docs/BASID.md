# Look pipeline: basid

Atta owns geometry. **basid** owns what the geometry looks like: verified PBR
texture sets, planned cameras and measured framing, textured renders, and
textured GLB export. It is a separate MIT-licensed package and MCP server; Atta
does not vendor it and adds no runtime dependency on it.

```bash
# one-time, outside Atta
pip install -e /path/to/basid      # or install from its repository
```

```python
from atta import look

look.available()                                     # False when basid is absent
look.doctor()                                        # Blender / texture library / backend
look.plan(mesh="model.glb", kit="product_hero")      # numeric cameras, no render
look.shoot("model.glb", texture="tex_ab12cd34", kit="product_hero")
look.deliver("model.glb", "tex_ab12cd34", out="out/model.glb")
```

The bridge is lazy. A missing or incompatible basid raises `AttaError` with
`code="look_unavailable"` or `look_incompatible` and a `fix`; it never breaks
`import atta` or the geometry tools. basid also ships its own MCP tool, so an
agent can call it directly instead of going through Atta.

## File handoff

1. Build, verify and export geometry with Atta (`atta_deliver --formats glb`).
2. Plan, shoot and texture that GLB with basid.
3. If you want Atta's own preview/deliver to use a basid texture set, pass
   `texture_set` in the material spec. Atta resolves the frozen
   `atta.texture/v1` layout and can use its scalar or image-based material path;
   it no longer creates texture sets itself.

Geometry acceptance stays with `atta_verify`. Framing receipts and texture
verdicts are basid's evidence; a good photograph is not a geometry proof.
