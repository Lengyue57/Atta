# 混合几何内核端到端示例

`spec.json` 将精确圆角底座、NURBS 厚化面板、精确倒角盖和 gyroid 场树嵌件组成四零件装配。定向端口负责初始定位，三个 lock mate 将自由度约束到 0。公开参数为 mm/deg，最终网格和 GLB 坐标为 m。

这是内核与交付契约的回归样件。面板目前是平面 NURBS patch，端口包含刻意留出的展示间距；它不代表复杂自由曲面造型或制造认证。曲面精度、微尺度和非均匀节点的独立覆盖见 `tests/test_nurbs.py`。

## 自动验收

```bash
cd /path/to/Atta
python3 -m pytest -o addopts='' tests/test_kernel_showcase.py -q
```

测试使用独立临时 `ATTA_HOME`，经过真实 `engine.spec → model → describe → verify → deliver`，无需 Blender。依赖项目现有 optional `brep` extra；本地未安装时明确 skip，新增 `.github/workflows/kernel.yml` 安装现有 `.[all]` 并先导入 build123d/OCP，防止 CI 意外跳过 CAD。

检查包括：

- 四个闭合连通分量、包围尺寸 84 × 40 × 25 mm、DOF=0、定向 attach 残差。
- 精确零件来源哈希、NURBS 的 sampled target 与空的全局误差界、场树非精确距离声明。
- STEP 回读两个有效实体的解析体积与最终世界包围盒；GLB 回读全部四个零件和米制尺寸。
- 产物 SHA-256、manifest 与验证回执的一致性。
- 将整机 X 尺寸约束改为 200–210 mm 后，交付必须 blocked，不能生成文件。

## 可复制交付

在仓库内执行以下 Python。脚本为每次运行新建一个 `_atta/out` 子目录，保留已有状态和产物，不删除旧目录。

```python
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from atta import engine

root = Path.cwd()
os.environ['ATTA_HOME'] = str(root)
out = root / '_atta' / 'out'
out.mkdir(parents=True, exist_ok=True)
out = Path(mkdtemp(prefix='kernel-acceptance-', dir=out))
spec = json.loads((root / 'examples/kernel_showcase/spec.json').read_text())
assert engine.spec(spec, op='validate')['ok']
model = engine.model(spec=spec, quality='review')
mid = model['model_id']
description = engine.describe(mid)
receipt = engine.verify(mid, constraints={
    'fully_constrained': True,
    'bbox': {'x': [83.99, 84.01], 'y': [39.99, 40.01], 'z': [24.99, 25.01]},
})
assert receipt['verdict'] == 'pass', receipt['issues']
delivery = engine.deliver(mid, formats=['glb', 'step'], outdir=out)
assert delivery['status'] == 'partial'
(out / 'acceptance.json').write_text(json.dumps({
    'model_id': mid, 'description': description,
    'verify': receipt, 'delivery': delivery,
}, ensure_ascii=False, indent=2))
print(mid, receipt['mesh_hash'], receipt['spec_hash'])
print(delivery['manifest_path'])
```

`partial` 是预期的 **格式覆盖不完整**，与几何 `verify=pass` 是两个不同结果：GLB 含全部零件；STEP 只含 `base` 和 `cap`，`panel`/`lattice` 在 `omitted` 和 `errors` 中明确列出。NURBS/场网格没有被伪装成解析 BREP。真正的 BREP 构建、依赖或文件错误保留原始错误码，不改称 mesh-only omission。

当前 STEP 是世界坐标 compound，不包含完整 CAD 装配层级、参数关联或求解器 mate。三角化控制和 NURBS 采样误差都不是全曲面 Hausdorff 证明；拓扑验证也不是自交或制造工艺证明。
