# 有界组合场与数值契约

`field_tree` 在一个显式采样域内组合基本场、CSG、TPMS、噪声与坐标变形，然后统一提取闭合网格。公开 Spec 长度为 mm，角度为 deg；`atta.fieldtree` Python 内核长度为 m。它不承诺精确欧氏距离、全域曲面误差界或任意尺度都能同时表示。

## 入口与算子

Graph 参数为 `tree`、`bounds=[[xmin,ymin,zmin],[xmax,ymax,zmax]]`、正 `spacing`，可选 `max_samples/max_evaluations/max_triangles/chunk_points/budget_policy/min_feature`。完整注册 schema 可用 `python3 -m atta catalog --feature field_tree` 查询。

树支持 sphere、box、cylinder、torus、capsule、gyroid、schwarz_p、union/intersection/difference 及 smooth 变体，以及 transform、shell、offset、noise、twist。各节点仅接受自身声明的字段；不执行 Python 表达式。`transform` 为正均匀缩放、XYZ 旋转和平移；顺序决定噪声与变形使用的坐标系。

`bounds` 是采样域，不是裁切实体。需要裁切时在树中显式 intersection 一个基本体。六个补边面的场值必须为正；与边界相交报 `sdf_open`。TPMS 的 `band` 是无量纲等值区间，不能直接当成实际壁厚。

## 预算与近似性质

| 资源 | 硬限制 |
|---|---:|
| 节点 | 128 |
| 深度 | 24 |
| 采样点 | 2,500,000 |
| 估计节点计算次数 | 200,000,000 |
| 输出三角形 | 2,000,000 |
| 单次求值点数 | 262,144 |

默认 `budget_policy="error"`，预算不够立即拒绝；只有显式 `coarsen` 才会放宽 spacing。声明 `min_feature` 时，实际 spacing 必须保留至少三个采样间隔，否则 `field_under_resolved`。这是一项分辨率条件，不证明任意拓扑、小孔或薄壁已被恢复。

节点/深度和采样/工作预算在相应分配前检查。提取按活动 cell 分块，并在保留三角形批次之前检查输出预算；跨块使用整数网格边身份连接，不以空间容差焊接。预算不是 Python/NumPy/原生库的进程 RSS 硬上限。

`mesh.meta.field.distance_exact=false`，`sampling` 记录请求和实际 spacing、shape、coarsened 与估计工作量。网格不具有严格全曲面误差界。布尔后的记录保存在 `geometry_provenance.sources`，不伪装成新布尔结果的解析参数。

## 数值边界

- sphere/capsule 与 box/cylinder/torus 距离路径使用稳定 `hypot` 范数，避免先平方带来的溢出/下溢。回归测试包括 `1e-200..1e200 m` 的点距离运算，以及 `scale=1e200` 作用于 `radius=1e-200 m` 的球。**这不是这些尺度上的网格、体积、Blender 或制造能力声明。**
- 每个树节点都检查浮点运算和输出。上溢、下溢、非有限中间坐标或场值明确拒绝，避免在父 CSG 的 min/max 中被掩盖为有限值。失败返回 `AttaError` 的 `bad_arg` 或 `field_invalid`，并保留树路径。
- 非零长度/调整量在单位换算后消失，以及非零 twist 角度产生零扭转率，明确拒绝；零 offset/噪声幅度仍合法。
- 普通 transform 使用固定顺序归约。相同树和格点在不同 chunk 大小下不依赖 BLAS 的批量选择；独立六节点 CSG/noise/twist/旋转组合在 1/127/32768 点 chunk 下哈希一致。
- 噪声整数格点坐标按 octave 检查范围；超过范围要求局部坐标系，不能让整数溢出改变图案。
- 采样格点与插值顶点还受世界坐标 float64 分辨率限制。超出表示能力时要求先在局部坐标建模，再定位模块；不能从已被舍入的输入恢复细节。
- 超大 JSON 整数、非法查询点、零维 ndarray 向量、错误 axis 和采样预算均返回结构化错误。这里覆盖已测试入口，不声称整个历史几何库的所有输入都已审计。

## 验证

```bash
cd /path/to/Atta
python3 -m pytest -o addopts='' tests/test_fieldtree.py tests/test_field_budget.py tests/test_field_numeric_contract.py -q
```

完整机械/曲面/场混合交付见 [kernel_showcase](../examples/kernel_showcase/README.md)。提取器复测数据见 [kernel-field-acceptance.json](benchmarks/kernel-field-acceptance.json)，属于单次、有界子进程观察，不是 CI 性能门槛。
