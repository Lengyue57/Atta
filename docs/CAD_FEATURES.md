# 精确 graph 圆角 / 倒角（OCCT）

## 契约：只在需要 CAD 时进入精确分支

`fillet` / `chamfer` 不是三角网格平滑或渲染 bevel。`features.evaluate` 检查**输出可达依赖闭包**：其中包含任一操作时，由 `brep.mesh_graph(features, params, units, output)` 一次构造整个闭包的 OCCT BREP，再从该解析实体细分出米制 Mesh。STEP 交付调用相同的 `build_graph`、几何选边和精确操作；不会先改 mesh 再另造一个近似 STEP。

普通 mesh 图、只有不可达 fillet/chamfer 的图不进入 BREP，不因注册新 op 而导入 build123d。family 注册没有可选依赖的冷启动成本。函数不能从任意 Mesh 猜回解析祖先：直接调用 family fallback 会报 `brep_provenance_required`。

这是一套**同源构造算法**，不是跨进程保持同一个 OCCT 对象，也不是跨 OCCT 版本永久不变的拓扑命名系统。mesh/STEP receipt 含 `source_sha256`（规范化为 mm、解析参数后的可达 graph），以及使用的几何选边描述。重新构造会使用同一个选择器；选边数量必须固定时使用 `expect_count`。哈希标识输入，不证明几何完全相等。

### 支持的整个闭包

`box`, `cylinder`, `sphere`, `cone`, `torus`, `revolve`, `extrude`, `union`, `difference`, `intersection`, `transform`, `mirror`, `pattern_linear`, `pattern_polar`, `fillet`, `chamfer`。

- 不支持的祖先**和后继**（如 `subdivide`, `sdf`, `loft`, `thicken`, NURBS mesh 等）均报 `brep_unsupported`。不会将三角面转换成假解析实体，也不混用 mesh 布尔。
- `caps=false` 的开放输入不允许；每步检查正体积、OCCT 有效性及闭合 shell。空交集、无效实体、开放面失败关闭。
- `transform` 支持平移、XYZ 欧拉旋转、pivot、**正的均匀缩放**；负缩放/非均匀缩放不支持，应显式使用 `mirror` 或普通 mesh 路径。
- `revolve` 的闭合截面绕转；未闭合截面在 `caps=true` 时补至轴形成端盘。`extrude.direction` 使用与 mesh 相同的局部垂直截面框架，含 `holes_xy`；不是把世界 XY 截面斜拉伸。
- 公共 Spec v3（`atta_spec` / `engine.model`）的 graph `units` 固定为 `mm`，其他单位会被 Spec 校验拒绝；提交 Spec 前须换算为 mm。低层 Python API `features.evaluate`、`brep.build_graph` / `mesh_graph` 支持 `mm/cm/m/in`：尺寸、坐标、选择容差在该边界转换一次为 mm。OCCT 以 mm 计算、STEP 明确声明 mm、Mesh 顶点最终乘 `0.001` 变为 m。角度是度，方向不缩放。

## 最小例子

```json
{
  "schema": "atta.spec/v3",
  "name": "rounded_plate",
  "model": {
    "kind": "graph", "units": "mm", "output": "round",
    "params": {"r": 2},
    "features": [
      {"id": "body", "op": "box", "args": {"size": [30, 20, 10]}},
      {"id": "round", "op": "fillet", "inputs": ["body"],
       "args": {"radius": {"$param": "r"},
                "edges": {"parallel": "z", "kind": "line", "expect_count": 4},
                "linear_deflection": 0.05,
                "angular_deflection_deg": 12,
                "max_triangles": 100000}}
    ]
  }
}
```

将 `op` 改为 `chamfer`，把 `radius` 改为 `length`，即等距两侧倒角；本版不支持不等距/角度倒角，不会默默忽略这些参数。两个操作都要求显式 `edges`，不传则报错。

## 几何选边（禁止数字 edge index）

`edges` 必须是 `"all"`，或者一个非空选择对象。对象中的条件为 **AND**，匹配的是集合，而非“选枚举顺序第 N 条”。

| 条件 | 语义 |
|---|---|
| `parallel: "x" / "y" / "z"` | 当前世界坐标中与该轴平行的**直线边**（不区分方向正负）；不会用圆弧某一点的切线代替整条边方向 |
| `kind: "line" / "circle"` | OCCT 解析曲线种类 |
| `at: {"axis":"z", "value":5}` | **整条边**的轴向包围区间均在坐标平面 z=5 的容差内，不只是边中心 |
| `at: {"axis":"z", "value":"max"}` | 上述平面取输入实体整体包围盒该轴的最大值；`min` 同理 |
| `at.tolerance` | 正有限长度，使用 graph 单位；省略时为物理 `1e-6 mm`；这是选择容差，不是焊接容差 |
| `expect_count: 4` | 要求恰好匹配 4 条，否则报 `brep_selection_count` |

例如选择顶部周边：`{"at":{"axis":"z","value":"max"},"expect_count":4}`；选择圆柱顶缘：`{"kind":"circle","at":{"axis":"z","value":"max"},"expect_count":1}`。

只写 `expect_count` 不构成选择器。`{}`、数字、数字列表、未知条件都拒绝；显式 `null` 条件（包括 `parallel`、`kind`、`at`、`expect_count`）报 `bad_selector`，不能用空值表示忽略条件或选择全部边。零匹配报 `brep_selection_empty`。所选边按几何种类、包围盒、长度、中心和无方向采样签名确定排序；不把原生拓扑索引或进程地址放进公开 receipt。receipt 包含操作尺寸、匹配数量、逐边几何描述。

**all 不是成功保证**：它请求全部原生边，包括不适合倒圆/倒角的缝线、切线交界或退化边。即便选边正确，过大半径、复杂角点、相邻特征干涉也可能使 OCCT 失败。返回 `brep_operation_failed` 和建议减小尺寸/收窄选边，绝不自动减小半径、删掉失败边或退回渲染 bevel。

## 水密细分与预算的准确边界

不能直接把 build123d `Shape.tessellate` 的逐面顶点拼起来，因为共享边会重复；更不能全局容差 weld，它会把相切的独立实体合并成非流形。

本实现用 OCCT `BRep_Tool.Triangulation` 和 `PolygonOnTriangulation`：

1. 整个 solid 由 `BRepMesh_IncrementalMesh` 统一生成边采样与各面三角形。
2. 同一个 TopoDS edge 的 polygon 序号/参数对齐、TopoDS vertex 的端点身份合并，缝线的两侧同样按原生边处理。
3. 退化极点按其原生端点收缩；仅删除由此产生的重复索引三角形。
4. 每个 solid 独立拓扑命名空间，坐标重合但不同的 solid 仍然各自闭合。封闭空腔则保留内外两套边界 shell。
5. 最后检查水密、方向一致和非流形边；失败报 `brep_tessellation`，没有空间焊接兜底。mesher 状态仅接受无错误、`ReMesh`（边细分加密）、`Reused`（有效缓存复用），其余和未知位失败关闭，原始状态保存在 `occt_status_flags`。

| 参数 | 默认 / 限制 |
|---|---|
| `linear_deflection` | 缺省物理 0.05 mm；显式值使用 graph 单位；至少 `max(1e-7 mm, bbox_diagonal_mm / 100000)` |
| `angular_deflection_deg` | 默认 12°，只接受 `[1,45]` |
| `max_triangles` | 默认 100,000，整数 `[4,500000]`；超限不返回部分网格 |

多个 CAD 节点时使用它们最严格（各项最小）的设置细分最终输出。过小误差请求在进入 mesher 前拒绝；OCCT 完成后逐面读取**计数**，在构造 Python 顶点/面数组前校验三角数和原始节点预算。输出不超预算，不自动放宽误差。

**重要：这不是 OCCT 原生内存/时间的硬上限。** 原生 mesher 没有按三角数中途取消的当前接口，它可能在返回计数之前分配更多内存。`native_allocation_hard_limit:false` 明确保留此限制；不能据此声称消除了 OOM 风险或证明最大 RSS。不要给复杂图无限缩小 deflection，宿主仍应提供外部进程资源约束。

`linear_deflection` / angle 是 OCCT 细分控制，不是严格全曲面 Hausdorff 上界，OCCT 也可能受形体本身 tolerance 限制。receipt 为 `exact:false`、`strict_surface_error_bound:false`。`analytic_volume_mm3` 与 mesh 的实测体积分开：它们有可量化的离散误差，不把 mesh 体积冒充解析体积。模型仍需通过常规 verify；STEP 仍经交付闸门导出。

## 测试与复现

```bash
python3 -m pytest tests/test_cad_features.py tests/test_implicit_brep.py -q
```

覆盖 STEP 再导入（有效实体、解析体积、bbox、面数量）、all/方向/位置/圆边选择、无效选择与数量、不可行尺寸、单位/参数/前向引用、变换与后继布尔、普通路径不调用 BREP、依赖缺失、预算拒绝、球/圆锥极点、圆柱/环缝线、相切分量独立和重复构建确定性。

30×20×10 mm 盒子：全部边 R1 的解析体积约 5949.834793 mm³，默认细分 2396 三角形，mesh 体积约 5949.494300 mm³（本机 build123d 0.13.0；数值为观察，不是所有版本的位级保证）。沿 Z 的四条竖边 C1 的解析/网格体积均为 5980 mm³。

## 可核对上游 API / 许可

本实现仅使用现有 optional `build123d` 及它带来的 OCP/OCCT，没有安装新包，没有复制上游实现。已在本机 build123d 0.13.0 验证；上游 latest 文档会移动，以下 OCCT 源链接固定在 7.9.0 标签。

- [build123d Direct API：Mixin3D.fillet/chamfer、Shape.mesh/tessellate、拓扑](https://build123d.readthedocs.io/en/latest/direct_api_reference.html)
- [build123d topology selection](https://build123d.readthedocs.io/en/latest/topology_selection.html)
- [build123d import/export（STEP / Unit.MM）](https://build123d.readthedocs.io/en/latest/import_export.html)
- [build123d Apache-2.0 许可](https://github.com/gumyr/build123d/blob/v0.13.0/LICENSE)
- [OCCT 7.9.0 meshing guide：deflection、统一边离散、PolygonOnTriangulation](https://github.com/Open-Cascade-SAS/OCCT/blob/V7_9_0/dox/user_guides/mesh/mesh.md)
- [OCCT 7.9.0 IMeshData_Status：错误与缓存复用状态](https://github.com/Open-Cascade-SAS/OCCT/blob/V7_9_0/src/IMeshData/IMeshData_Status.hxx)
- [OCCT LGPL-2.1 许可文本](https://github.com/Open-Cascade-SAS/OCCT/blob/master/LICENSE_LGPL_21.txt) 与 [OCCT linking exception](https://github.com/Open-Cascade-SAS/OCCT/blob/master/OCCT_LGPL_EXCEPTION.txt)。使用本功能即使用 OCCT 提供的设施；如另行分发它的二进制，仍须遵守其许可、告知和源码/可重链接等适用义务，不能把上游依赖说成项目自己的许可。
