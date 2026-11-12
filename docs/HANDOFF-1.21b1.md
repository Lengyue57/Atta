# Atta v1.21b1 接手文档

> 写给下一位接手者（人或 AI）。
> 日期：2026-11-12
> 代码位置：`/home/laszapen/Atta-Beta`
> 分支：`v1.3-beta`（从开源 v1.2.0 的 `88e8e5c` 建出）

---

## 一、三十秒速览

**Atta 是什么**：一个给语言模型驱动的参数化 CAD 内核。你把零件写成可重建的 spec，它用确定性 Python 算出几何、量出真实数字、过三态验收闸门，然后才允许导出。

**这一版（1.21b1）在做什么**：v1.2 有三个结构性问题，这一版针对性地补：

1. **几何自由度太窄** —— 只能画基础图元。现在加了细分曲面、平行传输扫掠、SDF 平滑融合、网格修复，feature op 从 25 个扩到 33 个。
2. **必须上网查数据才能开工** —— 虚构题材（星球、幻想剑）根本查不到。现在有 `conjure`：29 个原型表 + 比例规则 + 确定性种子，**直接发明**可信参数，每个数字都标注来源。
3. **只强化单个模型，缺乏整体形体把握** —— 参数全对但形体是错的。现在有骨架长肉（`skeleton`）、轮廓视觉外壳（`silhouette`）、视觉一致性闸门（`silhouette_iou` 约束），以及 `design` 设计法则检查器（轮廓可读性、形体层级、比例、可制造性）。

**当前状态**：278 个测试全绿，33 个 feature op 全部实测可用，28 个 conjure 原型全部能建出闭合实体。**尚未装到 ZCode**（配置文件已生成，最后一步待做）。

---

## 二、文件位置与目录结构

```
/home/laszapen/Atta-Beta/          ← 本项目，git 仓库，分支 v1.3-beta
├── atta/                          引擎源码
│   ├── form.py            (660行) 骨架长肉 / 轮廓外壳 / 视觉一致性   ★新增
│   ├── subdiv.py          (379行) Catmull-Clark / Loop 细分曲面      ★新增
│   ├── sweepkit.py        (485行) 平行传输扫掠 / 管道 / 放样对应      ★新增
│   ├── sdfkit.py          (574行) SDF 算子 / 噪声 / 扭曲弯曲          ★新增
│   ├── repair.py          (703行) 填孔 / 定向 / 重网格 / 平滑         ★新增
│   ├── conjure.py        (1047行) 虚构生成：原型表 + 种子 + 出处      ★新增
│   ├── design.py          (461行) 设计法则检查器                      ★新增
│   ├── families/
│   │   ├── form_ops.py    (181行) 注册 skeleton / silhouette          ★新增
│   │   └── kernel_ops.py  (569行) 注册 subdivide/sweep2/pipe/loft2/
│   │                              sdf_blend/repair                     ★新增
│   ├── boolean.py                 布尔运算（★修了相切 bug）
│   ├── features.py                feature DAG 求值（★修了 mesh 注入）
│   ├── verify.py                  三态验收（★加了 silhouette_iou）
│   ├── spec.py                    Spec v3 校验（★加了 silhouette_iou）
│   ├── engine.py                  引擎编排（★加了 conjure / design）
│   ├── agent/tools.py             工具契约（★10 → 12 个工具）
│   └── version.py                 VERSION = "1.2.1b1"
├── plugins/atta/.mcp.json         已生成的 MCP 配置（指向本目录）
├── docs/
│   ├── GLOSSARY.zh-CN.md  (517行) 大白话术语表（上一轮产物）
│   └── HANDOFF-1.21b1.md          本文档
└── tests/                         278 个测试，全绿
```

**相关但不在本项目内的**：

| 路径 | 是什么 | 注意 |
|---|---|---|
| `/home/laszapen/DSH/Atta-V1.2/` | 本机 v1.3 引擎，**目录名有误导** | 实为 `v1.3-colony` 分支，与 ZCode 当前插件绑定。**不要动**，会打断现有插件 |
| `~/.zcode/cli/plugins/known_marketplaces.json` | ZCode 市场配置 | `dev-atta-local` 当前指向 `/home/laszapen/DSH/Atta-V1.2/plugins` |
| `https://github.com/laszapens/Atta` | 开源仓库 | 只有 `main` 分支（v1.2.0），本项目的改动都没推 |

---

## 三、这一版做了什么（按问题分）

### 3.1 几何自由度：25 op → 33 op

| 新 op | 能力 | 实现位置 |
|---|---|---|
| `subdivide` | Catmull-Clark / Loop / 简单中点细分，支持 `crease_angle_deg` 保持硬边 | `subdiv.py` |
| `sweep2` | 沿 3D 路径扫掠，**双反射法旋转最小化框架**（不扭转、不翻转），支持 twist/roll/逐站缩放 | `sweepkit.py` |
| `pipe` | 圆管，`inner_radius` 做真空心管（genus 1） | `sweepkit.py` |
| `loft2` | 放样，带**对应点匹配**（循环移位 + 反向尝试），避免扭曲 | `sweepkit.py` |
| `sdf_blend` | SDF 图元组合：平滑并/噪声位移/扭曲/空腔挖除，一次提取水密面 | `kernel_ops.py` + `sdfkit.py` |
| `skeleton` | 骨架长肉：关节 + 骨，按距离场生长，`blend_mm` 圆滑关节 | `form.py` |
| `silhouette` | 三视图轮廓求交成视觉外壳 | `form.py` |
| `repair` | 焊接 / 定向 / 填孔 / 去碎片 / Taubin 平滑 | `repair.py` |

**实测数据**（全部闭合、单连通）：

```
subdivide        tris=432     closed=True  vol=36549
sweep2           tris=24      closed=True  vol=1191
pipe             tris=96      closed=True  vol=3696
pipe hollow      tris=192     closed=True  genus=1  vol=2956
loft2            tris=16      closed=True  vol=37333
sdf_blend        tris=28848   closed=True  vol=14483
sdf_blend cavity tris=161948  closed=True  vol=25421
sdf_blend noise  tris=41016   closed=True  vol=29063
skeleton         tris=6552    closed=True  vol=9885
silhouette       tris=127016  closed=True  vol=19186
repair           tris=528     closed=True  vol=4084
```

### 3.2 修掉的根因 bug：布尔相切

**这是本版最重要的修复**，也是之前"骨架长肉"失败的真正原因。

`boolean.py` 原本在 manifold3d 输出后又做了一次 `weld(1e-9)`。当两个实体沿边或面**相切**时，manifold3d 会输出两套重合但**拓扑独立**的顶点（每侧一套）。`weld` 把它们焊在一起，把一个合法的 2-流形塌成"一条边挂四个面"的非流形边，于是 `closed` 判定失败。

几何本来是对的，是后处理把它弄坏了。

修复（已验证）：

- 布尔输出**不再 weld**，只在失败时用 `weld(1e-12)` 兜底重试
- 直接调 manifold3d，不走 trimesh（省 0.5 秒冷启动）
- 接受 `n_components > 1`（相切并集合法地产出两个连通分量）
- 记录 `meta["boolean"] = {backend, components}`

实测：

```
相切并集（边贴边）  closed=True comps=1 vol=2000.0
面贴面并集          closed=True comps=1 vol=2000.0
正常重叠并集        closed=True comps=1 vol=1400.0
挖孔                closed=True genus=1 vol=717.7
```

### 3.3 虚构生成：`atta_conjure`

**解决的痛点是"必须上网查数据"**。给一个语义标签（中英文都行），按内置原型表发明参数。

```python
atta_conjure(label="a battered rifle magazine", seed=7, style="brutal")
# → archetype: magazine, size: [31.2, 84.9, 181.9] mm
# → build: {op: "sdf_blend", primitives: [...], cavity: {...}}
# → provenance: {size_mm: "invented", ratios.wall: "archetype", ...}
```

**29 个原型**（全部实测能建出闭合实体）：

| 类别 | 原型 |
|---|---|
| 手持 | mug, bottle, knife, sword, axe, shield, magazine, grip, helmet, tool_box |
| 硬件 | rifle, drone, car, spaceship |
| 家具/建筑 | chair, table, column, building |
| 角色 | humanoid, quadruped, bird, insect, fish |
| 自然 | tree, mushroom |
| 天体 | planet, asteroid, ringed_planet |
| 兜底 | generic |

**8 种风格**：neutral / brutal / elegant / organic / industrial / scifi / fantasy / ancient。每种是四维向量（bulk / taper / detail / sharp），只调制比例，不改拓扑。

**三个关键设计**：

1. **出处标注**：每个数值标 `measured`（你给的）/ `archetype`（原型比例）/ `invented`（种子生成）。这是诚实性的核心——不假装发明的是量出来的。
2. **确定性**：`sha256(label|style|seed)` 做种子，同输入必得同结果。
3. **中英双语 + 最长匹配**：`"a battered rifle magazine"` 正确落到 `magazine` 而不是 `rifle`（最长同义词优先）。未知标签优雅降级到 `generic`，不报错——虚构题材本来就没有真实参考。

**人体尺度锚点**已内置（`HUMAN_SCALE_MM`）：身高 1700、手长 190、握径 32、门高 2000、座高 450、桌高 750。

### 3.4 整体形体：三套机制 + 一个检查器

**（a）骨架长肉 `skeleton`**

比例和姿态在骨架里，体积在半径里。移动关节，身体跟着走；改半径，只改肉的多少。

```json
{"id":"body","op":"skeleton","args":{
  "joints":[{"id":"a","at_mm":[0,0,0],"radius_mm":8},
            {"id":"b","at_mm":[0,0,50],"radius_mm":6}],
  "bones":[{"a":"a","b":"b"}],
  "blend_mm":3}}
```

实现是**距离场**而非布尔并集：每个骨贡献 `到线段距离 - 半径(t)`，所有项用平滑最小值融合。这样关节处自然长在一起，不会因为相切而失败。

**（b）轮廓视觉外壳 `silhouette`**

三视图（front=x-z / top=x-y / right=y-z）的轮廓求交。也是场式：`max(多边形距离, 深度板距离)`，求交精确且水密。

**（c）视觉一致性闸门 `silhouette_iou`**

这是视觉真正进入验收的唯一通道。在 spec 里加约束：

```json
"constraints": {
  "silhouette_iou": [{"view":"front", "outline_mm":[[x,z],...], "min_iou":0.90}]
}
```

`verify` 会比对实际轮廓与声明轮廓的 IoU，不达标就 warn（`strict` 档升级为 error）。**故意是 warn 级**：轮廓是投影，投影证明不了实体。

**（d）设计法则检查器 `atta_design`**

`verify` 判"是不是合法实体"，`design` 判"形体读不读得懂、能不能造"。八条规则：

| 规则 | 判什么 | 阈值 |
|---|---|---|
| `readability` | 缩小到 32px 后轮廓还认得出吗（数值化眯眼测试） | IoU ≥ 0.90 |
| `negative_space` | 包围盒空腔比例是否可读 | 0.15–0.92 |
| `form_hierarchy` | 特征尺寸是否跨三个尺度（主/次/三级） | ≥2 个模态且面积比 ≥8 |
| `proportion` | 长宽比是否尴尬 | 长:中 ≤ 12 |
| `symmetry` | 哪些轴镜像对称 | p95 误差 ≤ 1% 对角线 |
| `printability` | 最小壁厚 + 悬垂比例 | 壁 ≥0.8mm，悬垂 ≤25% |
| `moldability` | 拔模角 | 垂直面 ≤5% |
| `machinability` | 内角是否比刀具半径还尖 | 锐边 ≤15% |

每条返回 `{id, ok, value, advice}`。`ok=None` 表示不适用（不是失败）。

**实测案例**：conjure 一个 mug，`verify` 给 pass（几何合法），`design` 报出 4 个真实问题——轮廓缩小后崩、特征只有一个尺度、垂直面 52%、锐边 100%。**这正是我们要的**：几何合法但形体有问题，现在能被量化指出。

---

## 四、当前已知问题（接手者必读）

按严重程度排序。这些是**真实未解决**的，不是猜测。

### P0：内存与性能

**症状**：`sdf_blend` 在长细物体上会吃光内存（本机 30G 曾被强制终止）。

**根因**：spacing 按 `min(size)/resolution` 算。剑是 1070×197×32 mm，按 32mm 那根轴算 spacing，网格点数爆炸；marching tets 每格还要分配多个索引数组。

**已做的缓解**：

- `_fit_spacing()` 硬性采样预算 1.2M 点（约 100MB 峰值）
- 空腔壁厚采样从 `wall/2.5` 放宽到 `wall/3.0`（少采样，少三角面）

**仍未解决**：

- 弹夹仍有 110 万三角面（预算内但偏多），mug 49 万
- **没有可用的降面兜底**：`quadric_decimate` 是纯 Python 循环，百万面级会超时（实测 240 秒未完成）。这是明确的待优化点
- 建议方向：把 `repair.quadric_decimate` 改成 numpy 向量化，或引入 `pymeshlab`（GPL-3，需先确认许可）

### P1：conjure 的构建方案质量参差

28 个原型都能建出闭合实体，但**形体质量差别很大**：

- **好**：planet / asteroid（球 + 噪声）、sword / axe / knife（盒体组合合理）、chair / table（结构清晰）
- **中等**：rifle / car / drone / spaceship（图元堆叠，能看出是什么但粗糙）
- **差**：mug / bottle / helmet / grip（空腔方案勉强，壁厚与 blend 的关系脆弱）

**根因**：`_build_hint()` 里的图元布局是手写的经验值，没有系统化的比例推导。

**建议方向**：研究子代理已交回一份**基于真实人体测量数据（ANSUR II）** 的原型表设计，包含"一个自由参数 + 其余按比例/关系式派生"的方法论。这份报告的核心结论值得吸收：

- 绝对尺度必须单独提供（所有 text-to-CAD 系统都丢失尺度）
- 人体测量是唯一免费高质量先验（p95/p5 中位数 1.196，即人体尺寸自然离散约 ±10%）
- 合理性是**关系式**而非绝对值：弹夹高 = 弹数 × 节距，楼梯 2R+T≈630
- 风格应建模为**对数空间的低维乘性因子**（不是逐参数独立抖动）
- 已知陷阱：原型标称值必须离硬约束至少 1.5σ，否则风格调制全被夹到边界（我实测过：所有风格壁厚都变成 1.50mm）

### P2：`design` 检查器阈值未经校准

八条规则的阈值是我按经验设的，**没有用真实模型验证过**。例如：

- `form_hierarchy` 用三角形面积分布做特征尺寸代理，对细分曲面会误判（细分后面积趋于均匀，但形体是好的）
- `moldability` 只看法线 Z 分量，不识别侧向抽芯
- `readability` 的 32px 阈值对细长物体偏严

**建议**：拿 20–30 个已知良好的模型（现有 23 个 recipe）跑一遍，看通过率是否合理，再调阈值。

### P3：`skeleton` 的骨架质量依赖手工关节坐标

`conjure` 的 `_creature_hint()` 里，人形/四足/鸟/鱼/虫的关节坐标是手写的经验值。人形能看出是人，但姿态僵硬（没有中间关节，四肢是直线）。

**建议**：加中间关节（肘、膝、踝），或接入研究报告中提到的 space colonization 算法做分支结构（树已经在用简化版）。

### P4：装配与拼合能力未强化

原计划有"装配与拼合能力强化"这一项，**本版没有做**。v1.2 的 `assembly` 层（parts + ports + 13 种配合 + SE(3) 求解）保持原样，未改动也未测试新 op 与装配的组合。

**这是明确的功能缺口**：用户提到"忽略了拼合与大体的把握能力"，而拼合部分这一版只做了单个模型的形体，没有做多零件组合的形体。

### P5：未装到 ZCode

配置文件已生成（`plugins/atta/.mcp.json`），但**没有真正注册到 ZCode**。当前 ZCode 仍指向 `~/DSH/Atta-V1.2`（v1.3）。

安装步骤见第六节。**注意**：装之前要想清楚是否要替换现有 v1.3 插件——两者都叫 `atta`，同时启用会出现重复工具列表。

---

## 五、代码约定（改之前必须知道）

### 单位：这是最容易出错的地方

| 层 | 单位 | 字段命名 |
|---|---|---|
| Spec / 用户输入 | **mm** | 带 `_mm` 后缀（`at_mm`、`radius_mm`） |
| 内核（form/subdiv/sweepkit/sdfkit/repair） | **米** | **不带后缀**（`at`、`radius`、`blend`） |
| Blender 场景 | 米 | — |

**转换只在 `families/*_ops.py` 的 `prepare()` 里做一次**。内核绝不重复转换。

我踩过这个坑：`form.py` 早期版本字段名带 `_mm` 但值是米，导致 `blend_mm=4` 被当成 4 米，体积算出 1.7e13 mm³。现在内核层字段名一律不带后缀。

### 新 op 的注册方式

在 `families/` 下新建模块，用 `features.register_op()`：

```python
def _my_op(*, mesh, param):
    ...
    return out_mesh

def _my_prepare(args, scale, path):
    # 在这里做 mm → m 转换和校验
    return {"param": float(args["param"]) * scale}

register_op("my_op", {"required": (), "arity": 1, "optional": ("param",), "doc": "..."}, _my_op)
_my_op.prepare = _my_prepare
```

**关键**：`arity >= 1` 的 op，`mesh` 由 `_prepare_extra` 自动注入（`features.py:546`）。自定义 `prepare` 不需要也不能自己处理 `mesh`——这个 bug 我修过，之前每个自定义 prepare 都得自己加，漏了就报 `missing keyword-only argument: 'mesh'`。

### 场式构造优先于布尔

新代码里凡是"多个形体融合"的场景，**优先用距离场 + 平滑最小值**，而不是布尔并集：

```python
phi = d1
phi = smooth_union(phi, d2, k)   # 而不是 boolean.union(m1, m2)
```

原因：布尔在相切处脆弱（虽然已修，但仍有边界情况），场式天然水密。`skeleton` 和 `silhouette` 都是这么实现的。

### 采样预算

任何要采样网格的地方，**必须**有硬上限：

- `form.py`: `_MAX_SAMPLES = 2_500_000`，`_field_grid()` 自动粗化
- `kernel_ops.py`: `_fit_spacing(max_samples=1_200_000)`

没有上限的采样 = 内存耗尽 = 进程被内核杀掉。本机 30G 内存已被打爆过一次。

---

## 六、如何安装到 ZCode

### 方式 A：作为本地插件市场（推荐，有技能文件）

```bash
cd /home/laszapen/Atta-Beta
python3 -m atta.agent.install zcode-plugin    # 已执行过，生成 plugins/atta/.mcp.json
```

然后在 ZCode 里：
1. Plugin Marketplace → Add → Add Plugin Marketplace
2. 粘贴 `/home/laszapen/Atta-Beta/plugins`
3. Personal → dev-atta-local → Atta → Install

**注意**：`known_marketplaces.json` 里 `dev-atta-local` 已指向 `~/DSH/Atta-V1.2/plugins`。要装本版，要么改这个路径，要么用不同的 marketplace id。

### 方式 B：直接 MCP 配置（更简单，无技能）

把 `plugins/atta/.mcp.json` 的内容合并进 `~/.zcode/cli/config.json` 的 `mcp.servers`：

```json
{
  "mcp": {
    "servers": {
      "atta": {
        "type": "stdio",
        "command": "/usr/bin/python3",
        "args": ["-m", "atta.agent.mcp"],
        "cwd": "/home/laszapen/Atta-Beta",
        "env": {
          "ATTA_ROOT": "/home/laszapen/Atta-Beta",
          "PYTHONPATH": "/home/laszapen/Atta-Beta",
          "PYTHONUNBUFFERED": "1"
        },
        "timeoutMs": 3600000
      }
    }
  }
}
```

**必须重启 ZCode** 才会加载 `atta_*` 工具。

### 验证安装

```bash
cd /home/laszapen/Atta-Beta
python3 -m atta doctor --deep
# 期望：version 1.2.1b1，12 个工具，33 个 op
```

---

## 七、需要达到的目标

### 用户原话的核心诉求

> "我需要的是极端可怕的一个无头脚本驱动的建模怪物级工具"

拆解成可验收的目标：

| 目标 | 现状 | 差距 |
|---|---|---|
| **能建任何形体** | 33 op，能建有机/硬表面/建筑/角色 | 复杂硬表面（机械细节）仍弱；没有布尔阵列、没有倒角 op |
| **不需要查数据** | conjure 29 原型，中英双语 | 原型表覆盖窄（无家具细分、无载具细分）；构建方案质量参差（P1） |
| **整体形体把握** | 骨架 + 轮廓 + 视觉闸门 + 设计检查 | 视觉闸门只比对轮廓，不比对内部结构；装配层未强化（P4） |
| **美学符合设计法则** | design 8 条规则 | 阈值未校准（P2） |
| **无头脚本驱动** | CLI + MCP + HTTP 全可用 | 已达成 |
| **性能可用** | 简单件亚秒；复杂 SDF 件数十秒 | 内存与面数是硬约束（P0） |

### 建议的下一步优先级

1. **修 P0 内存/性能** —— 这是当前最硬的墙。方向：向量化降面，或引入 pymeshlab（先确认 GPL 兼容）
2. **吸收研究报告改进 conjure** —— 用 ANSUR II 数据 + 关系式规则 + 对数空间风格，替换手写经验值。研究报告的核心方法论已经验证过，见第四节 P1
3. **校准 design 阈值** —— 拿 23 个现有 recipe 跑一遍，看通过率
4. **强化装配** —— 用户明确提到"拼合"是短板，这一版没做
5. **装到 ZCode 并实测** —— 用真实任务验证，而不是只跑单元测试

### 三份研究子代理报告的可用结论

这一版启动时派了三路研究，其中两路已回。**核心可用结论**：

**几何内核报告**（已部分吸收）：

- ✅ 布尔相切根因（已修）
- ⬜ `manifold3d.Manifold.level_set` 可替代 marching tets（更稳，带顶点吸附）
- ⬜ `smooth_out()` + `refine_to_length()` 可做整体 G1 圆角（实测立方体 12→432 面，体积 1000→2359）
- ⚠️ **陷阱**：`smooth_out` 后接 `refine_to_tolerance` 会 **segfault**（全锐边时）。用 `refine_to_length` 或 `refine(n)` 替代
- ⬜ build123d 已装但 11 个 op 未接线：`fillet`、`chamfer`、`hollow`、`offset_3d`、`thicken`、`loft`、`sweep`、`draft`、`split`、`helix`、`pipe`。**这是最大的未开发能力块**
- ⬜ `pymeshfix`（numpy-only，pip 可装）比 `trimesh.fill_holes` 可靠得多——实测能修复 trimesh 静默失败的破洞球

**虚构生成报告**（尚未吸收，方法论完整）：

- ANSUR II 人体测量真实数据（已下载计算，非引用）
- 三种规则类型：锚点（绝对数据）/ 比例（无量纲）/ 关系式（耦合参数）
- 风格 = 对数空间低维乘性因子（不是逐参数抖动）
- 失败模式分类：尺度丢失、退化特征、操作替换、界面忽视、仅粗粒度保真
- 30 个原型的起始表，含"一个自由参数 + 其余派生"的纪律

**美术法则报告**：仍在运行，未回。

---

## 八、快速上手命令

```bash
cd /home/laszapen/Atta-Beta
export PYTHONPATH=/home/laszapen/Atta-Beta
export ATTA_HOME=/tmp/atta-work        # 别污染仓库

# 自检
python3 -m atta doctor --deep

# 看能力
python3 -m atta catalog --verbose | head -50

# 虚构生成（不查数据）
python3 -c "
from atta import conjure
p = conjure.conjure('a battered rifle magazine', seed=7, style='brutal')
print(p['archetype'], p['size_mm'])
print('build:', p['build']['op'])
"

# 建模 + 验收 + 设计检查（走工具层，和 agent 一致）
python3 -c "
from atta.agent import tools
r = tools.call_tool('atta_conjure', {'label':'fantasy sword','seed':1,'build':True})
mid = r['data']['model_id']
print('model:', mid)
print(tools.call_tool('atta_verify', {'model_id':mid})['data']['verdict'])
print(tools.call_tool('atta_design', {'model_id':mid})['data']['summary'])
"

# 跑测试
python3 -m pytest tests/ -q
```

---

## 九、给接手 AI 的特别提醒

1. **先跑 `pytest`**，确认 278 个测试全绿再动手。任何变红都说明你改坏了东西。
2. **不要在 `~/DSH/Atta-V1.2` 里改任何东西**。那是本机 v1.3，ZCode 插件正绑着它。
3. **单位是最大的坑**。改内核代码前，确认你写的是米；改 spec 层前，确认你写的是毫米。
4. **任何采样都要有上限**。本机已被打爆过一次内存。
5. **布尔输出不要 weld**。这是本版修掉的根因，别改回去。
6. **conjure 的诚实性不能妥协**。`provenance` 标注是设计核心——发明就是发明，永远不要标成 measured。
7. **验证要实测，不要只读代码**。这一版所有"能用"的结论都是跑出来的，包括布尔相切、骨架体积、28 个原型的闭合性。

---

## 十、变更清单（相对开源 v1.2.0）

**新增文件**（9 个，约 5060 行）：

```
atta/form.py
atta/subdiv.py
atta/sweepkit.py
atta/sdfkit.py
atta/repair.py
atta/conjure.py
atta/design.py
atta/families/form_ops.py
atta/families/kernel_ops.py
```

**修改文件**（10 个）：

| 文件 | 改动 |
|---|---|
| `atta/boolean.py` | 修相切根因；直接调 manifold3d；接受多分量 |
| `atta/features.py` | `_prepare_extra` 统一注入 `mesh`（修自定义 prepare 漏注入） |
| `atta/verify.py` | 加 `silhouette_iou` 检查 + `_silhouette_checks()` |
| `atta/spec.py` | 注册 `silhouette_iou` 约束 + 校验 |
| `atta/engine.py` | 加 `conjure()` / `design()` / `archetypes()` |
| `atta/agent/tools.py` | 10 → 12 工具（`atta_conjure`、`atta_design`） |
| `atta/version.py` | `1.2.0` → `1.2.1b1` |
| `pyproject.toml` | 同上 |
| `tests/test_agent_tools.py` | 工具数 10 → 12 |
| `tests/test_agent_mcp.py` | 同上 |

**未提交**：所有改动都在工作区，`git status` 显示 10 个修改 + 10 个未跟踪文件。分支 `v1.3-beta` 尚未 commit。

---

## 十一、参考

- 项目术语表：[`docs/GLOSSARY.zh-CN.md`](GLOSSARY.zh-CN.md)（517 行，大白话解释所有术语）
- 架构说明：`docs/ARCHITECTURE.md`（v1.2 原文，未改）
- 为 LLM 设计的原则：`docs/LLM_NATIVE.md`（v1.2 原文，未改）
- 上游开源：https://github.com/laszapens/Atta
