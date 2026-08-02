# 给纯文本模型装上眼睛：手写 MCP 视觉服务器全解析（小米 MiMo V2.5 实测，21 个工具，含 PCB 元件扫描与 Computer Use 实战）

> 摘要：DeepSeek、Codex 这类纯文本大模型推理能力很强，但天生没有视觉能力，无法看图、读截图、识别 UI。本文完整记录如何用小米 MiMo V2.5 多模态 API 手写一个零依赖、单文件的 MCP 视觉服务器（vision-bridge-mcp），让纯文本模型获得描述图片、目标定位（坐标输出）、OCR 文字提取、圈画标注、区域裁切放大、PCB 元件异常扫描、Computer Use 电脑操控等完整视觉能力。全文包含 21 个工具的功能解析、真实实测精度数据（定位误差从 70px 优化到 20px）、多模型对比（MiMo vs 本地模型）、MCP 协议踩坑记录与性能边界。

---

## 一、背景：为什么需要给 LLM 装眼睛

大语言模型的发展分成两条路：对话推理模型（DeepSeek、Qwen、GPT 系列）和多模态模型（GPT-4V、Qwen-VL 等）。前者在推理、编程、Agent 任务上表现优异，但**没有视觉输入能力**——你给它一张截图、一张 PCB 照片、一个 UI 设计稿，它只会回答"我无法查看图片"。

现实中的 Agent 任务大量依赖视觉信息：

- 截图分析：报错弹窗、页面状态、测试结果
- UI 自动化：定位按钮坐标、验证点击结果
- 硬件场景：PCB 元件检测、歪斜元件扫描、丝印识别
- 文档场景：OCR 提取、图表理解、设计稿还原

解决方案通常有三条路：

1. **换多模态模型**：成本高，需要迁移全部链路，且很多场景下推理能力不如纯文本模型
2. **本地部署视觉模型**：需要显存和工程维护，小模型的视觉精度（尤其定位）不达标
3. **视觉工具化（本文方案）**：纯文本模型负责推理，通过 MCP 工具调用视觉 API 完成"看"的动作，推理与视觉解耦

第三条路的核心思想来自 DeepSeek 的《Thinking with Visual Primitives》（视觉原语）：把视觉任务拆解为**描述、定位（坐标）、OCR、标注、裁切**等原子操作，模型按需调用，形成"看 → 想 → 操作 → 验证"的闭环。

## 二、整体架构：零依赖单文件 MCP 服务器

```
┌─────────────────────┐
│ 纯文本模型（推理）    │  Codex / DeepSeek / Claude
│ 负责决策与工具编排    │
└──────────┬──────────┘
           │ MCP 协议（stdio，JSON-RPC 2.0 + Content-Length 帧）
           ▼
┌─────────────────────┐
│ vision-bridge-mcp   │  单文件 Python，仅依赖 Pillow
│ 21 个视觉工具        │  手写 MCP 协议，零第三方运行时依赖
└──────────┬──────────┘
           │ OpenAI 兼容 API（多模态）
           ▼
┌─────────────────────┐
│ 小米 MiMo V2.5      │  云端视觉后端
└─────────────────────┘
```

设计要点：

- **单文件实现**：`vision_bridge_mcp.py` 一个文件搞定全部逻辑，Python 标准库 + Pillow
- **手写 MCP 协议**：不依赖任何 MCP SDK，直接实现 JSON-RPC 2.0 帧协议
- **可切换后端**：通过环境变量支持小米 MiMo、LM Studio 本地模型（qwen3.5-9b、minicpm-v-4_5）等多后端
- **结果缓存**：按图片 sha256 + 问题 + 模型缓存结果，重复调用秒回

## 三、21 个工具全览

| 分类 | 工具 | 作用 |
|---|---|---|
| 基础视觉 | `describe_image` | 文字描述图片内容（支持针对性提问、细节程度控制） |
| | `analyze_image` | 结构化分析：描述 + visual_primitives（坐标框/点 + 标签 + 置信度） |
| 定位 | `locate_object` | 定位目标对象，输出坐标；`refine=true` 两阶段精修 |
| 文字 | `ocr_image` | 逐文本块 OCR，返回 bbox（像素 + 归一化坐标） |
| 图像处理 | `annotate_image` | 在图上画框/圆点/标签，保存标注图 |
| | `crop_image` | 按坐标裁切（支持边缘外扩） |
| | `zoom_region` | 区域放大（1-8 倍） |
| 高级推理 | `compare_images` | 2-4 张图对比分析（A/B 截图、设计稿一致性、多帧分析） |
| | `compare_infer` | 多图联合推理，每图可带独立标注 |
| | `reason_graph` | 交互式图形推理协议：定位→语义→标注多轮循环 |
| | `annotate_infer` | 虚拟标注注入：框/点/连线/箭头/多边形/气泡，增强图形推理 |
| | `scan_anomalies` | 自动异常扫描：切块定位候选→高清逐点验证→输出报告 |
| 电脑控制 | `screen_capture` | 截屏（全屏或指定区域） |
| | `screen_info` | 屏幕分辨率 / DPI / 控制开关状态 |
| | `screen_click` / `screen_move` / `screen_drag` / `screen_scroll` / `screen_type` / `screen_key` | 鼠标键盘控制（安全开关默认关闭） |
| 诊断 | `vision_health` | 检查后端配置与连通性 |

**坐标系统**：所有定位/标注工具支持 `pixel`（像素，默认，实测最准）和 `norm`（0-1000 归一化）两种坐标系，越界坐标自动钳制并标记。

## 四、核心能力与实测数据

### 4.1 描述：语义准确，细节完整

[配图：docs/demo-annotate.png]

`describe_image` 支持 `detail` 参数（brief / balanced / detailed）和针对性 `question`。程序化交叉验证（PIL 像素级核对）结果：元素、颜色、文字、布局描述全部准确，连背景装饰元素都能注意到。

### 4.2 定位：几何图形零误差，复杂元素 refine 精修

[配图：docs/demo-locate-matrix.png]

`locate_object` 让模型输出目标边界框坐标，是 UI 自动化、元件检测的核心能力。实测精度：

| 目标类型 | 单次定位误差 | 说明 |
|---|---|---|
| 简单几何图形（圆形/色块） | **0 误差** | 对比度强、形状规则的几何体 |
| 圆角按钮/文字区域 | 20-70px | 边缘模糊、语义边界不确定 |
| refine 两阶段精修后 | **20px（y 方向完全精确）** | 粗定位→裁切放大→二次定位 |

`refine` 的原理：第一轮模型在整图上"估算"（受视觉 token 粒度限制，MiMo 对 200×100 图仅 18 个 image token），第二轮在放大的局部图上"细看"，消除整图量化误差。

### 4.3 OCR：中英文准确，位置精确

`ocr_image` 逐文本块提取，返回像素与归一化双坐标系 bbox。实测：文字内容 4/4 全部正确，位置框 x 方向近乎精确命中，y 偏差 ≤15px。中文场景经过专门修复（原测试图用 PIL 渲染中文为 notdef 乱码导致误判"中文 OCR 弱"，改用 GDI+ 生成后确认 MiMo V2.5 中文 OCR 正常，高铁站 LED 屏车次/站名均正确读取）。

### 4.4 图像处理：标注、裁切、放大

`annotate_image` 支持框/圆点/标签，程序化像素验证通过（框线命中、裁切尺寸正确）。`crop_image` 可按坐标裁切并边缘外扩，`zoom_region` 放大 1-8 倍用于细节识别。这组工具是"定位 → 细看"闭环的支撑。

### 4.5 自动异常扫描：PCB 元件检测实战

[配图建议：你自己的 PCB 扫描报告截图，全文最有说服力的素材]

`scan_anomalies` 专为"大图中找异常元件"设计，自动完成完整流程：

1. 把区域（默认全图）切成带重叠的块（最多 12 块），逐块 `locate_object` 收集候选（完整边界框 + rotation 角度）
2. 按空间位置合并去重（中心距离 / IoU）
3. `verify=true` 时：每个候选从**原图**高清裁切放大，客观化提问验证（是否歪斜/角度/丝印/类型），解析为结构化判定
4. 输出按"歪斜 → 不确定 → 正常"排序的报告

实测（200MP PCB 板，region 限定右下区域）：4 块扫描 → 3 候选 → 自动验证排除 2 个误报 → 锁定歪斜元件（A1142C LDO 所在区域，歪斜约 30°，丝印 5C）。总耗时约 2 分钟（8 次视觉调用）。

**方法论价值**：视觉模型对"歪斜"这类模糊目标的单次定位召回率低、角度估计不稳定（10°-35° 波动），但"多候选 + 逐点验证"的流程可以自动筛掉误报。**模型不完美，流程可以补**，这是视觉 Agent 工程的核心思路。

### 4.6 交互式图形推理：reason_graph 与 annotate_infer

`reason_graph` 提供完整的图形推理协议：`locate`（定位，支持 refine）→ `measure`（程序化测量距离/角度/面积，零 API 成本）→ `annotate`（固化为标注）→ `semantic`（语义记录）→ `hypothesis`（假设）→ `verify`（虚拟标注验证）→ `next`（下一步建议），session 跨轮传递状态。

`annotate_infer` 支持把标注几何（框/点/连线/箭头/圆/多边形/气泡）以坐标文本注入 prompt（原图零修改），或生成半透明叠加层后送模型推理，还有多轮修正（add/remove/move/resize）与自动框选（auto_boxes）。

实测案例：200MP PCB 图上虚拟标注 A1142C（歪斜 LDO）+ SL2.1S HUB + 供电箭头，模型正确推理出供电链路（TD1583→LDO→HUB VCC）与三大可靠性风险（歪斜焊点/散热/引脚）。

### 4.7 Computer Use：纯文本模型操控真实电脑

[配图：docs/demo-cu-before.png + docs/demo-cu-after.png]

完整闭环：

```
1. screen_capture → 截图
2. locate_object(截图, "确定按钮") → 坐标
3. screen_click(x, y) → 点击
4. screen_capture → 验证结果（视觉反馈循环）
```

实测完成了"打开 B 站 → 刷新 → 定位并点击第一个视频 → 验证进入播放页"全流程。中文输入走剪贴板粘贴，ASCII 直接按键，支持 ctrl+shift 组合键。

**安全设计**：`VISION_ALLOW_SCREEN_CONTROL` 默认关闭，关闭时所有控制类工具拒绝执行，只有截屏/信息可用，防止纯文本模型未经允许操控鼠标键盘。

## 五、踩坑记录（值得单独读的部分）

### 5.1 MCP 响应帧缺 id 字段，所有工具"超时"

**症状**：在严格校验的 MCP 客户端（HanaAgent）里，21 个工具调用全部超时；在 Codex 里一切正常。

**排查**：直接调视觉 API 正常（21.7s 返回，识别全对），桥进程活着，MCP 握手正常。最后写 probe 手动走 MCP 协议逐帧对话，发现工具**确实处理完了，响应也发回来了**，但响应帧里**没有 id 字段**。

```python
# 修复前：只有 result，没有 jsonrpc / id
return {"result": {"content": [...], "isError": is_error}}
```

MCP 协议里，客户端靠 id 把请求和响应配对。Codex 对 id 缺失宽容，所以一直没暴露；HanaAgent 严格校验，直接丢弃无 id 的响应，然后干等超时。**一个字段的缺失，表现为 21 个工具全线瘫痪。**

**教训**：宽松客户端会掩盖协议实现 bug。在 A 平台"能用"不等于协议正确。所有 MCP server 都应该用严格客户端过一遍。

### 5.2 MiMo 视觉 token 粒度：定位精度的根本限制

MiMo 对 200×100 图仅生成 18 个 image token，视觉粒度粗 + 无 grounding 专门训练，导致复杂元素定位偏差 20-70px。缓解手段（已内置）：refine 两阶段精修、VISION_SAMPLES 多次取样中位数、scan_anomalies 逐点验证、坐标钳制。

### 5.3 坐标格式实验：像素坐标最稳定

本地小模型（minicpm-v-4_5）定位实验，同图 3 种输出格式：像素坐标误差 40/60px（最佳）；0-1000 归一化格式混乱（LaTeX 尾巴 + 不合理坐标）；0-1 比例误差 130px（最差）。结论：**像素坐标是视觉模型最稳定的输出格式**，比例坐标假设被实证否定。

### 5.4 extract_json 鲁棒性

模型输出经常是"JSON + LaTeX 尾巴 / 文本前缀 + JSON / 代码块"混合，逐位置 raw_decode 取代贪婪正则后全部正确处理。

## 六、多后端实测对比（同一 benchmark）

| 模型 | describe/OCR | 定位 | 备注 |
|---|---|---|---|
| MiMo V2.5（云端） | 优秀 | 可用（refine 后 20px） | 当前推荐，单次 15-25s |
| qwen3.5-9b（本地 LM Studio） | 优秀（<15s） | **不可用**（误差 210px，颜色-位置映射错误） | 定位弱，refine 因粗框偏移失效 |
| minicpm-v-4_5（本地） | 优秀（bbox 准确） | 可用（x 精确，y 系统性偏上 ~90px） | 需 VISION_DISABLE_THINKING=1 |

结论：当前阶段，**云端 MiMo 是性价比最好的视觉后端**；本地模型做描述/OCR 可以，精细定位还不够。MiMo V2.5 Pro 在该 API 上不支持图片输入（404），选型时注意。

## 七、性能与边界

- **延迟**：MiMo 为推理型模型，单次视觉调用 15-25 秒，完整闭环（扫描/对比）约 2 分钟。这是当前最大短板
- **图片限制**：≤20MB，png/jpg/jpeg/webp/gif/bmp，大图自动降采样（长边 2600px 上限，可配置）
- **定位波动**：受采样影响，关键任务建议 VISION_SAMPLES=3 或 refine，圈画后人工核对
- **网络**：调用失败自动重试 1 次

## 八、快速上手

```bash
# 1. 克隆仓库
git clone https://github.com/zouyuanqing/vision-primitives-mcp.git

# 2. 配置环境变量（~/.codex/config.toml 或环境）
export VISION_API_BASE="https://api.xiaomimimo.com/v1"
export VISION_API_KEY="你的小米MiMo密钥"
export VISION_MODEL="mimo-v2.5"

# 3. 启动（MCP 服务器，接入 Codex / HanaAgent / 任意 MCP 客户端）
python vision_bridge_mcp.py

# 4. 验证
python vision_bridge_mcp.py --health
```

测试套件：`python test/run_tests.py`（116 项 mock 测试，不依赖真实 key）；`python test/e2e_mimo.py`（真实端到端）。

## 九、路线图与讨论

- 多后端支持已就绪（MiMo / LM Studio 本地模型），可继续扩展
- VISION_SAMPLES 多次取样提升定位稳定性
- Computer Use 更多真实场景（表单填写、自动化测试、GUI 回归）
- 视觉原语方法论进一步沉淀：measure / verify 循环的自动化

如果你也在做"给 Agent 装眼睛"的事，欢迎 star、提 issue 交流：

**仓库**：[github.com/zouyuanqing/vision-primitives-mcp](https://github.com/zouyuanqing/vision-primitives-mcp)

**相关关键词**：MCP、MCP 服务器、视觉 MCP、图像识别 API、LLM 视觉、DeepSeek 看图、Codex 视觉、纯文本模型、多模态、视觉原语、Visual Primitives、OCR、目标定位、坐标输出、PCB 检测、元件扫描、Computer Use、UI 自动化、小米 MiMo、MiMo V2.5
