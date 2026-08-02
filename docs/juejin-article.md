# 给纯文本模型装上眼睛：手写一个 MCP 视觉服务器，把定位误差从 70px 压到 20px

> 摘要：DeepSeek 这类推理模型很强，但天生没有视觉。我用小米 MiMo V2.5 写了一个零依赖、单文件的 MCP 视觉服务器，让纯文本模型能描述图片、输出坐标、OCR、圈画标注，甚至操控电脑。这篇文章记录完整实现思路、实测精度数据和一个藏在协议深处的 bug。

## 痛点：最强的推理模型，看不懂一张图

Codex、DeepSeek 这类对话模型在推理、写代码上已经很强，但有一个尴尬的事实：它们没有视觉。你丢一张截图给它，它只能回你"我无法查看图片"。

换多模态模型？成本高，还得迁移整个链路。本地部署视觉模型？显存、精度、工程复杂度都上去了。

但换个思路想：**模型不需要"看见"，它只需要"获得视觉信息"**。把看图这件事拆成可调用的工具，模型通过工具调用获得描述、坐标、文字，就等于有了眼睛。

这个思路有个名字：Visual Primitives（视觉原语），DeepSeek 在《Thinking with Visual Primitives》里提过。核心是把视觉任务拆成原语：描述、定位（输出坐标）、OCR、标注、裁切。**纯文本模型负责推理，视觉 API 负责看，MCP 负责把它们接起来。**

## 方案：MCP 服务器 + 小米 MiMo V2.5

架构很简单：

```
纯文本模型 (Codex / DeepSeek)
        │  MCP 协议 (stdio)
        ▼
vision-bridge-mcp（单文件 Python，零第三方运行时依赖）
        │  OpenAI 兼容 API
        ▼
小米 MiMo V2.5（全模态视觉模型）
```

- 视觉后端：小米 MiMo V2.5，OpenAI 兼容 API，支持图片输入，**能输出坐标**
- 实现：单文件 Python，只依赖 Pillow，手写 MCP stdio 协议（JSON-RPC 2.0 + Content-Length 帧），不需要任何 MCP SDK
- 21 个工具：describe / analyze / locate / ocr / annotate / crop / zoom / scan_anomalies / reason_graph / computer use 全套

仓库：[zouyuanqing/vision-primitives-mcp](https://github.com/zouyuanqing/vision-primitives-mcp)

## 它能干什么：四个核心能力

### 1. 描述 + OCR：语义准确，位置精确

[配图：docs/demo-annotate.png]

实测 OCR 中英文都准确，返回逐文本块的 bbox（像素 + 归一化坐标）。程序化验证过：文字位置 x 方向近乎精确命中，y 偏差 ≤15px，文字内容 4/4 全对。

### 2. 定位：几何图形零误差，复杂元素用 refine

[配图：docs/demo-locate-matrix.png]

locate_object 让模型输出目标坐标。实测：
- 简单几何图形（圆形、色块）：**完全精确**
- 圆角按钮、文字区域等复杂元素：偏移 20-70px（MiMo 视觉 token 粒度粗 + 无 grounding 专门训练）
- `refine=true` 两阶段精修（粗定位 → 裁切放大 → 二次定位）：误差从 **70px 降到 20px**，y 方向完全精确

### 3. 自动异常扫描：PCB 板歪斜元件，2 分钟出报告

[配图建议：你自己实测 PCB 时的扫描报告截图，这张图是全文最有说服力的素材，值得单独裁一张清晰的]

scan_anomalies 把大图切成带重叠的块，逐块定位候选，再从原图高清裁切逐点验证。实测 200MP PCB 板：4 块扫描 → 3 候选 → 自动验证排除 2 个误报 → 锁定歪斜元件（A1142C LDO，歪斜约 30°，丝印识别成功）。

这个功能的思路值得单独说：**视觉模型对"歪斜"的召回率低、角度不稳定，但用"多候选 + 逐点验证"的流程，可以自动把误报筛掉**。模型不完美，流程可以补。

### 4. Computer Use：纯文本模型操控真实电脑

[配图：docs/demo-cu-before.png / docs/demo-cu-after.png]

闭环：截图 → locate_object 定位按钮 → screen_click 点击 → 再截图验证。实测完成了"打开 B 站 → 刷新 → 定位并点击第一个视频 → 验证进入播放页"全流程。安全开关 VISION_ALLOW_SCREEN_CONTROL 默认关闭，防止模型未经允许操控鼠标键盘。

## 踩坑：MCP 响应帧缺 id 字段，所有工具"超时"

这个 bug 值得单独写，因为它藏得很深。

**症状**：在严格校验的 MCP 客户端（HanaAgent）里，21 个工具调用全部超时；但在 Codex 里一切正常。

**排查过程**：直接调视觉 API 一切正常（21.7s 返回，识别全对），桥进程活着，MCP 握手正常。最后写了个 probe 手动走 MCP 协议逐帧对话，发现：工具**确实处理完了，响应也发回来了**，但响应帧里**没有 id 字段**。

```python
# 修复前：只有 result，没有 jsonrpc / id
return {"result": {"content": [...], "isError": is_error}}
```

MCP 协议里，客户端靠 id 把请求和响应配对。Codex 对 id 缺失宽容，所以一直没暴露；HanaAgent 严格校验，直接丢弃无 id 的响应，然后干等超时。**一个字段的缺失，表现为 21 个工具全线瘫痪。**

```python
# 修复后
return {"jsonrpc": "2.0", "id": rid, "result": {...}}
```

教训：**宽松客户端会掩盖协议实现 bug**。在 A 平台"能用"不等于协议正确，换到严格客户端就现形。建议所有 MCP server 都用严格客户端过一遍。

## 边界与诚实结论

- **定位精度受采样波动影响**：几何图形稳定，复杂元素要开 refine 或 VISION_SAMPLES 多次取样取中位数
- **延迟**：MiMo 是推理型模型，单次视觉调用 15-25 秒，完整闭环约 2 分钟。这是当前最大短板
- **多模型实测对比**（同一 benchmark）：
  - MiMo V2.5：定位可用（refine 后 20px）
  - qwen3.5-9b（本地）：描述/OCR 优秀，**定位不可用**（误差 210px，颜色-位置映射错误）
  - minicpm-v-4_5（本地）：定位可用，但 y 方向系统性偏上 ~90px

结论：当前阶段，**云端 MiMo 是性价比最好的视觉后端**；本地模型做描述/OCR 可以，做精细定位还不行。

## 后续

- 多后端支持（已兼容 LM Studio 本地模型，可通过环境变量切换）
- VISION_SAMPLES 多次取样提升定位稳定性
- 更多 Computer Use 实战场景

项目持续迭代中，欢迎 star、提 issue，也欢迎各种花式用法交流。

**仓库**：[github.com/zouyuanqing/vision-primitives-mcp](https://github.com/zouyuanqing/vision-primitives-mcp)
